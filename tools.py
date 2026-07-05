"""
工具模块：搜索、筛选、阅读、对比论文
"""

import json
import os
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests
from openai import OpenAI

from config import Config
from prompts import SCREEN_PROMPT, SUMMARIZE_PROMPT, COMPARE_PROMPT


def _get_llm_client() -> OpenAI:
    return OpenAI(api_key=Config.DEEPSEEK_API_KEY, base_url=Config.DEEPSEEK_BASE_URL)


def _call_llm(system_prompt: str, user_prompt: str, temperature: float = None) -> str:
    if temperature is None:
        temperature = Config.LLM_TEMPERATURE
    client = _get_llm_client()
    try:
        response = client.chat.completions.create(
            model=Config.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=4096,
        )
        return response.choices[0].message.content
    except Exception as e:
        return json.dumps({"error": f"LLM 调用失败: {e}"})


def _retry_request(url: str, params: dict = None, timeout: int = 30, max_retries: int = 2,
                   stream: bool = False) -> requests.Response:
    """带重试的 HTTP 请求，使用指数退避"""
    import time
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            if params:
                resp = requests.get(url, params=params, timeout=timeout, stream=stream)
            else:
                resp = requests.get(url, timeout=timeout, stream=stream)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 ** attempt  # 1s, 2s, 4s...
                time.sleep(wait)
    raise last_error


# ============================================================
# 搜索结果缓存
# ============================================================
_search_cache: dict[str, tuple[float, str]] = {}  # {query: (timestamp, result_json)}


def _cached_search_papers(query: str, max_results: int = 10) -> str:
    """带缓存的论文搜索"""
    cache_key = f"{query}:{max_results}"
    now = time.time()
    if cache_key in _search_cache:
        cached_time, cached_result = _search_cache[cache_key]
        if now - cached_time < Config.SEARCH_CACHE_TTL:
            return cached_result
    result = search_papers(query, max_results)
    _search_cache[cache_key] = (now, result)
    return result


def _clean_json(text: str) -> str:
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    return text.strip()


# ============================================================
# 工具 1: 搜索论文
# ============================================================
def search_papers(query: str, max_results: int = 10) -> str:
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": min(max_results, 10),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    try:
        resp = _retry_request(base_url, params=params, timeout=30)
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        papers = []
        for entry in root.findall("atom:entry", ns):
            paper_id = entry.find("atom:id", ns).text.strip()
            arxiv_id = paper_id.split("/abs/")[-1]
            title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
            summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
            authors = [author.find("atom:name", ns).text for author in entry.findall("atom:author", ns)]
            published = entry.find("atom:published", ns).text[:10]
            papers.append({
                "paper_id": arxiv_id, "title": title, "summary": summary[:500],
                "authors": authors, "published": published,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            })
        if not papers:
            return json.dumps({"status": "empty", "message": f"未找到相关论文，建议换关键词。", "papers": []}, ensure_ascii=False)
        return json.dumps({"status": "success", "count": len(papers), "papers": papers}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


# ============================================================
# 工具 2: 摘要筛选
# ============================================================
def screen_abstracts(papers: list[dict], user_goal: str) -> str:
    if not papers:
        return json.dumps({"status": "empty", "selected": []}, ensure_ascii=False)
    papers_for_llm = [{"paper_id": p["paper_id"], "title": p["title"], "summary": p["summary"][:300]} for p in papers]
    user_prompt = json.dumps(papers_for_llm, ensure_ascii=False)
    system_prompt = SCREEN_PROMPT % (user_goal, "见用户消息中的 JSON")
    response = _call_llm(system_prompt, user_prompt, temperature=0.1)
    cleaned = _clean_json(response)
    try:
        selected = json.loads(cleaned)
        paper_map = {p["paper_id"]: p for p in papers}
        enriched = []
        for item in selected[:Config.MAX_PAPERS_TO_READ]:
            pid = item.get("paper_id", "")
            if pid in paper_map:
                enriched.append({**paper_map[pid], "score": item.get("score", 0), "reason": item.get("reason", "")})
        return json.dumps({"status": "success", "count": len(enriched), "selected": enriched}, ensure_ascii=False)
    except json.JSONDecodeError:
        return json.dumps({"status": "fallback", "message": "LLM 解析失败，取前 3 篇", "selected": papers[:3]}, ensure_ascii=False)


# ============================================================
# 辅助：下载单篇论文文本（不调 LLM）
# ============================================================
def _download_paper_text(paper: dict) -> dict:
    """下载 PDF 并提取文本，返回 {"paper_id", "full_text", "pdf_downloaded", "error"}"""
    paper_id = paper.get("paper_id", "unknown")
    pdf_url = paper.get("pdf_url", "")
    summary = paper.get("summary", "")
    full_text = ""
    pdf_downloaded = False

    if not pdf_url:
        return {"paper_id": paper_id, "full_text": f"[无 PDF 链接，使用摘要] {summary}",
                "pdf_downloaded": False, "error": None}

    try:
        pdf_path = os.path.join(Config.PAPERS_DIR, f"{paper_id}.pdf")
        resp = _retry_request(pdf_url, timeout=60, stream=True)
        if resp.status_code == 200 and "application/pdf" in resp.headers.get("content-type", ""):
            with open(pdf_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            pdf_downloaded = True
            try:
                import fitz
                doc = fitz.open(pdf_path)
                full_text = "".join(page.get_text() for page in doc)[:30000]
                doc.close()
            except Exception as e:
                full_text = f"[PDF 解析失败，使用摘要] {summary}"
                return {"paper_id": paper_id, "full_text": full_text,
                        "pdf_downloaded": pdf_downloaded, "error": str(e)}
        else:
            full_text = f"[PDF 下载失败，使用摘要] {summary}"
            return {"paper_id": paper_id, "full_text": full_text,
                    "pdf_downloaded": False, "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        full_text = f"[下载异常，使用摘要] {summary}"
        return {"paper_id": paper_id, "full_text": full_text,
                "pdf_downloaded": False, "error": str(e)}

    return {"paper_id": paper_id, "full_text": full_text,
            "pdf_downloaded": pdf_downloaded, "error": None}


def batch_download_papers(papers: list[dict], max_workers: int = 5) -> dict[str, dict]:
    """并行下载多篇论文的 PDF 并提取文本，返回 {paper_id: download_result}"""
    results = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(papers))) as executor:
        futures = {executor.submit(_download_paper_text, p): p["paper_id"] for p in papers}
        for future in as_completed(futures):
            try:
                result = future.result()
                results[result["paper_id"]] = result
            except Exception as e:
                pid = futures[future]
                results[pid] = {"paper_id": pid, "full_text": f"[下载异常] {e}",
                                "pdf_downloaded": False, "error": str(e)}
    return results


# ============================================================
# 工具 3: 阅读论文
# ============================================================
def read_paper(paper: dict, pre_downloaded_text: str = None) -> str:
    """阅读单篇论文：调 LLM 生成笔记。如果提供了 pre_downloaded_text 则跳过下载步骤。"""
    paper_id = paper.get("paper_id", "unknown")
    title = paper.get("title", "无标题")
    authors = ", ".join(paper.get("authors", [])[:5])
    published = paper.get("published", "未知")
    summary = paper.get("summary", "")
    pdf_downloaded = False

    if pre_downloaded_text is not None:
        # 使用并行下载阶段已获取的文本
        full_text = pre_downloaded_text
        # 判断是否真的下载了 PDF（不是降级文本）
        pdf_downloaded = not full_text.startswith("[")
    else:
        # 原有逻辑：串行下载
        pdf_url = paper.get("pdf_url", "")
        full_text = ""
        if pdf_url:
            try:
                pdf_path = os.path.join(Config.PAPERS_DIR, f"{paper_id}.pdf")
                resp = requests.get(pdf_url, timeout=60, stream=True)
                if resp.status_code == 200 and "application/pdf" in resp.headers.get("content-type", ""):
                    with open(pdf_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    pdf_downloaded = True
                    try:
                        import fitz
                        doc = fitz.open(pdf_path)
                        full_text = "".join(page.get_text() for page in doc)[:30000]
                        doc.close()
                    except Exception:
                        full_text = f"[PDF 解析失败，使用摘要] {summary}"
                else:
                    full_text = f"[PDF 下载失败，使用摘要] {summary}"
            except Exception:
                full_text = f"[下载异常，使用摘要] {summary}"
        else:
            full_text = f"[无 PDF 链接，使用摘要] {summary}"

    system_prompt = SUMMARIZE_PROMPT % (title, authors, published, full_text)
    notes = _call_llm(system_prompt, "请生成结构化笔记。", temperature=0.3)
    return json.dumps({
        "status": "success", "paper_id": paper_id, "title": title,
        "pdf_downloaded": pdf_downloaded, "notes": notes, "url": paper.get("url", ""),
    }, ensure_ascii=False)


# ============================================================
# 工具 4: 横向对比
# ============================================================
def compare_papers(paper_notes: dict[str, str], user_goal: str) -> str:
    if len(paper_notes) < 2:
        return json.dumps({"status": "insufficient", "comparison": "需要至少 2 篇论文才能对比。"}, ensure_ascii=False)
    notes_json = json.dumps(paper_notes, ensure_ascii=False)
    system_prompt = COMPARE_PROMPT % (user_goal, notes_json)
    comparison = _call_llm(system_prompt, "请对比分析。", temperature=0.3)
    return json.dumps({"status": "success", "paper_count": len(paper_notes), "comparison": comparison}, ensure_ascii=False)