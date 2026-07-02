"""
工具模块：Agent 可调用的所有工具函数
- search_papers: 调用 arXiv API 搜索论文
- screen_abstracts: 用 LLM 评估摘要相关度
- read_paper: 下载 PDF + 提取全文 + 生成笔记
- compare_papers: 横向对比多篇论文
- rag_search: 从长期知识库检索
"""

import json
import os
import re
import time
import xml.etree.ElementTree as ET
from typing import Any

import requests
from openai import OpenAI

from config import Config
from memory import ShortTermMemory, VectorStore
from prompts import COMPARE_PROMPT, SCREEN_PROMPT, SUMMARIZE_PROMPT


# ============================================================
# 初始化 LLM 客户端
# ============================================================
def _get_llm_client() -> OpenAI:
    """获取 DeepSeek API 客户端（兼容 OpenAI SDK）"""
    return OpenAI(
        api_key=Config.DEEPSEEK_API_KEY,
        base_url=Config.DEEPSEEK_BASE_URL,
    )


def _call_llm(system_prompt: str, user_prompt: str, temperature: float = None) -> str:
    """
    调用 DeepSeek API，返回文本响应
    """
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
        return json.dumps({"error": f"LLM 调用失败: {str(e)}"})


def _clean_json(text: str) -> str:
    """从 LLM 返回文本中提取纯 JSON"""
    # 去掉 Markdown 代码块标记
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    return text.strip()


# ============================================================
# 工具 1: 搜索论文 (arXiv API)
# ============================================================
def search_papers(query: str, max_results: int = 5) -> str:
    """
    调用 arXiv API 搜索论文
    参数:
        query: 搜索关键词（支持 AND, OR 等运算符）
        max_results: 最大返回数量（默认 5）
    返回:
        JSON 字符串：论文列表 [{id, title, summary, authors, published, url}, ...]
    """
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": min(max_results, 10),  # 限制最多 10 篇
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    try:
        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()

        # 解析 Atom XML
        root = ET.fromstring(resp.text)
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        papers = []
        for entry in root.findall("atom:entry", ns):
            paper_id = entry.find("atom:id", ns).text.strip()
            # 提取 arXiv ID（去掉 URL 前缀）
            arxiv_id = paper_id.split("/abs/")[-1]

            title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
            summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")

            authors = [
                author.find("atom:name", ns).text
                for author in entry.findall("atom:author", ns)
            ]

            published = entry.find("atom:published", ns).text[:10]  # YYYY-MM-DD

            papers.append({
                "paper_id": arxiv_id,
                "title": title,
                "summary": summary[:500],  # 摘要截断 500 字
                "authors": authors,
                "published": published,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            })

        if not papers:
            return json.dumps({
                "status": "empty",
                "message": f"未找到与 '{query}' 相关的论文，建议尝试更宽泛的关键词。",
                "papers": []
            }, ensure_ascii=False)

        return json.dumps({
            "status": "success",
            "count": len(papers),
            "papers": papers,
        }, ensure_ascii=False)

    except requests.RequestException as e:
        return json.dumps({
            "status": "error",
            "message": f"arXiv API 请求失败: {str(e)}"
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"搜索过程出错: {str(e)}"
        }, ensure_ascii=False)


# ============================================================
# 工具 2: 摘要筛选
# ============================================================
def screen_abstracts(papers: list[dict], user_goal: str) -> str:
    """
    用 LLM 评估每篇论文与用户目标的相关度，返回 Top 3-5
    参数:
        papers: 论文列表
        user_goal: 用户的研究目标
    返回:
        JSON 字符串：高相关论文列表
    """
    if not papers:
        return json.dumps({
            "status": "empty",
            "message": "没有论文可供筛选。",
            "selected": []
        }, ensure_ascii=False)

    # 准备输入：只传 id、标题、摘要，减少 token
    papers_for_llm = []
    for p in papers:
        papers_for_llm.append({
            "paper_id": p.get("paper_id", ""),
            "title": p.get("title", ""),
            "summary": p.get("summary", "")[:300],
        })

    user_prompt = json.dumps(papers_for_llm, ensure_ascii=False)
    system_prompt = SCREEN_PROMPT.format(
        user_goal=user_goal,
        papers_json="{papers_json}"  # 占位，实际在 user_prompt 中
    )
    # 手动构造完整的 system prompt
    system_prompt = SCREEN_PROMPT.replace("{user_goal}", user_goal).replace(
        "{papers_json}", "（见用户消息中的 JSON）"
    )

    response = _call_llm(system_prompt, user_prompt, temperature=0.1)
    cleaned = _clean_json(response)

    try:
        selected = json.loads(cleaned)
        # 补全论文完整信息
        paper_map = {p["paper_id"]: p for p in papers}
        enriched = []
        for item in selected[:Config.MAX_PAPERS_TO_READ]:
            pid = item.get("paper_id", "")
            if pid in paper_map:
                full = paper_map[pid]
                enriched.append({
                    **full,
                    "score": item.get("score", 0),
                    "reason": item.get("reason", ""),
                })

        return json.dumps({
            "status": "success",
            "count": len(enriched),
            "selected": enriched,
        }, ensure_ascii=False)

    except json.JSONDecodeError:
        # LLM 返回格式不对，降级：取前 3 篇
        return json.dumps({
            "status": "fallback",
            "message": "LLM 评分解析失败，返回原始搜索结果的前 3 篇。",
            "selected": papers[:3],
        }, ensure_ascii=False)


# ============================================================
# 工具 3: 阅读论文（下载 PDF + 提取全文 + LLM 总结）
# ============================================================
def read_paper(paper: dict) -> str:
    """
    下载并深度阅读一篇论文
    参数:
        paper: 论文信息字典（包含 paper_id, title, authors, pdf_url 等）
    返回:
        JSON 字符串：结构化笔记
    """
    paper_id = paper.get("paper_id", "unknown")
    title = paper.get("title", "无标题")
    authors = ", ".join(paper.get("authors", [])[:5])
    published = paper.get("published", "未知")
    pdf_url = paper.get("pdf_url", "")
    summary = paper.get("summary", "")

    # 尝试下载 PDF
    full_text = ""
    pdf_downloaded = False

    if pdf_url:
        try:
            pdf_path = os.path.join(Config.PAPERS_DIR, f"{paper_id}.pdf")
            resp = requests.get(pdf_url, timeout=60, stream=True)
            if resp.status_code == 200 and "application/pdf" in resp.headers.get("content-type", ""):
                with open(pdf_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                pdf_downloaded = True

                # 用 PyMuPDF 提取文本
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(pdf_path)
                    full_text = ""
                    for page in doc:
                        full_text += page.get_text()
                    doc.close()
                    # 截断过长文本（DeepSeek 上下文够大，但也不要太浪费）
                    full_text = full_text[:30000]
                except ImportError:
                    full_text = f"[PyMuPDF 未安装，无法提取 PDF 文本。论文摘要：{summary}]"
                except Exception as e:
                    full_text = f"[PDF 解析失败: {e}。论文摘要：{summary}]"
            else:
                full_text = f"[PDF 下载失败，状态码: {resp.status_code}。论文摘要：{summary}]"
        except Exception as e:
            full_text = f"[PDF 下载异常: {e}。论文摘要：{summary}]"
    else:
        full_text = f"[无 PDF 链接。论文摘要：{summary}]"

    # 用 LLM 生成结构化笔记
    system_prompt = SUMMARIZE_PROMPT.format(
        title=title,
        authors=authors,
        published=published,
        content="{content}"
    ).replace("{content}", full_text)

    user_prompt = "请根据上述论文内容生成结构化笔记。"

    notes = _call_llm(system_prompt, user_prompt, temperature=0.3)

    result = {
        "status": "success",
        "paper_id": paper_id,
        "title": title,
        "pdf_downloaded": pdf_downloaded,
        "notes": notes,
        "url": paper.get("url", ""),
    }

    return json.dumps(result, ensure_ascii=False)


# ============================================================
# 工具 4: 横向对比
# ============================================================
def compare_papers(paper_notes: dict[str, str], user_goal: str) -> str:
    """
    横向对比多篇已读论文
    参数:
        paper_notes: {paper_id: notes_text, ...}
        user_goal: 用户的研究目标
    返回:
        JSON 字符串：对比分析报告
    """
    if len(paper_notes) < 2:
        return json.dumps({
            "status": "insufficient",
            "message": "至少需要 2 篇论文笔记才能进行对比。",
            "comparison": ""
        }, ensure_ascii=False)

    notes_json = json.dumps(paper_notes, ensure_ascii=False)
    system_prompt = COMPARE_PROMPT.format(
        user_goal=user_goal,
        notes_json="{notes_json}"
    ).replace("{notes_json}", notes_json)

    user_prompt = "请根据上述论文笔记进行横向对比分析。"

    comparison = _call_llm(system_prompt, user_prompt, temperature=0.3)

    return json.dumps({
        "status": "success",
        "paper_count": len(paper_notes),
        "comparison": comparison,
    }, ensure_ascii=False)


# ============================================================
# 工具 5: RAG 检索（查长期知识库）
# ============================================================
def rag_search(query: str, vector_store: VectorStore) -> str:
    """
    从 ChromaDB 长期知识库中检索相关论文
    参数:
        query: 检索查询
        vector_store: VectorStore 实例
    返回:
        JSON 字符串：检索结果
    """
    results = vector_store.search(query, top_k=5)

    if not results:
        return json.dumps({
            "status": "empty",
            "message": "知识库中没有找到相关论文。建议先用搜索工具找新论文。",
            "results": []
        }, ensure_ascii=False)

    return json.dumps({
        "status": "success",
        "count": len(results),
        "results": results,
    }, ensure_ascii=False)


# ============================================================
# 工具执行入口（Agent 调用此函数来执行工具）
# ============================================================
def execute_tool(
    action: str,
    action_input: dict[str, Any],
    memory: ShortTermMemory,
    vector_store: VectorStore,
) -> str:
    """
    根据 action 名称执行对应工具，并更新 memory
    返回: JSON 字符串（工具执行结果）
    """
    try:
        if action == "search_papers":
            query = action_input.get("query", "")
            max_results = action_input.get("max_results", 5)
            result = search_papers(query, max_results)

            # 更新 memory
            parsed = json.loads(result)
            if parsed.get("status") == "success":
                memory.papers = parsed.get("papers", [])
            memory.observation = result

        elif action == "screen_abstracts":
            papers = memory.papers if memory.papers else action_input.get("papers", [])
            user_goal = memory.user_goal
            result = screen_abstracts(papers, user_goal)

            parsed = json.loads(result)
            if parsed.get("status") in ("success", "fallback"):
                memory.selected_papers = parsed.get("selected", [])
            memory.observation = result

        elif action == "read_paper":
            paper_id = action_input.get("paper_id", "")
            # 从 selected_papers 中找到对应论文
            paper = next(
                (p for p in memory.selected_papers if p.get("paper_id") == paper_id),
                None
            )
            if not paper:
                # 可能直接从 papers 中找
                paper = next(
                    (p for p in memory.papers if p.get("paper_id") == paper_id),
                    {"paper_id": paper_id, "title": paper_id, "authors": [], "summary": ""}
                )

            result = read_paper(paper)

            parsed = json.loads(result)
            if parsed.get("status") == "success":
                memory.paper_notes[paper_id] = parsed.get("notes", "")
                # 存入长期知识库
                vector_store.add_paper(
                    paper_id=paper_id,
                    title=paper.get("title", ""),
                    abstract=paper.get("summary", ""),
                    notes=parsed.get("notes", ""),
                    authors=", ".join(paper.get("authors", [])),
                    url=paper.get("url", ""),
                )
            memory.observation = result

        elif action == "compare_papers":
            result = compare_papers(memory.paper_notes, memory.user_goal)
            memory.observation = result

        elif action == "rag_search":
            query = action_input.get("query", memory.user_goal)
            result = rag_search(query, vector_store)
            memory.observation = result

        else:
            result = json.dumps({
                "status": "error",
                "message": f"未知的工具: {action}"
            }, ensure_ascii=False)

        return result

    except Exception as e:
        error_result = json.dumps({
            "status": "error",
            "message": f"工具 '{action}' 执行失败: {str(e)}"
        }, ensure_ascii=False)
        memory.observation = error_result
        return error_result