"""
Agent 核心模块：固定流水线模式
流程：解析意图 → 搜索 arXiv → 筛选摘要 → 逐篇阅读 → 生成回复

不再使用 LangGraph ReAct 循环，改为预定的流水线，
彻底消除"递归超限"问题。
"""

import json
from typing import Generator
from openai import OpenAI

from config import Config
from memory import ShortTermMemory, VectorStore
from prompts import RESPOND_PROMPT, INTENT_CLASSIFY_PROMPT, PARSE_INTENT_PROMPT
from tools import search_papers, screen_abstracts, read_paper, compare_papers, batch_download_papers, _cached_search_papers


class ResearchAgent:
    """封装好的 Agent 接口，给 Streamlit 调用"""

    def __init__(self):
        Config.validate()
        self.llm_client = OpenAI(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_BASE_URL,
        )
        self.vector_store = VectorStore()
        self.memory = ShortTermMemory()

    # ============================================================
    # 核心流水线
    # ============================================================
    def run(self, user_message: str) -> str:
        """
        固定流水线：
        1. 用 LLM 把用户意图翻译成英文搜索关键词
        2. 搜索 arXiv
        3. 用 LLM 筛选 Top 5
        4. 逐篇阅读（下载 PDF + 生成笔记）
        5. 多篇时横向对比
        6. 生成最终回复
        """
        memory = self.memory

        # 区分：是新话题还是追问
        memory.add_message("user", user_message)

        if not memory.user_goal:
            # user_goal 为空 → 一定是新话题
            is_new_topic = True
        else:
            # user_goal 已有值 → 用 LLM 判断是否换了话题
            classification = self._classify_intent(memory.user_goal, user_message)
            is_new_topic = (classification == "new_topic")

        if is_new_topic:
            # 如果之前有对话内容，先归档
            if memory.user_goal and (memory.papers or memory.paper_notes):
                # 论文笔记已经存入了 ChromaDB，这里只需清空当前会话
                pass
            memory.user_goal = user_message
            memory.papers = []
            memory.selected_papers = []
            memory.paper_notes = {}
            return self._run_pipeline(user_message)

        # 追问模式
        return self._run_followup(user_message)

    def run_stream(self, user_message: str, progress_callback=None) -> Generator[str, None, None]:
        """流式版本的 run，支持进度回调和流式输出"""
        memory = self.memory
        memory.add_message("user", user_message)

        if not memory.user_goal:
            is_new_topic = True
        else:
            classification = self._classify_intent(memory.user_goal, user_message)
            is_new_topic = (classification == "new_topic")

        if is_new_topic:
            if memory.user_goal and (memory.papers or memory.paper_notes):
                pass
            memory.user_goal = user_message
            memory.papers = []
            memory.selected_papers = []
            memory.paper_notes = {}

            # Step 1: 意图解析
            if progress_callback: progress_callback("🔍 正在分析搜索关键词...")
            query = self._parse_intent(user_goal=user_message)
            memory.add_message("system", f"[解析意图] 搜索关键词: {query}")

            # Step 2: 查知识库 + 搜索
            if progress_callback: progress_callback("📖 正在查知识库并搜索 arXiv...")
            kb_papers = self.vector_store.search(query, top_k=5)
            memory.kb_papers = [p for p in kb_papers if p.get("score", 0) > 0.3]
            result = search_papers(query, max_results=10)
            parsed = json.loads(result)
            if parsed.get("status") != "success":
                yield f"搜索论文失败: {parsed.get('message', '未知错误')}"
                return
            memory.papers = parsed.get("papers", [])
            if not memory.papers:
                yield self._no_papers_response()
                return

            # Step 3: 筛选
            if progress_callback: progress_callback("📋 正在筛选最相关的论文...")
            screen_result = screen_abstracts(memory.papers, user_message)
            screen_parsed = json.loads(screen_result)
            memory.selected_papers = screen_parsed.get("selected", memory.papers[:5])
            if not memory.selected_papers:
                yield self._no_papers_response()
                return

            # Step 4: 并行下载 + 逐篇阅读
            read_count = min(len(memory.selected_papers), Config.MAX_PAPERS_TO_READ)
            papers_to_read = memory.selected_papers[:read_count]

            if Config.READ_MODE == "abstract":
                downloaded = {}
            else:
                if progress_callback: progress_callback(f"📥 正在并行下载 {read_count} 篇 PDF...")
                downloaded = batch_download_papers(papers_to_read, max_workers=min(read_count, 5))

            for i, paper in enumerate(papers_to_read):
                if progress_callback: progress_callback(f"📖 正在阅读第 {i+1}/{read_count} 篇...")
                pid = paper.get("paper_id", "")
                pre_text = downloaded.get(pid, {}).get("full_text") if pid in downloaded else None
                if Config.READ_MODE == "abstract":
                    pre_text = None
                read_result = read_paper(paper, pre_downloaded_text=pre_text)
                read_parsed = json.loads(read_result)
                if read_parsed.get("status") == "success":
                    memory.paper_notes[read_parsed["paper_id"]] = read_parsed["notes"]
                    self.vector_store.add_paper(
                        paper_id=read_parsed["paper_id"],
                        title=paper.get("title", ""),
                        abstract=paper.get("summary", ""),
                        notes=read_parsed["notes"],
                        authors=", ".join(paper.get("authors", [])),
                        url=paper.get("url", ""),
                    )

            # Step 5: 对比
            comparison = ""
            if len(memory.paper_notes) >= 2:
                if progress_callback: progress_callback("🔬 正在横向对比论文...")
                compare_result = compare_papers(memory.paper_notes, user_message)
                compare_parsed = json.loads(compare_result)
                comparison = compare_parsed.get("comparison", "")

            # Step 6: 流式生成回复
            if progress_callback: progress_callback("✍️ 正在生成回复...")
            system_prompt = self._build_response_context(user_message, comparison)
            full_reply = []
            for token in self._stream_llm(system_prompt, "请生成总结回复"):
                full_reply.append(token)
                yield token
            memory.add_message("assistant", "".join(full_reply))

        else:
            # 追问模式
            yield from self._run_followup_stream(user_message, progress_callback)

    def _run_followup_stream(self, user_message: str, progress_callback=None) -> Generator[str, None, None]:
        """流式版本的追问模式"""
        memory = self.memory
        if progress_callback: progress_callback("🔍 正在查知识库...")
        rag_results = self.vector_store.search(user_message, top_k=3)

        context_parts = [f"用户目标：{memory.user_goal}"]
        if memory.selected_papers:
            papers_list = "\n".join([
                f"- [{p.get('title', 'N/A')}]({p.get('url', '')})"
                for p in memory.selected_papers[:5]
            ])
            context_parts.append(f"已讨论的论文：\n{papers_list}")
        if memory.paper_notes:
            for pid, notes in list(memory.paper_notes.items())[:3]:
                context_parts.append(f"\n论文 {pid} 笔记：\n{notes[:2000]}")
        if rag_results:
            rag_text = "\n".join([
                f"- {r.get('title', '')}: {r.get('notes', '')[:300]}"
                for r in rag_results
            ])
            context_parts.append(f"\n知识库相关论文：\n{rag_text}")

        system_prompt = RESPOND_PROMPT % (memory.user_goal, "\n".join(context_parts))
        if progress_callback: progress_callback("✍️ 正在生成回复...")
        full_reply = []
        for token in self._stream_llm(system_prompt, user_message):
            full_reply.append(token)
            yield token
        memory.add_message("assistant", "".join(full_reply))

    def _classify_intent(self, current_goal: str, new_message: str) -> str:
        """用 LLM 判断新消息是追问还是切换到了新话题"""
        system_prompt = INTENT_CLASSIFY_PROMPT % (current_goal, new_message)
        try:
            response = self.llm_client.chat.completions.create(
                model=Config.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": new_message},
                ],
                temperature=0.0,
                max_tokens=10,
            )
            result = response.choices[0].message.content.strip().lower()
            return "followup" if "followup" in result else "new_topic"
        except Exception:
            # LLM 挂了就当追问，走快速通道
            return "followup"

    def _run_pipeline(self, user_goal: str) -> str:
        """首次提问：走完整流水线"""
        memory = self.memory

        # Step 1: 意图解析 → 英文搜索词
        query = self._parse_intent(user_goal)
        memory.add_message("system", f"[解析意图] 搜索关键词: {query}")

        # Step 2: 先查知识库，再搜 arXiv
        kb_papers = self.vector_store.search(query, top_k=5)
        memory.kb_papers = [p for p in kb_papers if p.get("score", 0) > 0.3]  # 相似度 > 0.3 的才保留

        result = search_papers(query, max_results=10)
        parsed = json.loads(result)
        if parsed.get("status") != "success":
            return f"搜索论文失败: {parsed.get('message', '未知错误')}"
        memory.papers = parsed.get("papers", [])
        memory.add_message("system", f"[搜索] 找到 {len(memory.papers)} 篇论文")

        if not memory.papers:
            return self._no_papers_response()

        # Step 3: 筛选 Top 5
        screen_result = screen_abstracts(memory.papers, user_goal)
        screen_parsed = json.loads(screen_result)
        memory.selected_papers = screen_parsed.get("selected", memory.papers[:5])
        memory.add_message("system", f"[筛选] 选出 {len(memory.selected_papers)} 篇高相关论文")

        if not memory.selected_papers:
            return self._no_papers_response()

        # Step 4: 并行下载 PDF + 逐篇阅读
        read_count = min(len(memory.selected_papers), Config.MAX_PAPERS_TO_READ)
        papers_to_read = memory.selected_papers[:read_count]

        # 4a. 并行下载所有 PDF（仅摘要模式跳过下载）
        if Config.READ_MODE == "abstract":
            memory.add_message("system", f"[阅读] 仅摘要模式，跳过 PDF 下载")
            downloaded = {}
        else:
            memory.add_message("system", f"[阅读] 正在并行下载 {read_count} 篇论文的 PDF...")
            downloaded = batch_download_papers(papers_to_read, max_workers=min(read_count, 5))

        # 4b. 逐篇调 LLM 生成笔记（串行，避免 API 限流）
        for i, paper in enumerate(papers_to_read):
            memory.add_message("system", f"[阅读] 正在读第 {i+1}/{read_count} 篇...")
            pid = paper.get("paper_id", "")
            if Config.READ_MODE == "abstract":
                pre_text = None  # read_paper 会自己用摘要
            else:
                pre_text = downloaded.get(pid, {}).get("full_text") if pid in downloaded else None
            read_result = read_paper(paper, pre_downloaded_text=pre_text)
            read_parsed = json.loads(read_result)
            if read_parsed.get("status") == "success":
                paper_id = read_parsed.get("paper_id", "")
                memory.paper_notes[paper_id] = read_parsed.get("notes", "")
                # 存入知识库
                self.vector_store.add_paper(
                    paper_id=paper_id,
                    title=paper.get("title", ""),
                    abstract=paper.get("summary", ""),
                    notes=read_parsed.get("notes", ""),
                    authors=", ".join(paper.get("authors", [])),
                    url=paper.get("url", ""),
                )

        # Step 5: 多篇时横向对比
        comparison = ""
        if len(memory.paper_notes) >= 2:
            compare_result = compare_papers(memory.paper_notes, user_goal)
            compare_parsed = json.loads(compare_result)
            comparison = compare_parsed.get("comparison", "")

        # Step 6: 生成最终回复
        return self._generate_response(user_goal, comparison)

    def _parse_intent(self, user_goal: str) -> str:
        """用 LLM 把用户的中文意图翻译为英文搜索关键词"""
        try:
            response = self.llm_client.chat.completions.create(
                model=Config.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": PARSE_INTENT_PROMPT},
                    {"role": "user", "content": user_goal},
                ],
                temperature=0.1,
                max_tokens=100,
            )
            query = response.choices[0].message.content.strip()
            # 去除可能的引号
            query = query.strip('"').strip("'").strip()
            return query
        except Exception:
            # 降级：直接用中文
            return user_goal

    def _run_followup(self, user_message: str) -> str:
        """追问模式：用户对已读论文的补充提问"""
        memory = self.memory

        # 先查知识库
        rag_results = self.vector_store.search(user_message, top_k=3)

        context_parts = [f"用户目标：{memory.user_goal}"]
        if memory.selected_papers:
            papers_list = "\n".join([
                f"- [{p.get('title', 'N/A')}]({p.get('url', '')})"
                for p in memory.selected_papers[:5]
            ])
            context_parts.append(f"已讨论的论文：\n{papers_list}")

        if memory.paper_notes:
            for pid, notes in list(memory.paper_notes.items())[:3]:
                context_parts.append(f"\n论文 {pid} 笔记：\n{notes[:2000]}")

        if rag_results:
            rag_text = "\n".join([
                f"- {r.get('title', '')}: {r.get('notes', '')[:300]}"
                for r in rag_results
            ])
            context_parts.append(f"\n知识库相关论文：\n{rag_text}")

        system_prompt = RESPOND_PROMPT % (memory.user_goal, "\n".join(context_parts))

        try:
            response = self.llm_client.chat.completions.create(
                model=Config.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.5,
                max_tokens=2048,
            )
            reply = response.choices[0].message.content
        except Exception:
            reply = self._fallback_response()

        memory.add_message("assistant", reply)
        return reply

    def _build_response_context(self, user_goal: str, comparison: str = "") -> str:
        """构建回复所需的上下文（供流式和非流式共用）"""
        memory = self.memory
        context_parts = [f"用户目标：{user_goal}"]

        if memory.selected_papers:
            papers_list = "\n".join([
                f"- **[{p.get('title', 'N/A')}]({p.get('url', '')})**\n"
                f"  作者: {', '.join(p.get('authors', [])[:3])} | 日期: {p.get('published', 'N/A')}\n"
                f"  摘要: {p.get('summary', '')[:300]}..."
                for p in memory.selected_papers[:5]
            ])
            context_parts.append(f"筛选出的高相关论文：\n{papers_list}")

        if memory.paper_notes:
            for pid, notes in memory.paper_notes.items():
                context_parts.append(f"\n--- {pid} ---\n{notes[:2000]}")

        if comparison:
            context_parts.append(f"\n横向对比：\n{comparison[:2000]}")

        return RESPOND_PROMPT % (user_goal, "\n".join(context_parts))

    def _stream_llm(self, system_prompt: str, user_message: str,
                    temperature: float = 0.5, max_tokens: int = 2048) -> Generator[str, None, None]:
        """流式调用 LLM，逐块 yield token"""
        try:
            stream = self.llm_client.chat.completions.create(
                model=Config.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception:
            yield self._fallback_response()

    def _generate_response(self, user_goal: str, comparison: str = "") -> str:
        """生成最终的汇总回复（非流式，兼容旧调用）"""
        memory = self.memory
        system_prompt = self._build_response_context(user_goal, comparison)
        try:
            response = self.llm_client.chat.completions.create(
                model=Config.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "请生成总结回复"},
                ],
                temperature=0.5,
                max_tokens=2048,
            )
            reply = response.choices[0].message.content
        except Exception:
            reply = self._fallback_response()
        memory.add_message("assistant", reply)
        return reply

    def _no_papers_response(self) -> str:
        return (
            "抱歉，没有搜索到相关论文。\n\n"
            "建议：\n"
            "1. 用更宽泛的关键词重试\n"
            "2. 换个角度描述你的研究主题\n"
            "3. 确认网络连接正常（需要访问 arXiv API）\n\n"
            "你可以告诉我你想调整的方向，我来重新搜索。"
        )

    def _fallback_response(self) -> str:
        """LLM 调用失败时的降级回复"""
        memory = self.memory
        parts = []
        if memory.selected_papers:
            parts.append("## 📚 为你找到以下相关论文：\n")
            for p in memory.selected_papers[:5]:
                parts.append(f"- **[{p.get('title', 'N/A')}]({p.get('url', '')})**")
                parts.append(f"  - 作者: {', '.join(p.get('authors', [])[:3])}")
                parts.append(f"  - 日期: {p.get('published', 'N/A')}")
                parts.append(f"  - 摘要: {p.get('summary', '')[:200]}...")
                parts.append("")
        return "\n".join(parts) if parts else "未能生成回答，请重试。"

    # ============================================================
    # 记忆管理
    # ============================================================
    def reset(self):
        """重置对话"""
        self.memory = ShortTermMemory()

    def dump_memory(self) -> dict:
        """导出 memory 状态（用于对话历史保存）"""
        return self.memory.to_dict()

    def load_memory(self, data: dict):
        """恢复 memory 状态（用于对话历史恢复）"""
        self.memory.from_dict(data)