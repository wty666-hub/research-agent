"""
Agent 核心模块：基于 LangGraph 的 ReAct 状态图
节点：think_node, tool_node, respond_node
路由：根据 LLM 决策选择下一步
"""

import json
import re
from typing import Any, Literal

from langgraph.graph import StateGraph, END
from openai import OpenAI

from config import Config
from memory import ShortTermMemory, VectorStore
from prompts import THINKING_PROMPT, RESPOND_PROMPT
from tools import execute_tool


# ============================================================
# 状态定义
# ============================================================
class AgentState(dict):
    """
    LangGraph 状态，需要在图中流转的所有数据
    继承 dict 是因为 LangGraph 要求状态是可序列化的字典
    """
    memory: ShortTermMemory
    vector_store: VectorStore
    llm_client: OpenAI
    next_action: str
    final_response: str

    def __init__(
        self,
        memory: ShortTermMemory,
        vector_store: VectorStore,
        llm_client: OpenAI,
    ):
        super().__init__()
        self.memory = memory
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.next_action = ""
        self.final_response = ""

    def to_dict(self) -> dict:
        """序列化（LangGraph 要求）"""
        return {
            "next_action": self.next_action,
            "final_response": self.final_response,
        }


# ============================================================
# 节点 1: think_node — LLM 决策
# ============================================================
def think_node(state: AgentState) -> AgentState:
    """
    核心决策节点：
    1. 检查 step_count 是否超限
    2. 调用 LLM 分析当前状态，决定下一步 action
    3. 返回更新后的 state
    """
    memory = state.memory

    # 检查是否超过最大步数
    if memory.is_over_limit():
        state.next_action = "respond"
        memory.add_message("system", "已达到最大步数限制，强制结束并返回结果。")
        return state

    # 构造 LLM 输入
    state_context = memory.get_state_context()
    system_prompt = THINKING_PROMPT.format(state_context=state_context)

    # 构造对话历史作为 user prompt
    history_text = ""
    for msg in memory.messages[-10:]:  # 只取最近 10 条，避免 token 过长
        role = msg["role"]
        content = msg["content"][:500]  # 每条截断
        history_text += f"[{role}]: {content}\n"

    user_prompt = f"## 对话历史\n{history_text}\n\n请根据当前状态决定下一步操作。"

    # 调用 LLM
    try:
        response = state.llm_client.chat.completions.create(
            model=Config.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=1024,
        )
        llm_output = response.choices[0].message.content

        # 解析 LLM 决策
        decision = _parse_decision(llm_output)
        state.next_action = decision.get("action", "respond")
        thought = decision.get("thought", "")

        # 将决策存入 memory（以 action_input 的形式传给 tool_node）
        memory._last_action_input = decision.get("action_input", {})

        memory.add_message("system", f"[思考] {thought}")
        memory.add_message("system", f"[决策] 选择执行: {state.next_action}")

    except Exception as e:
        # API 调用失败，直接进入 respond
        state.next_action = "respond"
        memory.add_message("system", f"[错误] LLM 决策失败: {e}，转到 respond。")

    memory.increment_step()
    return state


def _parse_decision(llm_output: str) -> dict[str, Any]:
    """
    从 LLM 输出中解析决策 JSON
    支持多种格式：纯 JSON、带 Markdown 代码块等
    """
    try:
        # 尝试直接解析
        cleaned = llm_output.strip()
        # 去掉 Markdown 代码块标记
        cleaned = re.sub(r"```json\s*", "", cleaned)
        cleaned = re.sub(r"```\s*", "", cleaned)

        # 提取 JSON 对象
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except (json.JSONDecodeError, AttributeError):
        pass

    # 返回默认决策
    return {
        "thought": "无法解析 LLM 输出，默认返回 respond。",
        "action": "respond",
        "action_input": {},
    }


# ============================================================
# 节点 2: tool_node — 执行工具
# ============================================================
def tool_node(state: AgentState) -> AgentState:
    """
    根据 think_node 的决策，执行对应的工具
    """
    memory = state.memory
    action = state.next_action
    action_input = getattr(memory, "_last_action_input", {})

    # 特殊处理：respond 不执行工具
    if action == "respond":
        return state

    # 特殊处理：read_paper 可能需要批量读取
    if action == "read_paper":
        # 如果指定了具体 paper_id，只读那篇
        # 如果没指定，自动读所有 selected_papers 中还没读过的
        if "paper_id" in action_input and action_input["paper_id"]:
            # 读指定论文
            result = execute_tool(action, action_input, memory, state.vector_store)
            memory.add_message("tool", result)
        else:
            # 自动读所有未读的筛选论文
            unread_papers = [
                p for p in memory.selected_papers
                if p.get("paper_id") not in memory.paper_notes
            ]
            if not unread_papers:
                memory.observation = json.dumps({
                    "status": "info",
                    "message": "所有筛选论文已读完。"
                }, ensure_ascii=False)
            else:
                # 每次只读一篇（LangGraph 节点是原子操作）
                paper = unread_papers[0]
                result = execute_tool(
                    "read_paper",
                    {"paper_id": paper.get("paper_id")},
                    memory,
                    state.vector_store
                )
                memory.add_message("tool", result)
                # 如果还有未读的，标记继续 read
                if len(unread_papers) > 1:
                    memory.observation += f"\n还有 {len(unread_papers) - 1} 篇待读。"
    else:
        # 其他工具：直接执行
        result = execute_tool(action, action_input, memory, state.vector_store)
        memory.add_message("tool", result)

    return state


# ============================================================
# 节点 3: respond_node — 生成最终回答
# ============================================================
def respond_node(state: AgentState) -> AgentState:
    """
    终结节点：汇总所有信息，生成最终回答给用户
    """
    memory = state.memory

    # 检查是否有实质内容
    has_papers = bool(memory.papers)
    has_notes = bool(memory.paper_notes)

    if not has_papers and not has_notes:
        # 没搜到任何东西
        state.final_response = (
            "抱歉，我尝试搜索了相关论文但没有找到合适的结果。\n\n"
            "建议：\n"
            "1. 试试更宽泛的关键词\n"
            "2. 换个角度描述你的研究主题\n"
            "3. 确认网络连接正常（需要访问 arXiv API）\n\n"
            "你可以告诉我你想调整的方向，我来重新搜索。"
        )
        return state

    # 构造上下文给 LLM 生成回答
    context_parts = [f"用户目标：{memory.user_goal}"]

    if memory.selected_papers:
        papers_list = "\n".join([
            f"- [{p.get('title', 'N/A')}]({p.get('url', '')}) (相关度: {p.get('score', 'N/A')}分)"
            for p in memory.selected_papers[:5]
        ])
        context_parts.append(f"筛选出的高相关论文：\n{papers_list}")

    if memory.paper_notes:
        for pid, notes in memory.paper_notes.items():
            context_parts.append(f"\n--- 论文 {pid} 笔记 ---\n{notes[:1500]}")

    if memory.observation:
        context_parts.append(f"\n最近操作结果：{memory.observation[:500]}")

    state_context = "\n".join(context_parts)

    system_prompt = RESPOND_PROMPT.format(
        user_goal=memory.user_goal,
        state_context=state_context
    )

    user_prompt = "请你根据以上信息，给用户一个清晰、专业的总结性回答。"

    try:
        response = state.llm_client.chat.completions.create(
            model=Config.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,  # 回答可以稍微自然一点
            max_tokens=2048,
        )
        state.final_response = response.choices[0].message.content
    except Exception as e:
        # 降级：手动拼一个回答
        parts = []
        if memory.selected_papers:
            parts.append("## 📚 为你找到以下相关论文：\n")
            for p in memory.selected_papers[:5]:
                parts.append(f"- **[{p.get('title', 'N/A')}]({p.get('url', '')})**")
                parts.append(f"  - 作者: {', '.join(p.get('authors', [])[:3])}")
                parts.append(f"  - 发布日期: {p.get('published', 'N/A')}")
                parts.append(f"  - 摘要: {p.get('summary', '')[:200]}...")
                parts.append("")

        if memory.paper_notes:
            parts.append("\n## 📝 论文笔记：\n")
            for pid, notes in list(memory.paper_notes.items())[:2]:
                parts.append(notes[:1000])
                parts.append("\n---\n")

        if parts:
            parts.append("\n（注：LLM 生成回答失败，以上为原始数据汇总。）")

        state.final_response = "\n".join(parts) if parts else "未能生成回答，请重试。"

    return state


# ============================================================
# 路由函数：决定 think_node 之后走哪个分支
# ============================================================
def router(state: AgentState) -> Literal["tool_node", "respond_node"]:
    """
    根据 think_node 的决策路由
    - respond → 直接走 respond_node
    - 其他 → 走 tool_node
    """
    if state.next_action == "respond":
        return "respond_node"
    return "tool_node"


def after_tool_router(state: AgentState) -> Literal["think_node", "respond_node"]:
    """
    tool_node 执行后的路由
    - 如果超过步数限制 → 强制 respond
    - 否则 → 回到 think_node 继续思考
    """
    if state.memory.is_over_limit():
        return "respond_node"
    return "think_node"


# ============================================================
# 构建 LangGraph 图
# ============================================================
def build_agent_graph() -> StateGraph:
    """
    构建并编译 ReAct Agent 状态图

    图结构：
    think_node ─┬─(respond)──→ respond_node ──→ END
               │
               └─(tool)──→ tool_node ──(超限)→ respond_node
                              │
                              └─(继续)→ think_node（循环）
    """
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("think_node", think_node)
    workflow.add_node("tool_node", tool_node)
    workflow.add_node("respond_node", respond_node)

    # 设置入口
    workflow.set_entry_point("think_node")

    # think_node 的条件路由
    workflow.add_conditional_edges(
        "think_node",
        router,
        {
            "tool_node": "tool_node",
            "respond_node": "respond_node",
        }
    )

    # tool_node 之后的条件路由
    workflow.add_conditional_edges(
        "tool_node",
        after_tool_router,
        {
            "think_node": "think_node",
            "respond_node": "respond_node",
        }
    )

    # respond_node 直接结束
    workflow.add_edge("respond_node", END)

    return workflow.compile()


# ============================================================
# 便捷接口：运行一次完整的 Agent 请求
# ============================================================
class ResearchAgent:
    """
    封装好的 Agent 接口，给 Streamlit 调用
    """

    def __init__(self):
        Config.validate()
        self.llm_client = OpenAI(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_BASE_URL,
        )
        self.vector_store = VectorStore()
        self.graph = build_agent_graph()
        # 每个会话一个 memory
        self.memory = ShortTermMemory()

    def run(self, user_message: str) -> str:
        """
        处理一条用户消息，返回 Agent 的最终响应
        """
        # 初始化本次请求
        self.memory.add_message("user", user_message)

        # 首次对话时设置 user_goal
        if not self.memory.user_goal:
            self.memory.user_goal = user_message

        # 先检查长期记忆
        if self.vector_store.paper_count() > 0:
            self.memory.add_message(
                "system",
                f"知识库中已有 {self.vector_store.paper_count()} 篇已读论文，"
                "如果用户问的是之前研究过的主题，可以先用 rag_search 查一查。"
            )

        # 创建初始状态
        initial_state = AgentState(
            memory=self.memory,
            vector_store=self.vector_store,
            llm_client=self.llm_client,
        )

        # 运行图，最多执行 15 步（LangGraph 内部有 recursion limit）
        try:
            final_state = self.graph.invoke(
                initial_state,
                config={"recursion_limit": Config.MAX_STEPS + 5}
            )
        except Exception as e:
            return f"Agent 运行出错: {str(e)}"

        response = final_state.final_response if hasattr(final_state, 'final_response') else ""

        if not response:
            response = "抱歉，Agent 未能生成有效回答。请重试或换一个研究主题。"

        self.memory.add_message("assistant", response)
        return response

    def reset(self):
        """重置对话（开始新话题时调用）"""
        self.memory = ShortTermMemory()