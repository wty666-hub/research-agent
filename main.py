"""
Streamlit 聊天界面
- 左侧边栏：论文列表、知识库状态
- 右侧主区域：聊天窗口
"""

import streamlit as st
from agent import ResearchAgent

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="学术论文研读 Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 初始化
# ============================================================
@st.cache_resource
def init_agent():
    """初始化 Agent（全局单例，用 st.cache_resource 避免重复创建）"""
    return ResearchAgent()


# 初始化 session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_ready" not in st.session_state:
    st.session_state.agent_ready = False
if "agent_error" not in st.session_state:
    st.session_state.agent_error = ""

# 尝试初始化 Agent
if not st.session_state.agent_ready and not st.session_state.agent_error:
    try:
        st.session_state.agent = init_agent()
        st.session_state.agent_ready = True
    except Exception as e:
        st.session_state.agent_error = str(e)

# ============================================================
# 左侧边栏
# ============================================================
with st.sidebar:
    st.title("📚 学术论文研读 Agent")
    st.markdown("---")

    # 状态显示
    if st.session_state.agent_error:
        st.error(f"❌ Agent 初始化失败:\n\n{st.session_state.agent_error}")
        st.markdown("请检查 `.env` 文件中的 `DEEPSEEK_API_KEY` 是否正确配置。")
    elif st.session_state.agent_ready:
        st.success("✅ Agent 就绪")

        # 知识库状态
        agent = st.session_state.agent
        paper_count = agent.vector_store.paper_count()
        st.metric("📖 知识库论文数", paper_count)

        if paper_count > 0:
            st.caption("已读论文已存入长期记忆，可随时追问。")

    st.markdown("---")

    # 快捷操作
    if st.session_state.agent_ready:
        if st.button("🔄 开始新话题", use_container_width=True):
            st.session_state.agent.reset()
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")

    # 使用提示
    with st.expander("💡 使用提示", expanded=False):
        st.markdown("""
        **用法示例：**
        - "帮我找几篇关于 Transformer 注意力机制的最新论文"
        - "对比一下这几篇论文的研究方法"
        - "第一篇论文的核心创新点是什么？"

        **工作流程：**
        1. 输入研究主题 → Agent 搜索 arXiv
        2. 自动筛选高相关论文（Top 3-5）
        3. 逐篇下载并生成结构化笔记
        4. 支持横向对比和多轮追问
        """)

    # 论文列表（运行时显示）
    if st.session_state.agent_ready and hasattr(st.session_state, "agent"):
        agent = st.session_state.agent
        memory = agent.memory

        if memory.selected_papers:
            st.markdown("### 📋 当前论文列表")
            for i, p in enumerate(memory.selected_papers):
                score = p.get("score", "?")
                with st.expander(f"{i+1}. {p.get('title', 'N/A')[:60]}... ({score}分)", expanded=False):
                    st.markdown(f"**作者**: {', '.join(p.get('authors', [])[:3])}")
                    st.markdown(f"**日期**: {p.get('published', 'N/A')}")
                    st.markdown(f"**摘要**: {p.get('summary', '')[:200]}...")
                    st.markdown(f"[📄 arXiv 链接]({p.get('url', '')})")
                    st.markdown(f"[📥 PDF 下载]({p.get('pdf_url', '')})")

        if memory.paper_notes:
            st.markdown("### 📝 已生成笔记")
            for pid in memory.paper_notes:
                st.caption(f"✅ {pid}")

# ============================================================
# 右侧主区域：聊天窗口
# ============================================================
st.title("📚 学术论文研读 Agent")
st.caption("基于 DeepSeek + LangGraph 的智能论文搜索与阅读助手")

# 显示错误状态
if st.session_state.agent_error:
    st.error(
        "### ⚠️ Agent 无法启动\n\n"
        f"**错误信息：** {st.session_state.agent_error}\n\n"
        "**解决方法：**\n"
        "1. 确保已创建 `.env` 文件（参考 `.env.example`）\n"
        "2. 在 `.env` 中填入你的 DeepSeek API Key\n"
        "3. 格式：`DEEPSEEK_API_KEY=sk-xxxxxxxx`\n\n"
        "配置完成后请刷新页面。"
    )

# 显示聊天历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 聊天输入框
if st.session_state.agent_ready:
    if prompt := st.chat_input("输入你的研究主题或问题..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 调用 Agent
        with st.chat_message("assistant"):
            with st.spinner("🤔 Agent 思考中..."):
                try:
                    response = st.session_state.agent.run(prompt)
                    st.markdown(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                except Exception as e:
                    error_msg = f"❌ 运行出错: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

        # 刷新以更新左侧论文列表
        st.rerun()
else:
    st.info("👈 请先在左侧边栏确认 Agent 状态。")