import streamlit as st
import json
import os
import time
from agent import ResearchAgent

st.set_page_config(page_title="学术论文研读 Agent", page_icon="📚", layout="wide")

HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history.json")

def _load_history() -> list[dict]:
    """从本地 JSON 文件加载对话历史"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def _save_history(history: list[dict]):
    """保存对话历史到本地 JSON 文件"""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # 保存失败不影响主流程

# ---- session state init ----
if "agent" not in st.session_state:
    try:
        st.session_state.agent = ResearchAgent()
    except Exception as e:
        st.error(f"初始化失败: {e}")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = _load_history()

def title_of(msgs):
    for m in msgs:
        if m["role"] == "user":
            return m["content"].strip().replace("\n", " ")[:30]
    return "（空）"

agent = st.session_state.agent

# ============ SIDEBAR ============
with st.sidebar:
    st.title("📚 Agent")
    st.success("✅ 就绪")
    st.metric("📖 知识库", agent.vector_store.paper_count())

    st.divider()
    st.caption(f"当前：{title_of(st.session_state.messages)}" if st.session_state.messages else "新建对话")

    if st.button("🔄 新话题", use_container_width=True):
        if st.session_state.messages:
            # 保存：包含 messages + memory 快照
            st.session_state.history.append({
                "title": title_of(st.session_state.messages),
                "messages": list(st.session_state.messages),
                "memory": agent.memory.to_dict(),
            })
        st.session_state.messages = []
        agent.reset()
        st.rerun()

    st.divider()
    st.markdown("### 💬 历史")
    for i in range(len(st.session_state.history) - 1, -1, -1):
        h = st.session_state.history[i]
        c1, c2 = st.columns([5, 1])
        with c1:
            if st.button(f"📝 {h['title']}", key=f"h_{i}"):
                # 当前对话存回历史（如果不同）
                if st.session_state.messages and title_of(st.session_state.messages) != h["title"]:
                    st.session_state.history.append({
                        "title": title_of(st.session_state.messages),
                        "messages": list(st.session_state.messages),
                        "memory": agent.memory.to_dict(),
                    })
                # 恢复选中对话
                st.session_state.messages = list(h["messages"])
                agent.memory.from_dict(h.get("memory", {}))
                # 删掉原记录（它现在是"当前对话"了）
                st.session_state.history.pop(i)
                st.rerun()
        with c2:
            if st.button("🗑", key=f"d_{i}"):
                st.session_state.history.pop(i)
                st.rerun()

    if st.session_state.history and st.button("清空历史"):
        st.session_state.history = []
        st.rerun()

    # ---- 当前论文列表 ----
    mem = agent.memory
    if mem.selected_papers:
        st.divider()
        st.markdown("### 📋 论文")
        for i, p in enumerate(mem.selected_papers):
            with st.expander(f"{i+1}. {p.get('title', '')[:50]}", expanded=False):
                st.markdown(f"**作者**: {', '.join(p.get('authors', [])[:3])}")
                st.markdown(f"**日期**: {p.get('published', '')}")
                st.markdown(f"**摘要**: {p.get('summary', '')[:200]}...")
                st.markdown(f"[arXiv]({p.get('url', '')}) · [PDF]({p.get('pdf_url', '')})")
        if mem.paper_notes:
            st.markdown("**笔记**: " + ", ".join(mem.paper_notes.keys()))

    # ---- 已读论文库（ChromaDB） ----
    st.divider()
    st.markdown("### 🗄️ 已读论文库")
    all_ids = agent.vector_store.get_all_paper_ids()
    if all_ids:
        st.caption(f"共 {len(all_ids)} 篇已读论文")
        with st.expander(f"浏览已读论文库 ({len(all_ids)} 篇)", expanded=False):
            search_term = st.text_input("搜索", key="library_search", placeholder="输入关键词筛选...")
            if search_term:
                results = agent.vector_store.search(search_term, top_k=min(10, len(all_ids)))
                for r in results:
                    st.markdown(f"**{r.get('title', 'N/A')}**")
                    st.caption(f"arXiv: {r.get('paper_id', '')} · 相似度: {r.get('score', 0):.2f}")
                    if r.get('url'):
                        st.markdown(f"[arXiv]({r['url']})")
                    st.divider()
            else:
                # 默认显示最近存的论文 ID 列表
                for pid in all_ids[-10:]:  # 最近 10 篇
                    st.caption(f"📄 {pid}")
    else:
        st.caption("暂无已读论文")

# ============ MAIN CHAT ============
st.title("📚 学术论文研读 Agent")
st.caption("输入研究主题，自动搜索、筛选、阅读论文")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("输入研究主题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            # 进度占位
            progress_placeholder = st.empty()
            # 流式输出
            reply_placeholder = st.empty()
            full_reply = ""

            def update_progress(msg):
                progress_placeholder.info(msg)

            for token in agent.run_stream(prompt, progress_callback=update_progress):
                full_reply += token
                reply_placeholder.markdown(full_reply)
            progress_placeholder.success("✅ 完成")
            st.session_state.messages.append({"role": "assistant", "content": full_reply})
        except Exception as e:
            st.error(f"出错: {e}")
    st.rerun()

# 自动保存对话历史到本地文件
_save_history(st.session_state.history)
