"""
记忆模块：短期对话管理 + ChromaDB 长期向量存储
- ShortTermMemory: 管理对话历史和论文阅读状态
- VectorStore: ChromaDB 封装，存储和检索已读论文
"""

import json
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from config import Config


# ============================================================
# 短期记忆管理器
# ============================================================
class ShortTermMemory:
    """
    管理当前会话的短期状态
    - messages: 对话历史（用户和 Agent 的消息）
    - papers: 已搜索到的论文列表
    - selected_papers: 筛选后高相关论文
    - paper_notes: 各论文的结构化笔记 {paper_id: notes}
    - observation: 最近一次工具执行的返回
    - step_count: 当前 ReAct 循环步数
    """

    def __init__(self):
        self.messages: list[dict] = []
        self.user_goal: str = ""
        self.papers: list[dict] = []
        self.selected_papers: list[dict] = []
        self.paper_notes: dict[str, str] = {}
        self.observation: str = ""
        self.step_count: int = 0
        self.comparison_done: bool = False  # 是否已完成横向对比
        self.all_read: bool = False  # 所有筛选论文是否已读完
        self.kb_papers: list[dict] = []  # 知识库中查到的相关已读论文

    def add_message(self, role: str, content: str) -> None:
        """添加一条消息到对话历史"""
        self.messages.append({"role": role, "content": content})

    def get_state_context(self) -> str:
        """生成当前状态的文本摘要"""
        return f"""
- 用户目标: {self.user_goal}
- 搜索到的论文数: {len(self.papers)}
- 筛选后的论文数: {len(self.selected_papers)}
- 已读论文数: {len(self.paper_notes)}
"""

    def to_dict(self) -> dict:
        """序列化为 dict（用于保存到对话历史）"""
        return {
            "user_goal": self.user_goal,
            "papers": self.papers,
            "selected_papers": self.selected_papers,
            "paper_notes": self.paper_notes,
        }

    def from_dict(self, data: dict) -> None:
        """从 dict 恢复状态"""
        self.user_goal = data.get("user_goal", "")
        self.papers = data.get("papers", [])
        self.selected_papers = data.get("selected_papers", [])
        self.paper_notes = data.get("paper_notes", {})


# ============================================================
# ChromaDB 向量存储（长期记忆）
# ============================================================
class VectorStore:
    """
    基于 ChromaDB 的长期知识库
    - 存储已读论文的摘要和关键段落
    - 支持语义检索
    """

    def __init__(self):
        self.client = chromadb.PersistentClient(path=Config.CHROMA_DIR)
        # 使用 SentenceTransformer embedding（第一次需要从 HuggingFace 下载模型，约 80MB）
        # 下载后会缓存到本地，之后就不需要联网了。请确保 VPN 已开启。
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="research_papers",
            embedding_function=self.embedding_fn,
            metadata={"description": "已读学术论文的知识库"}
        )

    def add_paper(
        self,
        paper_id: str,
        title: str,
        abstract: str,
        notes: str,
        authors: str = "",
        url: str = ""
    ) -> None:
        """
        将一篇论文加入向量库
        - 摘要全文存入 metadata（精确匹配）
        - 笔记分块存入向量（语义检索）
        """
        # 存储完整文档到 metadata
        metadata = {
            "paper_id": paper_id,
            "title": title,
            "authors": authors,
            "url": url,
            "abstract": abstract,
            "notes": notes[:2000],  # metadata 有长度限制，截断长笔记
        }

        # 将摘要和笔记作为可检索文本
        document = f"标题：{title}\n作者：{authors}\n摘要：{abstract}\n笔记：{notes[:3000]}"

        # 去重：先检查是否已存在
        existing = self.collection.get(ids=[paper_id])
        if existing and existing["ids"]:
            self.collection.update(
                ids=[paper_id],
                documents=[document],
                metadatas=[metadata]
            )
        else:
            self.collection.add(
                ids=[paper_id],
                documents=[document],
                metadatas=[metadata]
            )

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        语义检索已读论文
        返回: [{paper_id, title, abstract, notes, url, score}, ...]
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )

            papers = []
            if results["ids"] and results["ids"][0]:
                for i, pid in enumerate(results["ids"][0]):
                    meta = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else 0
                    papers.append({
                        "paper_id": pid,
                        "title": meta.get("title", "未知标题"),
                        "abstract": meta.get("abstract", ""),
                        "notes": meta.get("notes", ""),
                        "url": meta.get("url", ""),
                        "score": round(1 - distance, 4) if distance else 0  # 距离转相似度
                    })
            return papers
        except Exception as e:
            print(f"[VectorStore] 检索失败: {e}")
            return []

    def get_all_paper_ids(self) -> list[str]:
        """获取所有已存论文的 ID"""
        try:
            result = self.collection.get()
            return result["ids"] if result["ids"] else []
        except Exception:
            return []

    def paper_count(self) -> int:
        """已存论文数量"""
        return self.collection.count()