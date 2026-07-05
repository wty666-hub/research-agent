"""
配置模块：读取环境变量和 API Key
支持 .env 文件，优先从环境变量读取
"""

import os
from dotenv import load_dotenv

# 加载 .env 文件（如果存在）
load_dotenv()


class Config:
    """全局配置，从环境变量读取"""

    # DeepSeek API
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv(
        "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
    )
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # Agent 设置
    MAX_STEPS: int = 30           # ReAct 循环最大步数
    MAX_SEARCHES: int = 3         # 最大搜索次数
    MAX_PAPERS_TO_READ: int = 5   # 最多深入阅读的论文数
    LLM_TEMPERATURE: float = 0.1  # 决策用低温，保证稳定
    READ_MODE: str = os.getenv("READ_MODE", "full")  # "full"=下载PDF全文, "abstract"=仅读摘要
    SEARCH_CACHE_TTL: int = 600   # 搜索结果缓存有效期（秒），默认 10 分钟
    PAPER_MIN_YEAR: int = int(os.getenv("PAPER_MIN_YEAR", "0"))  # 论文最早年份（0=不限）
    PAPER_MAX_YEAR: int = int(os.getenv("PAPER_MAX_YEAR", "0"))  # 论文最晚年年份（0=不限）

    # 路径
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    CHROMA_DIR: str = os.path.join(BASE_DIR, "chroma_db")
    PAPERS_DIR: str = os.path.join(BASE_DIR, "papers")

    @classmethod
    def validate(cls) -> bool:
        """校验必要配置是否存在"""
        if not cls.DEEPSEEK_API_KEY:
            raise ValueError(
                "未找到 DEEPSEEK_API_KEY！请创建 .env 文件并填入你的 API Key。\n"
                "参考 .env.example 文件。"
            )
        return True


# 确保必要目录存在
os.makedirs(Config.CHROMA_DIR, exist_ok=True)
os.makedirs(Config.PAPERS_DIR, exist_ok=True)