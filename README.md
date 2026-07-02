# 📚 学术论文研读 Agent (research-agent)

基于 **LangGraph + DeepSeek API** 的智能学术论文搜索、筛选、阅读与对比助手。

## 🚀 快速开始

### 1. 环境准备
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境（Windows）
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key
编辑 `.env` 文件，填入你的 DeepSeek API Key：
```
DEEPSEEK_API_KEY=sk-your-api-key-here
```

### 3. 启动应用
```bash
streamlit run main.py
```

浏览器打开 `http://localhost:8501` 即可使用。

## 📁 项目结构

```
research-agent/
├── main.py              # Streamlit 聊天界面
├── agent.py             # LangGraph 状态图（ReAct 循环）
├── tools.py             # 工具函数（搜索、筛选、阅读、对比、RAG）
├── prompts.py           # System Prompt 集中管理
├── memory.py            # 短期记忆 + ChromaDB 向量存储
├── config.py            # 配置管理（API Key、路径等）
├── requirements.txt     # 依赖清单
├── .env                 # 环境变量（API Key，不提交到 Git）
└── README.md            # 项目说明
```

## 🔄 Agent 工作流

```
用户输入 → think_node（LLM 决策）
  ├─ search   → 搜索 arXiv 论文
  ├─ screen   → LLM 摘要筛选（Top 3-5）
  ├─ read     → 下载 PDF + 提取全文 + 生成笔记
  ├─ compare  → 横向对比多篇论文
  ├─ rag      → 检索长期知识库
  └─ respond  → 输出回答
```

## 🛠 技术栈

| 组件 | 用途 |
|------|------|
| LangGraph | ReAct 状态图引擎 |
| DeepSeek API | LLM 推理（决策、总结、对比） |
| ChromaDB | 向量数据库（长期记忆） |
| arXiv API | 论文搜索 |
| PyMuPDF | PDF 文本提取 |
| Streamlit | Web 聊天界面 |

## 📝 使用示例

- "帮我找几篇关于 Transformer 注意力机制的最新论文"
- "对比一下这几篇论文的核心方法"
- "第一篇论文的创新点是什么？有哪些局限？"
- "我之前读过哪些关于 GNN 的论文？"

## ⚙️ 配置说明

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | （必填） |
| `DEEPSEEK_BASE_URL` | API 地址 | `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | 模型名称 | `deepseek-chat` |
| `LOG_LEVEL` | 日志级别 | `INFO` |

## 📄 License

MIT