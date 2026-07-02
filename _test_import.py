"""临时测试脚本：验证所有模块能正常导入"""
print("1/5 测试 config...")
from config import Config
print("   ✅ config OK")

print("2/5 测试 memory...")
from memory import ShortTermMemory, VectorStore
print("   ✅ memory OK")

print("3/5 测试 prompts...")
from prompts import THINKING_PROMPT, SUMMARIZE_PROMPT, COMPARE_PROMPT
print("   ✅ prompts OK")

print("4/5 测试 tools...")
from tools import search_papers
print("   ✅ tools OK")

print("5/5 测试 agent...")
from agent import ResearchAgent
print("   ✅ agent OK")

print("\n🎉 所有模块导入成功！")