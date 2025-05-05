from langchain_deepseek import ChatDeepSeek

from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage

from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.prebuilt import create_react_agent

from dotenv import load_dotenv

import os

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_KEY")

# 创建agent
memory = MemorySaver()
model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=2048,
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)
search = TavilySearchResults(max_results=2, tavily_api_key=TAVILY_KEY)
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# 使用agent
config = {"configurable": {"thread_id": "licy123"}}
for chunk in agent_executor.stream({"messages": [HumanMessage(content="你好我是小炽，我住在中国的南京市。")]}, config):
    print(chunk)
    print('-' * 50)
for chunk in agent_executor.stream({"messages": [HumanMessage(content="我住的地方明天天气怎么样？")]}, config):
    print(chunk)
    print('-' * 50)
