import langchain

from langchain_deepseek import ChatDeepSeek

from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv

import os

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
#print(DEEPSEEK_API_KEY)

message = [
    SystemMessage(content="将下面的内容从中文翻译到英文"),
    HumanMessage(content="我爱中国"),
]

model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=2048,
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)

print(model.invoke(message))
