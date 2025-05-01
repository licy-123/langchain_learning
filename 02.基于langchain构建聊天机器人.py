from dotenv import load_dotenv

import os

from langchain_deepseek import ChatDeepSeek

from langchain_core.messages import HumanMessage, trim_messages

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.runnables import RunnablePassthrough

from operator import itemgetter

import tiktoken

# 1.加载.env文件中的DEEPSEEK_API_KEY
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 2.构建模型
model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=2048,
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)

# 3.构建获取会话历史的方法
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 4. 构建提示词模板
prompt = ChatPromptTemplate(
    [
        ("system",
         "你是一个外表高冷内心却很柔软温柔的女生，请以该身份回答后续的问题"
         ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

# 5.构建计数器(由于deepseek模型不支持自动计数)

def count_tokens(text: str) -> int:
    if not isinstance(text, str):  # 检查输入是否为字符串
        text = str(text)  # 强制转换或跳过非字符串输入
    enc = tiktoken.get_encoding("cl100k_base")  # 通用编码（非 DeepSeek 专用）
    return len(enc.encode(text))

# 6.构建信息修剪器
trimmer = trim_messages(
    max_tokens=2048,
    strategy='last',
    token_counter=count_tokens,
    include_system=True,
    allow_partial=False,
    start_on='human'
)

# 7.构建链
chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

# 8.构建自动化存储记忆问答
run_with_messages_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages"
)
config = {"configurable": {"session_id": "licy123"}}

while True:
    print("温馨提示,按q/Q键退出程序")
    print("请输入提示词:", end='')
    query = input()
    if query == 'q' or 'Q':
        exit()
    response = run_with_messages_history.invoke(
        {"messages": [HumanMessage(content=query)]},
        config
    )
    print(response.content)
