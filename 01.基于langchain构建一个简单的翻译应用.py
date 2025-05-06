from langchain_deepseek import ChatDeepSeek

from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate

from fastapi import FastAPI

from langserve import add_routes

import uvicorn

from dotenv import load_dotenv

import os

# 加载代理
os.environ['http_proxy'] = "http://127.0.0.1:7078"
os.environ['https_proxy'] = "http://127.0.0.1:7078"

# 1.加载.env文件中的DEEPSEEK_API_KEY
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 2.构建提示词模板
system_prompt = "请将下面的内容从{language1}翻译到{language2}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{text}")]
)

# 3.构建模型
model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=2048,
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)

# 4.构建解析器
parser = StrOutputParser()

# 5.构建链
chain = prompt_template | model | parser

# 6.构建app
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Translate text from language1 to language2"    # 这里描述使用中文就会有问题
)

# 7.添加链的路由
add_routes(
    app,
    chain,
    path='/chain',
    enabled_endpoints=["invoke", "stream", "stream_log"]
)

'''# 获取结果
result = chain.invoke({"language1": "中文", "language2": "英文", "text": "我爱中国"})
print(result)'''

# 8.测试
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
