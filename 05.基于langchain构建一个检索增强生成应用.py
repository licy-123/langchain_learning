import bs4

from langchain import hub

from langchain_chroma import Chroma

from langchain_community.document_loaders import WebBaseLoader

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnablePassthrough

from langchain_deepseek import ChatDeepSeek

from langchain_core.prompts import ChatPromptTemplate

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.embeddings import Embeddings

import requests

from dotenv import load_dotenv

import os

from requests_toolbelt import user_agent

# 加载本地代理
os.environ['http_proxy'] = 'http://127.0.0.1:7078'
os.environ['https_proxy'] = 'http://127.0.0.1:7078'

# 1.加载环境
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GUIJI_API_KEY = os.getenv("GUIJI_API_KEY")

# 2.构建模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=2048,
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1",
)

# 3.加载博客内容
loader = WebBaseLoader(
    web_paths=["https://lilianweng.github.io/posts/2023-06-23-agent/"],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    )
)
docs = loader.load()

# 4.对内容进行分块
text_spliter = RecursiveCharacterTextSplitter()
splits = text_spliter.split_documents(docs)
print(splits)

# 5.构建索引(由于无法使用OpenAI,因此这里我们自己构建一个词嵌入的类)
class GuiJiEmbeddings(Embeddings):
    def __init__(self):
        super().__init__()
        self.model = "BAAI/bge-m3"
        self.url = "https://api.siliconflow.cn/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {GUIJI_API_KEY}",
            "Content-Type": "application/json"
        }
    def embed_query(self, text: str) -> list[float]:
        self.payload = {
            "model": self.model,
            "input": text,
            "encoding_format": "float"
        }
        response = requests.request("POST", self.url, json=self.payload, headers=self.headers)
        res = response.json()
        return res["data"][0]["embedding"]
    def embed_documents(self, text: list[str]) -> list[list[float]]:
        self.payload = {
            "model": self.model,
            "input": text,
            "encoding_format": "float"
        }
        response = requests.request("POST", self.url, json=self.payload, headers=self.headers)
        res = response.json()
        return [item["embedding"] for item in res["data"]]

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=GuiJiEmbeddings()
)

# 6.构建检索器
retriever = vectorstore.as_retriever()

# 7.构建提示词模板
# prompt = hub.pull("langchain-ai/rag-prompt")
prompt = ChatPromptTemplate.from_template("""
请根据以下上下文回答问题：
上下文：{context}
问题：{question}
答案：
""")

# 8.定义一个提取文档内容的函数
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 9.构建链
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke("What is agent?"))
