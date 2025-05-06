from langchain_core.documents import Document

from langchain_chroma import Chroma

from langchain_core.embeddings import Embeddings

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langchain_core.prompts import ChatPromptTemplate

from langchain_deepseek import ChatDeepSeek

import requests

import dotenv

import os

# 加载代理
os.environ['http_proxy'] = "http://127.0.0.1:7078"
os.environ['https_proxy'] = "http://127.0.0.1:7078"

# 加载环境
dotenv.load_dotenv()
GUIJI_API_KEY = os.getenv("GUIJI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 生成文档
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"}
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

# 自定义一个嵌入模型类(解决OpenAI没法使用的问题)    继承自langchain的Embeddings基类，需要实现以下两个方法
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
        return res['data'][0]['embedding']

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.payload = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float"
        }
        response = requests.request("POST", self.url, json=self.payload, headers=self.headers)
        res = response.json()
        return [item['embedding'] for item in res['data']]

# 向量存储
"""
默认使用的是欧式距离;当然也可以采用余弦相似度，需要在下面手动切换。
余弦相似度的取值范围为[-1, 1],值越大越相似，但是在chromadb里做了
调整，采用的是余弦距离值，也就是1-余弦相似度，这样设计的目的是为了
统一判断标准，即不管哪种标准都是值越小越相似。
"""
vectorstore = Chroma.from_documents(
    documents,
    embedding=GuiJiEmbeddings(),
    collection_metadata={"hnsw:space": "l2"}    # l2距离是默认的欧氏距离
)
'''
# 根据与字符串查询的相似性返回文档
print(vectorstore.similarity_search("cat"))

# 返回分数
print(vectorstore.similarity_search_with_score("dog"))

# 根据与嵌入查询的相似性返回文档
embedding = GuiJiEmbeddings().embed_query("dog")
print(vectorstore.similarity_search_by_vector(embedding))

# 返回分数
print(vectorstore.similarity_search_by_vector_with_relevance_scores(embedding))
"""由于这两种方式使用的词嵌入模型相同，所以结果应该是相同的"""
'''

# 检索器的构建(使用两种方法)
#retriever = RunnableLambda(vectorstore.similarity_search_with_score).bind(k=2) # 选择最相近的一个

#print(retriever.batch(["cat", "shark"]))

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}
)

#print(retriever.batch(["cat", "fish"]))

"""构建一个简单的检索增强程序"""

# 1.构建模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=2048,
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)

# 2.构建提示词模板
message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

# 3.构建链(检索器上面已经构建好了)
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

response = rag_chain.invoke("tell me about cats")
print(response.content)
