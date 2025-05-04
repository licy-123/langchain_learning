import chromadb

# 创建chroma客户端
chroma_client = chromadb.Client()

# 创建集合
collections = chroma_client.create_collection(
    name="my_collections"
)

# 向集合中添加一些文档
collections.add(
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ],
    ids=["id1", "id2"]
)

# 查询集合
results = collections.query(
    query_texts=["This is a query document about hawaii"],
    n_results=2
)
print(results)
