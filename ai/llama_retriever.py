from llamaindex_langchain_retriever import LlamaIndexLangChainRetriever
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader, Document
import os

def get_llama_retriever( vector_store, embed_model, dir: str = "knowledge_base/data"):
    # load documents
    documents = SimpleDirectoryReader(dir).load_data()
    os.environ["DASHSCOPE_API_KEY"] = "sk-6e5c573c75044deb80371b793a8b24a3"
    
    print("Document ID:", documents[0].doc_id)

    videoDoc = Document(
        text="怎么样拍嗝呢？ 首先准备一块纱布垫在肩上，大人的身体稍往后倾，一只手抱住宝宝的臀部，让宝宝竖直贴在胸前，头靠在肩膀上；另一只手以空心掌轻拍宝宝背部；听到宝宝发出“嗝~”地声音，就是胃里排出空气的声音。",
        metadata={
            "description": "【拍嗝技巧】超實用~2種幫寶寶拍嗝的方法&要點！新手爸媽一看就懂｜周彥怡醫師｜禾馨怡仁婦幼中心",
            "url": "https://www.youtube.com/watch?v=FpMi2ee-qvs",
            "content_type": "video",
        }
    )

    videos = [
    {
        "url": "https://example.com/video1.mp4",
        "description": "A cozy treehouse video tour with kids playing around the yard"
    },
    {
        "url": "https://www.youtube.com/watch?v=FpMi2ee-qvs",
        "description": "【拍嗝技巧】超實用~2種幫寶寶拍嗝的方法&要點！新手爸媽一看就懂｜周彥怡醫師｜禾馨怡仁婦幼中心"
    },
    {
        "url": "https://example.com/video2.mp4",
        "description": "A documentary about marine life in the Pacific Ocean"
    }]

    docs = [
        Document(page_content=video["description"], text=video["description"], metadata={"url": video["url"]})
        for video in videos
    ]

    documents.extend(docs)


    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model)
    
    llama_retriever = vector_index.as_retriever()
    # convert llama index retriever
    vector_retriever = LlamaIndexLangChainRetriever(llamaindex_retriever=llama_retriever)

    return vector_retriever





