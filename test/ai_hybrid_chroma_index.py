import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, KeywordTableIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from IPython.display import Markdown, display
import chromadb

from Hybrid_Retriever import HybridRetriever
from LlamaIndexLangChainRetriever import LlamaIndexLangChainRetriever
from LlamaIndexRetrieverWrapper import LlamaIndexRetrieverWrapper

# create client and a new collection
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")

# define embedding function
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# set up ChromaVectorStore and load in data
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
vector_index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

llama_retriever = vector_index.as_retriever()

# Create a Keyword Index
# Settings.llm =
# keyword_index = KeywordTableIndex.from_documents(documents)


from langchain.retrievers import LlamaIndexRetriever
# from langchain.chains import MultiRetrievalQA
# from langchain_community.retrievers.bm25 import BM25Retriever

# Set up retrievers for both vector and keyword searches
vector_retriever = LlamaIndexLangChainRetriever(llamaindex_retriever=llama_retriever)

# keyword_retriever = LlamaIndexRetriever(index=keyword_index)

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader("./data/paul_graham/", glob="**/*.*")
docs = loader.load()

keyword_retriever = BM25Retriever.from_documents(docs)

ensemble_retriever = EnsembleRetriever(
    retrievers=[keyword_retriever, vector_retriever
], weights=[0.5, 0.5]
)
# hybrid_retriever = HybridRetriever([vector_retriever, keyword_retriever])


# Add chat memory

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True
)
memory.output_key = "answer"

# Create conversational retrieval chain

from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
os.environ["DASHSCOPE_API_KEY"] = ""
# create a new Chat with OpenAI
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="deepseek-r1",  # 此处以qwen-plus为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    # other params...
)

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=ensemble_retriever,
    memory=memory,
    return_source_documents=True,
    output_key="answer"
)

import gradio as gr
# Wrapping in a function - note that history isn't used, as the memory is in the conversation_chain

def chat(message, history):
    result = conversation_chain.invoke({"question": message})
    return result["answer"]

# And in Gradio:
view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)


