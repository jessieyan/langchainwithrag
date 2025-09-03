import os, pymongo, pprint
import getpass
import gradio as gr

if not os.getenv("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("Enter your DeepSeek API key: ")

from llama_index.llms.deepseek import DeepSeek
from llama_index.llms.ollama import Ollama
from pymongo.operations import SearchIndexModel
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.settings import Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, ExactMatchFilter, FilterOperator
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine

from llama_index.core import get_response_synthesizer

os.environ["DEEPSEEK_API_KEY"] = ""

ATLAS_CONNECTION_STRING = "mongodb+srv://jessie:hhxxttxs@cluster0.7mm0sot.mongodb.net"
# ATLAS_CONNECTION_STRING = "mongodb://localhost:27017"
# Settings.llm = OpenAI()
# Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
Settings.llm = Ollama(model="deepseek-r1:7b", request_timeout=360.0)
Settings.chunk_size = 100
Settings.chunk_overlap = 10
# Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en")
Settings.embed_model = HuggingFaceEmbedding(model_name="WhereIsAI/UAE-Large-V1")
# uri = "mongodb://localhost:27018/?directConnection=true"

# Connect to your Atlas cluster
mongo_client = pymongo.MongoClient(ATLAS_CONNECTION_STRING)

documents = SimpleDirectoryReader(input_files=["data/Jonnie_ADHD_Support_Guide.pdf"]).load_data()

# Instantiate the vector store
atlas_vector_store = MongoDBAtlasVectorSearch(
    mongo_client,
    db_name = "llamaindex_db",
    collection_name = "test",
    vector_index_name = "vector_index3"
)
vector_store_context = StorageContext.from_defaults(vector_store=atlas_vector_store)

vector_store_index = VectorStoreIndex.from_documents(
   documents, storage_context=vector_store_context, show_progress=True
)

# Specify the collection for which to create the index
collection = mongo_client["llamaindex_db"]["test"]

# Create your index model, then create the search index
search_index_model = SearchIndexModel(
  definition={
    "fields": [
      {
        "type": "vector",
        "path": "embedding",
        "numDimensions": 1024,
        "similarity": "cosine"
      },
      {
        "type": "filter",
        "path": "metadata.page_label"
      }
    ]
  },
  name="vector_index3",
  type="vectorSearch"
)

collection.create_search_index(model=search_index_model)

retriever = vector_store_index.as_retriever(similarity_top_k=3)
# nodes = retriever.retrieve("MongoDB Atlas security")

# query_engine = vector_store_index.as_query_engine(similarity_top_k=10)

# def format_history(msg:str, history:list[list[str, str]], system_prompt):
#     chat_history = [{"role": "system", "content": system_prompt}]
#     for query, response in history:
#         chat_history.append({"role": "user", "content": query})
#         chat_history.append({"role": "assitant", "content": response})
#     chat_history.append({"role": "user", "content": msg})
#     return chat_history
#
#
# def generate_response(msg: str, history: list[list[str, str]], system_prompt:str):
#     chat_history = format_history(msg, history, system_prompt)
#     response = ollama.chat(model="deepseek-r1", stream=True, messages=chat_history)
#     message=""
#     for partial_resp in response:
#         token = partial_resp['message']['content']
#         message += token
#         yield message
#
#
#
# view = gr.ChatInterface(generate_response, type="messages").launch(inbrowser=True)

# define response synthesizer
response_synthesizer = get_response_synthesizer()
# vector query engine
vector_query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)


# Gradio interface function
def answer_question(query):
    # Retrieve relevant documents from Upstash Vector
    return vector_query_engine.query(query)

    # Use the most relevant document for QA
    # if results:
    #     context = results[0].page_content
    #     qa_input = {"question": query, "context": context}
    #     answer = qa_pipeline(qa_input)["answer"]
    #     return f"Answer: {answer}\n\nContext: {context}"
    # else:
    #     return "No relevant context found."


# Set up Gradio interface
iface = gr.Interface(
    fn=answer_question,
    inputs="text",
    outputs="text",
    title="RAG Application",
    description="Ask a question, and the app will retrieve relevant information and provide an answer."
)

# Launch the Gradio app
iface.launch()