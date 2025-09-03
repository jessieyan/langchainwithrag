from langchain.agents import initialize_agent, Tool
import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from llama_index.core import load_index_from_storage, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from langchain.memory import ConversationBufferMemory
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


import os

# Set your OpenAI key
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Replace with your key
db_name = "vector_db"
os.environ["DASHSCOPE_API_KEY"] = ""


#Create a LlamaIndex Index
documents = SimpleDirectoryReader("knowledge_base").load_data()
index = VectorStoreIndex.from_documents(documents)

text_splitter = SemanticChunker(HuggingFaceEmbeddings())

chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="db_name")
print(f"Vectorstore created with {vectorstore._collection.count()} documents")

collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"The vectors have {dimensions:,} dimensions")

# Save index to disk
index.storage_context.persist(persist_dir="./index_storage")

# Load index from disk
storage_context = StorageContext.from_defaults(persist_dir="./index_storage")
index = load_index_from_storage(storage_context)

# Create a query engine from LlamaIndex
query_engine = index.as_query_engine()

# Load the language model
# llm = ChatOpenAI(temperature=0)
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="deepseek-r1",  # 此处以qwen-plus为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    # other params...
)

# # Define the LlamaIndex tool
# llama_tool = Tool(
#     name="DocumentQueryTool",
#     func=lambda q: query_engine.query(q).response,
#     description="Useful for answering questions from loaded documents."
# )

# Create memory to store conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create conversation chain
conversation_chain = ConversationChain(llm=llm, memory=memory)


# # Initialize agent with memory
# agent_with_memory = initialize_agent(
#     tools=[llama_tool],
#     llm=llm,
#     agent="chat-conversational-react-description",  # This agent uses memory
#     memory=memory,
#     verbose=True
# )

# user_input = "Tell me about the history of OpenAI"

# Use LlamaIndex for document lookup
# llama_answer = query_engine.query(user_input)

# Feed the retrieved information into the conversation
# contextual_input = f"{llama_answer}\n\nNow, continue the conversation: {user_input}"
# response = conversation_chain.run(contextual_input)

# response = agent_with_memory.run("Summarize the main ideas in the documents.")
# print(response)


def chat(message, history):
    # user_input = "Tell me about the history of OpenAI"

    # Use LlamaIndex for document lookup
    llama_answer = query_engine.query(message)

    # Feed the retrieved information into the conversation
    contextual_input = f"{llama_answer}\n\nNow, continue the conversation: {message}"
    result = conversation_chain.run(contextual_input)
    # result = conversation_chain.invoke({"question": message})
    return result["answer"]

# And in Gradio:
view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)