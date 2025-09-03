from llama_index import VectorStoreIndex, SimpleDirectoryReader
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool


#Create a LlamaIndex Index
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

#Create a Query Engine from LlamaIndex
query_engine = index.as_query_engine()



# Create memory to store conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# Create LLM
llm = ChatOpenAI(temperature=0)

# Define tool using LlamaIndex
llama_tool = Tool(
    name="DocumentQueryTool",
    func=lambda q: query_engine.query(q).response,
    description="Use this for questions about your uploaded documents."
)

# Initialize agent with memory
agent_with_memory = initialize_agent(
    tools=[llama_tool],
    llm=llm,
    agent="chat-conversational-react-description",  # This agent uses memory
    memory=memory,
    verbose=True
)

print(agent_with_memory.run("What's the main idea of the document?"))
print(agent_with_memory.run("Can you explain that more simply?"))
print(agent_with_memory.run("What did I just ask you about?"))