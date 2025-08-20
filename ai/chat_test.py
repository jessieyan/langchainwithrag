aimport os
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from embedding_model import get_embedding
from vector_store import get_vector_store
from llama_retriever import get_llama_retriever
from hybrid_retriever import get_hybrid_retriever
from keyword_retriever import get_keyword_retriever
from prompt import get_contextualize_prompt, get_answer_prompt
from system_message import create_system_message
from custom_yomi_tool_agent import create_react_yomi_agent
from langchain_core.tools import tool

knowledge_base = "./knowledge_base/data"

def question_answer_chat(role: str = None):
    # Create embeddings
    embed_model = get_embedding('dashscope')

    # Create vector store
    vector_store = get_vector_store('milvus')

    # Create llama index retriever
    vector_retriever = get_llama_retriever(vector_store=vector_store, embed_mode=embed_model, dir=knowledge_base)

    # Create keyword retriever
    keyword_retriever = get_keyword_retriever(dir=knowledge_base)

    # Create hybrid retriever
    hybrid_retriever = get_hybrid_retriever([vector_retriever, keyword_retriever], [0.5, 0.5])

    # create a new Chat with OpenAI
    llm = ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="deepseek-r1",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    # Create conversational retrieval chain
    # Contextualize question
    contextualize_q_prompt = get_contextualize_prompt()

    history_aware_retriever = create_history_aware_retriever(
        llm, hybrid_retriever, contextualize_q_prompt
    )

    # Answer question
    qa_prompt = get_answer_prompt(role)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Statefully manage chat history
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    # conversational_rag_chain = RunnableWithMessageHistory(
    #     rag_chain,
    #     get_session_history,
    #     input_messages_key="input",
    #     history_messages_key="chat_history",
    #     output_messages_key="answer",
    # )
    # return conversational_rag_chain, store
    

    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """Retrieve information related to a query."""
        retrieved_docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
    @tool
    def ask_without_context(input: str) -> str:
        """Use this for standalone questions without any memory."""
        return f"[Stateless] You asked: {input}"

    
    # Get model response
    agent_executor = create_react_yomi_agent(llm, tools=[retrieve, ask_without_context])
    return agent_executor
    # response = agent_executor_taot.invoke({"messages": all_messages})
    # print(response['messages'][0]['content'])


from langchain_core.messages import SystemMessage
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing import List
from langchain_core.documents import Document
from langchain_core.tools import tool

llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="deepseek-r1",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)
embed_model = get_embedding('dashscope')
vector_store = get_vector_store('milvus')
vector_retriever = get_llama_retriever(vector_store=vector_store, embed_mode=embed_model, dir=knowledge_base)
    # Create keyword retriever
keyword_retriever = get_keyword_retriever(dir=knowledge_base)

    # Create hybrid retriever
hybrid_retriever = get_hybrid_retriever([vector_retriever, keyword_retriever], [0.5, 0.5])
contextualize_q_prompt = get_contextualize_prompt()

history_aware_retriever = create_history_aware_retriever(
    llm, hybrid_retriever, contextualize_q_prompt
)

    # Answer question
qa_prompt = get_answer_prompt(None)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    # retrieved_docs = conversational_rag_chain.invoke(query)

    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

class State(MessagesState):
    context: List[Document]


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: State):
    previous_messages = [
        # {"role": "system", "content": "You are a helpful AI assistant."}, # Commented out as we do not include system message
        {"role": "user", "content": "What is the capital of Australia?"},
        {"role": "assistant", "content": "The capital of Australia is Canberra."}
    ]
    system_message = """You are an assistant with access to specific tools. When the user's question requires a context, use the 'retrieve' tool. For the 'retrieve' tool, provide the user provided text as a string in to the 'query' argument in the tool.
    When the user's question does not require a context, use the 'ask_without_context' tool. For the 'ask_without_context' tool, provide the user provided text as a string into the 'input' argument in the tool."""
    system_message_taot = create_system_message(system_message)
    all_messages = [{"role": "system", "content": system_message_taot}]
    # Add previous messages (if available)
    all_messages.extend(previous_messages)
    """Generate tool call for retrieval or respond."""
    # llm_with_tools = llm.bind_tools([retrieve])
    agent = question_answer_chat()
    # response = llm_with_tools.invoke(state["messages"])
    result =  agent.invoke({"messages": all_messages})
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [result]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    context = []
    for tool_message in tool_messages:
        context.extend(tool_message.artifact)
    return {"messages": [response], "context": context}

graph_builder = StateGraph(MessagesState)

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()

# from IPython.display import Image, display

from PIL import Image
from io import BytesIO

img_bytes = graph.get_graph().draw_mermaid_png()
img = Image.open(BytesIO(img_bytes))
img.show()

input_message = "儿童敏感期?"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

# display(Image(graph.get_graph().draw_mermaid_png()))

# agent = question_answer_chat()
   
# import gradio as gr
previous_messages = [
        # {"role": "system", "content": "You are a helpful AI assistant."}, # Commented out as we do not include system message
        {"role": "user", "content": "What is the capital of Australia?"},
        {"role": "assistant", "content": "The capital of Australia is Canberra."}
    ]
system_message = """You are an assistant with access to specific tools. When the user's question requires a context, use the 'retrieve' tool. For the 'retrieve' tool, provide the user provided text as a string in to the 'query' argument in the tool.
    When the user's question does not require a context, use the 'ask_without_context' tool. For the 'ask_without_context' tool, provide the user provided text as a string into the 'input' argument in the tool."""
system_message_taot = create_system_message(system_message)
all_messages = [{"role": "system", "content": system_message_taot}]
    # Add previous messages (if available)
all_messages.extend(previous_messages)
    # Add current user prompt
# user_message = "hello, my name is robot"
# all_messages.append({"role": "user", "content": user_message})

# Wrapping in a function - note that history isn't used, as the memory is in the conversation_chain
# def chat(message, history):
#     all_messages.append({"role": "user", "content": message})
#     result =  agent.invoke({"messages": all_messages})
#     return result['messages'][0]['content']

# # And in Gradio:
# view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)


# result =  agent.invoke({"messages": all_messages})
# print(result['messages'][0]['content'])
