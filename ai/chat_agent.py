import os
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.tools.retriever import create_retriever_tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from embedding_model import get_embedding
from vector_store import get_vector_store
from llama_retriever import get_llama_retriever
from hybrid_retriever import get_hybrid_retriever
from keyword_retriever import get_keyword_retriever
from prompt import get_contextualize_prompt, get_answer_prompt

knowledge_base = "./knowledge_base/data"
#
# def question_answer_chat():
    # Create embeddings
# embed_model = get_embedding('dashscope')
#
#     # Create vector store
# vector_store = get_vector_store('milvus')
#
#     # Create llama index retriever
# vector_retriever = get_llama_retriever(vector_store=vector_store, embed_mode=embed_model, dir=knowledge_base)
#
#     # Create keyword retriever
# keyword_retriever = get_keyword_retriever(dir=knowledge_base)
#
#     # Create hybrid retriever
# hybrid_retriever = get_hybrid_retriever([vector_retriever, keyword_retriever], [0.5, 0.5])

    # create a new Chat with OpenAI
# llm = ChatOpenAI(
#         api_key=os.getenv("DASHSCOPE_API_KEY"),
#         base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#         model="deepseek-v3",
# )
# llm = ChatOpenAI(
#     model="deepseek-chat",
#     base_url="https://api.deepseek.com/v1",  # 👈 Required
#     api_key="sk-4511110ad1b343329d041b6faa649ba1",
#     temperature=0,
# )

# from langchain_deepseek import ChatDeepSeek
#
# llm = ChatDeepSeek(
#     model="deepseek-chat",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     # other params...
# )
# print(f"====={llm.supports_function_calling}====")
# messages = [
#     (
#         "system",
#         "You are a helpful assistant that translates English to French. Translate the user sentence.",
#     ),
#     ("human", "I love programming."),
# ]
# ai_msg = llm.invoke(messages)
# print(f"====ai message {ai_msg}=====")
# print(f"===content {ai_msg.content}====")

    # Create conversational retrieval chain
    # Contextualize question
    # contextualize_q_prompt = get_contextualize_prompt()
    #
    # history_aware_retriever = create_history_aware_retriever(
    #     llm, hybrid_retriever, contextualize_q_prompt
    # )
    #
    # # Answer question
    # qa_prompt = get_answer_prompt()
    # question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    #
    # rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Statefully manage chat history
    # store = {}
    #
    # def get_session_history(session_id: str) -> BaseChatMessageHistory:
    #     if session_id not in store:
    #         store[session_id] = ChatMessageHistory()
    #     return store[session_id]
    #
    # conversational_rag_chain = RunnableWithMessageHistory(
    #     rag_chain,
    #     get_session_history,
    #     input_messages_key="input",
    #     history_messages_key="chat_history",
    #     output_messages_key="answer",
    # )

    # memory = MemorySaver()
    #
    # ### Build retriever tool ###
    # tool = create_retriever_tool(
    #     vector_retriever,
    #     "test_retriever",
    #     "Searches and returns excerpts from the Autonomous Agents.",
    # )
    # tools = [tool]
    #
    # agent_executor = create_react_agent(llm, tools, checkpointer=memory)
    #
    # return agent_executor
import os
from openai import OpenAI

# Replace these with your model and provider
# API_KEY = "sk-4511110ad1b343329d041b6faa649ba1"
# BASE_URL = "https://api.deepseek.com/v1"  # or https://api.openai.com/v1 https://dashscope.aliyuncs.com/compatible-mode/v1
# MODEL = "deepseek-chat"  # or "gpt-3.5-turbo-0613", etc.

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # or https://api.openai.com/v1
MODEL = "deepseek-r1"  # or "gpt-3.5-turbo-0613", etc.

# Initialize client
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Define a test function
tools = [
    {
        "type": "function",
        "function": {
            "name": "test_func",
            "description": "Test function that multiplies two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            },
        },
    }
]

# Test message
messages = [{"role": "user", "content": "Use the tool to multiply 3 and 5."}]

try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    tool_calls = getattr(response.choices[0].message, "tool_calls", None)

    if tool_calls:
        print("✅ Tool calling is supported!")
        for call in tool_calls:
            print(f"Tool name: {call.function.name}")
            print(f"Arguments: {call.function.arguments}")
    else:
        print("❌ Tool calling NOT supported or not triggered.")
        print("Response message:", response.choices[0].message.content)

except Exception as e:
    print("❌ Error while testing function calling:", str(e))


from langgraph.graph import MessagesState, StateGraph

graph_builder = StateGraph(MessagesState)


from langchain_core.tools import tool


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


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
    return {"messages": [response]}
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition

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

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

from langgraph.prebuilt import create_react_agent
from IPython.display import Image, display
#
# agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)
# print(f"====={agent_executor}====")
# display(Image(agent_executor.get_graph().draw_mermaid_png()))



# Specify an ID for the thread
# config = {"configurable": {"thread_id": "abc123"}}

config = {"configurable": {"thread_id": "def234"}}

input_message = (
    "What is the standard method for Task Decomposition?\n\n"
    "Once you get the answer, look up common extensions of that method."
)

# for event in agent_executor.stream(
#     {"messages": [{"role": "user", "content": input_message}]},
#     stream_mode="values",
#     config=config,
# ):
#     event["messages"][-1].pretty_print()
#
# from langchain_core.messages import AIMessage, HumanMessage
#
# agent_executor = question_answer_chat()
# config = {"configurable": {"thread_id": "abc123"}}
#
#
# import gradio as gr
# def chat(message, history):
#     # result = conversation_chain.invoke({"question": message})
#     result = ""
#     for s in agent_executor.stream(
#             {"messages": [HumanMessage(content=message)]}, config=config
#     ):
#         print(s)
#         print("----")
#         result = s[agent][messages]
#     return result
#     # And in Gradio:
#
# view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)
#
