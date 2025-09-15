from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_contextualize_prompt():
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return contextualize_q_prompt

def get_answer_prompt(userRoleMessage: str):
    
    system_prompt = (
        "{userRoleMessage}" if userRoleMessage else "You are an assistant for question-answering tasks. "
        """User query: "{input}" """
        "Before answering, determine if the user’s question depends on the previous conversation. "
        "If the question depends on context, use the chat history to form a complete answer. " 
        "If the question is self-contained, ignore the history and answer based only on the current input. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise. Here are some relevant documents:"
        "{context}"
        "If a video is available in {context} and relevant to the user question, respond with its title and video URI. Otherwise, return a helpful text answer."
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return qa_prompt

