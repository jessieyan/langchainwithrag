from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
DASHSCOPE_API_KEY=""

def deepseek_llm():
    llm = ChatOpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="deepseek-r1",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    return llm



