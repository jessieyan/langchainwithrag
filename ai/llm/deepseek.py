"""DeepSeek LLM factory via DashScope-compatible OpenAI endpoint.

Requires the ``DASHSCOPE_API_KEY`` environment variable to be set.
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def deepseek_llm():
    """Return a ChatOpenAI instance targeting the DeepSeek-R1 model on DashScope."""
    llm = ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="deepseek-r1",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    return llm



