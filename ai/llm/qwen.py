#  Copyright (c) 2025 深圳优儿智能科技有限公司
#  All rights reserved.

import os
from openai import OpenAI

os.environ["DASHSCOPE_API_KEY"] = ""

def chat_qwen(user_message: str, imageUrl : str):
    client = OpenAI(
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",)
    
    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""     # 定义完整回复 
    is_answering = False 

    completion = client.chat.completions.create(
        model="qvq-max",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": imageUrl
                        },
                    },
                    {"type": "text", "text": f"{user_message}"},
                ],
            },
        ],
        # 设置输出数据的模态，当前支持两种：["text","audio"]、["text"] modalities=["text", "audio"],
        modalities=["text"],
        # audio={"voice": "Chelsie", "format": "wav"},
        # stream 必须设置为 True，否则会报错
        stream=True,
        stream_options={
            "include_usage": True
        }
    )

    print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

    for chunk in completion:
        # 如果chunk.choices为空，则打印usage
        if not chunk.choices:
            print("\nUsage:")
            print(chunk.usage)
        else:
            delta = chunk.choices[0].delta
            print(f"delta is {delta}")
            # 打印思考过程
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                print(delta.reasoning_content, end='', flush=True)
                reasoning_content += delta.reasoning_content
            else:
                # 开始回复
                if delta.content != "" and is_answering is False:
                    print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                    is_answering = True
                # 打印回复过程
                answer_content += delta.content
    return answer_content
  
