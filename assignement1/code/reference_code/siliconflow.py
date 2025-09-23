import requests
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv('./code/.env')

url = "https://api.siliconflow.cn/v1/chat/completions"

payload = {
    "model": "deepseek-ai/DeepSeek-R1",
    "messages": [
        {
            "role": "user",
            "content": "What opportunities and challenges will the Chinese large model industry face in 2025?"
        }
    ]
}
headers = {
    "Authorization": "Bearer " + os.getenv("SC_API_KEY"),
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)
response_data = response.json()
ai_reply = response_data['choices'][0]['message']['content']
print(ai_reply)