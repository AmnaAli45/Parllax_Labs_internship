import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  

groq_client=Groq(api_key=os.environ.get("GROQ_API_KEY"))

response=groq_client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role":"user","content":"what is RAG give in a single line"}
    ],
)

print(response.choices[0].message.content)