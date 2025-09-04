from openai import OpenAI
client = OpenAI(
        api_key="sk-mvjhwoypajnggqoasejoqumfaabvifdrvztgvmxdpdyukggy",
        base_url="https://api.siliconflow.cn/v1"
)
embedding = client.embeddings.create(input="hello world", model="Qwen/Qwen3-Embedding-8B",dimensions=1024)

print(embedding.data[0].embedding)
print(len(embedding.data[0].embedding))