from openai import OpenAI
client = OpenAI(
        api_key="sk-mvjhwoypajnggqoasejoqumfaabvifdrvztgvmxdpdyukggy",
        base_url="https://api.siliconflow.cn/v1"
)
embedding = client.embeddings.create(input=["hello world","hihihi","uio euiw e"], model="Qwen/Qwen3-Embedding-8B",dimensions=512)

print(embedding.data[0].embedding)
print(embedding.data[1].embedding)
print(embedding.data[2].embedding)
