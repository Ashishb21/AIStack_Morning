from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"   # required but ignored
)

response = client.embeddings.create(
    model="mxbai-embed-large",
    input="Ollama provides local LLM inference"
)

embedding = response.data[0].embedding
print(len(embedding))
print(embedding)