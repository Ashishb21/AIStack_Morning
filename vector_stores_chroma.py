
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import numpy as np

## embeddings from Ollma ###
embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url="http://localhost:11434"
)

### sample documents ####
documents = [
    Document(page_content="The Great Fire of London happened in 1666."),
    Document(page_content="Julius Caesar was assassinated in 44 BCE."),
    Document(page_content="The Black Death killed nearly one-third of Europe."),
]


embedding_dim = len(embeddings.embed_query("hello world"))
print(embedding_dim)
index = faiss.IndexFlatL2(embedding_dim)

vector_store = Chroma(
    collection_name="my_collections",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

vector_store.add_documents(documents=documents,ids=["1","2","3"])

users_question = "Tell me something interesting about diseases in history"
#similar_docs = vector_store.similarity_search("users_question")
#print(similar_docs)

query_embedding = np.array([embeddings.embed_query(users_question)])

# Search
distances, indices = index.search(query_embedding, k=2)

for i in indices[0]:
    print(documents[i])