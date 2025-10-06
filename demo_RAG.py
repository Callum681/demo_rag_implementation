# --- 1. Setup ---
!pip install faiss-cpu sentence-transformers openai

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI

# Initialize embedding model and LLM client
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

# --- 2. Create document corpus ---
documents = [
    "The Eiffel Tower is located in Paris and was completed in 1889.",
    "The Colosseum in Rome is an ancient amphitheater built in 80 AD.",
    "The Great Wall of China is visible from space and spans thousands of miles.",
    "Mount Everest, the world’s tallest mountain, lies on the border between Nepal and China."
]

# --- 3. Create embeddings and FAISS index ---
embeddings = embedder.encode(documents, normalize_embeddings=True)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# --- 4. User query ---
query = "When was the Eiffel Tower built?"
query_emb = embedder.encode([query], normalize_embeddings=True)

# Retrieve top-2 most relevant documents
k = 2
_, indices = index.search(query_emb, k)
retrieved_docs = [documents[i] for i in indices[0]]

# --- 5. Construct augmented prompt ---
context = "\n".join(retrieved_docs)
prompt = f"""Answer the question using the context below.

Context:
{context}

Question: {query}
Answer:"""

# --- 6. Generate final answer using LLM ---
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("Retrieved context:")
print(context, "\n")
print("Answer:", response.choices[0].message.content)
