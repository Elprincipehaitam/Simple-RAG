import requests
import numpy as np
from pathlib import Path

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text" 
CHAT_MODEL = "hf.co/unsloth/Llama-3.2-1B-Instruct-GGUF:Q4_K_M"

def embed(texts):
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": texts})
    return np.array(r.json()["embeddings"]) 

def ask_llm(prompt):
    r = requests.post(f"{OLLAMA}/api/chat", json={"model": CHAT_MODEL, "stream": False, "messages": [{"role": "user", "content": prompt}]})

    return r.json()["message"]["content"]

chunks = []
for file in Path("docs").glob("*.md"):
    chunks += [p.strip() for p in file.read_text(encoding="utf-8").split("\n\n") if p.strip()] 

chunk_vectors = embed(chunks)
print(f"Loaded {len(chunks)} chunks from docs")

def retrieve(question, k=3):
    q = embed([question])[0]
    scores = chunk_vectors @ q / (np.linalg.norm(chunk_vectors, axis=1) * np.linalg.norm(q))
    top_indices = np.argsort(scores)[:: -1][:k]
    return [chunks[i] for i in top_indices]

while True : 
    question = input("Enter your question (or 'exit' to quit): ")
    if question.lower() == "exit":
        break
    context = "\n\n".join(retrieve(question))
    prompt = f"""Answer the question using ONLY the context below. if the answer isn't in the context, say I don't know\n\nContext:\n{context}\n\nQuestion: {question}"""
    
    print("\n" + ask_llm(prompt) + "\n")


