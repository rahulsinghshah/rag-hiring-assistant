from fastapi import FastAPI
from backend.rag import load_documents, split_documents, create_vector_store, retrieve
from backend.utils import generate_answer

app = FastAPI(title="RAG Hiring Assistant")

print("🔄 Starting system...")

load_documents()
chunks = split_documents()
index, texts, metadata = create_vector_store(chunks)

print("✅ System Ready!")

@app.get("/")
def home():
    return {"message": "RAG Hiring Assistant Running 🚀"}

@app.get("/query")
def query(q: str):
    try:
        results = retrieve(q, index, texts, metadata)
        answer = generate_answer(q, results)

        return {
            "query": q,
            "answer": answer,
            "results_count": len(results),
            "sources": results
        }

    except Exception as e:
        return {"error": str(e)}