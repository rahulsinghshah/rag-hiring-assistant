# rag-hiring-assistant
AI-powered hiring assistant using Retrieval-Augmented Generation (RAG) with semantic search, hybrid ranking, and resume parsing.

# 🚀 RAG Hiring Assistant

An AI-powered hiring assistant that retrieves and ranks candidates based on job requirements using a **RAG (Retrieval-Augmented Generation) pipeline**.

---

## 🔥 Features

* 📂 Multi-format resume ingestion (PDF, TXT, DOCX)
* 🧠 Semantic search using embeddings
* ⚡ FAISS vector database for fast retrieval
* 🔍 Hybrid search (semantic + keyword matching)
* 📊 Re-ranking system for better candidate selection
* 🧾 Resume parsing (skills + experience extraction)
* ✅ Explainable results (why candidate was selected)
* 🚀 FastAPI backend for API access

---

## 🧠 How It Works

1. Load resumes from `/data` folder
2. Split into chunks
3. Generate embeddings using Sentence Transformers
4. Store vectors in FAISS
5. Retrieve relevant candidates
6. Apply hybrid scoring + re-ranking
7. Return structured and explainable results

---

## 🏗️ Tech Stack

* Python
* FastAPI
* FAISS
* Sentence Transformers
* LangChain (document loaders & splitters)

---

## 📂 Project Structure

```
rag-hiring-assistant/
│
├── backend/
│   ├── main.py
│   ├── rag.py
│   ├── utils.py
│
├── data/
│   ├── resumes...
│
└── README.md
```

---

## ⚙️ Installation

```bash
pip install fastapi uvicorn faiss-cpu sentence-transformers langchain langchain-community langchain-text-splitters pymupdf unstructured python-docx
```

---

## ▶️ Run the Project

```bash
uvicorn backend.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## 🔍 Example Query

```
manual testing 2 years
```

---

## 📊 Example Output

* Candidate skills
* Experience (years)
* Match score
* Explanation of selection

---

## 🧠 Key Concepts Implemented

* Retrieval-Augmented Generation (RAG)
* Vector similarity search
* Hybrid retrieval (semantic + keyword)
* Re-ranking strategies
* Structured information extraction

---

## 🚀 Future Improvements

* Add LLM for natural language responses
* Build frontend UI
* Add resume scoring (0–100%)
* Deploy on cloud

---

## 🎯 Interview Explanation

> Built a RAG-based hiring assistant with hybrid retrieval and re-ranking, including structured parsing for skills and experience to improve candidate matching accuracy.

---

## 📌 Author

Rahul Singh Shah
