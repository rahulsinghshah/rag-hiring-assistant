import os
import faiss
import re
from sentence_transformers import SentenceTransformer

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

model = SentenceTransformer('all-MiniLM-L6-v2')

documents = []

# -----------------------------
# EXPERIENCE EXTRACTION
# -----------------------------
def extract_experience(text):
    try:
        text = text.lower()
        text = text.replace("–", "-").replace("—", "-")

        match = re.search(r'(\d+)\+?\s*year', text)
        if match:
            return int(match.group(1))

        match = re.search(r'(20\d{2})\s*-\s*(20\d{2})', text)
        if match:
            return max(0, int(match.group(2)) - int(match.group(1)))

        return 0
    except:
        return 0


# -----------------------------
# SKILL EXTRACTION
# -----------------------------
def extract_skills(text):
    try:
        skills = []

        skill_keywords = {
            "python": ["python"],
            "java": ["java"],
            "testing": ["testing", "qa", "quality assurance"],
            "frontend": ["react", "javascript", "html", "css"]
        }

        text_lower = text.lower()

        for skill, variants in skill_keywords.items():
            for word in variants:
                if word in text_lower:
                    skills.append(skill)
                    break

        return list(set(skills))
    except:
        return []


# -----------------------------
# LOAD DOCUMENTS
# -----------------------------
def load_documents(data_path="data"):
    global documents
    documents = []

    print("📂 Loading files...")

    if not os.path.exists(data_path):
        print("❌ Data folder not found!")
        return

    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)

        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)

            elif file.endswith(".txt"):
                loader = TextLoader(file_path)

            elif file.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(file_path)

            else:
                continue

            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = file

            documents.extend(docs)

        except Exception as e:
            print(f"❌ Failed loading {file}: {e}")

    print(f"✅ Loaded documents: {len(documents)}")


# -----------------------------
# SPLIT DOCUMENTS
# -----------------------------
def split_documents():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["source"] = chunk.metadata.get("source", "unknown")

    print(f"✂️ Total chunks: {len(chunks)}")

    return chunks


# -----------------------------
# VECTOR STORE
# -----------------------------
def create_vector_store(chunks):
    if not chunks:
        raise ValueError("❌ No chunks found. Check data folder.")

    texts = [doc.page_content for doc in chunks if doc.page_content.strip()]
    metadata = [doc.metadata for doc in chunks if doc.page_content.strip()]

    if not texts:
        raise ValueError("❌ All documents are empty.")

    embeddings = model.encode(texts)

    if len(embeddings.shape) < 2:
        raise ValueError("❌ Embedding failed.")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, texts, metadata


# -----------------------------
# BUILD RESUME INDEX
# -----------------------------
def build_resume_index(texts, metadata):
    resume_data = {}

    for text, meta in zip(texts, metadata):
        source = meta["source"]

        if source not in resume_data:
            resume_data[source] = {
                "full_text": "",
                "skills": [],
                "experience": 0
            }

        resume_data[source]["full_text"] += " " + text

    for source in resume_data:
        full_text = resume_data[source]["full_text"]

        resume_data[source]["skills"] = extract_skills(full_text)
        resume_data[source]["experience"] = extract_experience(full_text)

    return resume_data


# -----------------------------
# RETRIEVE
# -----------------------------
def retrieve(query, index, texts, metadata, k=5):
    try:
        query_embedding = model.encode([query])
        distances, indices = index.search(query_embedding, k)

        resume_data = build_resume_index(texts, metadata)

        required_years = extract_experience(query)
        required_skills = extract_skills(query)

        results = []

        for idx, i in enumerate(indices[0]):
            text = texts[i]
            meta = metadata[i]
            source = meta["source"]

            candidate_years = resume_data[source]["experience"]
            candidate_skills = resume_data[source]["skills"]

            keyword_score = 0

            for skill in required_skills:
                if skill in candidate_skills:
                    keyword_score += 3

            if required_years > 0 and candidate_years >= required_years:
                keyword_score += 3

            semantic_score = float(1 / (1 + float(distances[0][idx])))
            final_score = float(keyword_score + semantic_score)

            results.append({
                "text": text,
                "source": source,
                "candidate_skills": candidate_skills,
                "candidate_experience": int(candidate_years),
                "keyword_score": int(keyword_score),
                "semantic_score": round(semantic_score, 3),
                "final_score": round(final_score, 3)
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)
        results = [r for r in results if r["final_score"] > 0.5]

        return results

    except Exception as e:
        return [{"error": str(e)}]