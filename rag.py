from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from groq import Groq
import chromadb
import requests
import uuid
import os
from dotenv import load_dotenv

# ---------------------------------------------------
# LOAD ENV VARIABLES
# ---------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in .env file")

# ---------------------------------------------------
# GROQ CLIENT
# ---------------------------------------------------
groq_client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------
# EMBEDDING MODEL
# ---------------------------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------
# CHROMA DATABASE
# ---------------------------------------------------
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(name="documents")

# ---------------------------------------------------
# TEXT SPLITTER
# ---------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

# ---------------------------------------------------
# CREATE EMBEDDINGS
# ---------------------------------------------------
def create_embeddings(texts):
    return embedding_model.encode(texts).tolist()

# ---------------------------------------------------
# INGEST DOCUMENT
# ---------------------------------------------------
def ingest_document(text, source_name):
    chunks = splitter.split_text(text)
    embeddings = create_embeddings(chunks)

    ids = [str(uuid.uuid4()) for _ in chunks]

    metadatas = [{"source": source_name} for _ in chunks]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(chunks)

# ---------------------------------------------------
# RETRIEVE DOCUMENTS
# ---------------------------------------------------
def retrieve(question):
    q_embedding = embedding_model.encode([question])[0].tolist()

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=3
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    return docs, metas

# ---------------------------------------------------
# GENERATE ANSWER USING GROQ
# ---------------------------------------------------
def generate_answer(question, docs):
    context = "\n\n".join(docs)

    prompt = f"""
You are an academic AI assistant.

RULES:
- Maximum 4 lines
- Simple language
- Direct answer only
- If not found say: "Answer not available in document."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=200
    )

    return response.choices[0].message.content

# ---------------------------------------------------
# SEND DATA TO N8N
# ---------------------------------------------------
def send_to_n8n(question, answer, sources):
    webhook_url = "https://sidraakhter.app.n8n.cloud/webhook/rag-log"

    data = {
        "question": question,
        "answer": answer,
        "sources": ", ".join(sources)
    }

    try:
        requests.post(webhook_url, json=data)
    except Exception as e:
        print("N8N ERROR:", e)

# ---------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------
def ask_question(question):
    docs, metas = retrieve(question)

    answer = generate_answer(question, docs)

    sources = list(set(meta["source"] for meta in metas))

    send_to_n8n(question, answer, sources)

    return {
        "answer": answer,
        "sources": sources
    }