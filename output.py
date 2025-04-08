from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Load OpenRouter key from environment
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Embedding model
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# Website URL to crawl
BASE_URL = "https://www.object-automation.com/"
visited_urls = set()

def extract_all_links(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return set()
        soup = BeautifulSoup(response.text, "html.parser")
        links = set()
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(BASE_URL, href)
            if BASE_URL in full_url and full_url not in visited_urls:
                links.add(full_url)
        return links
    except:
        return set()

def extract_website_content(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return ""
        soup = BeautifulSoup(response.text, "html.parser")
        content = [tag.get_text(strip=True) for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"])]
        return " ".join(content) if content else "No relevant content found."
    except:
        return ""

def crawl_website(start_url, max_pages=50):
    to_visit = {start_url}
    website_data = {}
    while to_visit and len(website_data) < max_pages:
        url = to_visit.pop()
        visited_urls.add(url)
        content = extract_website_content(url)
        if content:
            website_data[url] = content
        new_links = extract_all_links(url)
        to_visit.update(new_links - visited_urls)
    return website_data

# Crawl and embed docs
website_data = crawl_website(BASE_URL, max_pages=50)
documents = list(website_data.values())
doc_urls = list(website_data.keys())

if not documents:
    raise RuntimeError("No website content was extracted.")

# Embeddings
batch_size = 10
embeddings = []
for i in range(0, len(documents), batch_size):
    batch = documents[i : i + batch_size]
    batch_embeddings = model.encode(batch)
    embeddings.append(batch_embeddings)

dim = len(embeddings[0][0])
embeddings = np.vstack(embeddings)

# FAISS
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings, dtype=np.float32))
faiss.write_index(index, "faiss_index.idx")

doc_map = {i: {"text": doc, "url": doc_urls[i]} for i, doc in enumerate(documents)}

class QueryRequest(BaseModel):
    query: str

def retrieve_relevant_docs(query, top_k=3):
    index = faiss.read_index("faiss_index.idx")
    query_embedding = model.encode([query])
    _, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    return [doc_map[idx] for idx in indices[0] if idx in doc_map]

PROMPT_TEMPLATE = """
You are an AI assistant providing accurate information from the company website.
Use the following website content to answer the user's question:

{context}

User Query: {query}

AI Response:
"""

COURSE_QUERY_KEYWORDS = ["courses", "training", "learning", "education", "curriculum"]
COURSE_RESPONSE = "Available courses: Azure, GenAI, Chip Design, 5G, Cyber Security, HPC, Quantum Computing, Data Science."

COURSE_QUERY_KEYWORDS = ["founder", "ceo",]
COURSE_RESPONSE = "Ganesan Narayanasamy"

def generate_response(query, context):
    if any(keyword in query.lower() for keyword in COURSE_QUERY_KEYWORDS):
        return COURSE_RESPONSE
    if not context.strip():
        return "The given website content does not provide information."

    prompt = PROMPT_TEMPLATE.format(context=context, query=query)

    headers = {
        "Authorization": OPENROUTER_API_KEY,
        "Content-Type": "application/json"
    }

    body = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return "Error generating response: " + str(e)

@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    query = request.query
    try:
        if any(keyword in query.lower() for keyword in COURSE_QUERY_KEYWORDS):
            return {"response": COURSE_RESPONSE}
        relevant_docs = retrieve_relevant_docs(query)
        if not relevant_docs:
            return {"response": "This content is not in the website."}
        context = " ".join([doc["text"] for doc in relevant_docs])
        response = generate_response(query, context)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
