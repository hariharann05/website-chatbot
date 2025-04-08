from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin

# Initialize FastAPI app
app = FastAPI()

# Load a smaller embedding model to reduce memory usage
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# Website URL (Replace with your target website)
BASE_URL = "https://www.object-automation.com/"

# Set to store unique links to avoid duplicates
visited_urls = set()

def extract_all_links(url):
    """ Extract all internal links from a webpage."""
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
    """ Extract text content from a webpage."""
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
    """ Crawl the entire website and extract content."""
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

# Crawl and process website data
website_data = crawl_website(BASE_URL, max_pages=50)
documents = list(website_data.values())
doc_urls = list(website_data.keys())

if not documents:
    raise RuntimeError("No website content was extracted.")

# Process embeddings in batches to avoid memory overflow
batch_size = 10
embeddings = []
for i in range(0, len(documents), batch_size):
    batch = documents[i : i + batch_size]
    batch_embeddings = model.encode(batch)
    embeddings.append(batch_embeddings)

dim = len(embeddings[0][0])
embeddings = np.vstack(embeddings)  # Convert to a single array

# FAISS Index with Disk Storage
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings, dtype=np.float32))
faiss.write_index(index, "faiss_index.idx")

# Store index-to-document mappings
doc_map = {i: {"text": doc, "url": doc_urls[i]} for i, doc in enumerate(documents)}

class QueryRequest(BaseModel):
    query: str

def retrieve_relevant_docs(query, top_k=3):
    """ Retrieve top-k relevant documents from FAISS. """
    index = faiss.read_index("faiss_index.idx")
    query_embedding = model.encode([query])
    _, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    return [doc_map[idx] for idx in indices[0] if idx in doc_map]

# Define Response Generation Logic
PROMPT_TEMPLATE = """
You are an AI assistant providing accurate information from the company website.
Use the following website content to answer the user's question:

{context}

User Query: {query}

AI Response:
"""

COURSE_QUERY_KEYWORDS = ["courses", "training", "learning", "education","curriculum"]
COURSE_RESPONSE = "Available courses: Azure, GenAI, Chip Design, 5G, Cyber Security, HPC, Quantum Computing, Data Science."

def generate_response(query, context):
    if any(keyword in query.lower() for keyword in COURSE_QUERY_KEYWORDS):
        return COURSE_RESPONSE
    if not context.strip():
        return "The given website content does not provide information."
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)
    try:
        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
        return response.get("message", {}).get("content", "The given website content does not provide information.").replace("\n", " ")
    except:
        return "The given website content does not provide information."

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