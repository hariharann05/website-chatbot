from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Scraper function
def scrape_website(url: str, max_pages=10, delay=2):
    visited_urls = set()
    pages_to_visit = [url]
    all_content = []

    while pages_to_visit and len(visited_urls) < max_pages:
        current_url = pages_to_visit.pop(0)
        if current_url in visited_urls:
            continue

        try:
            response = requests.get(current_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            headings = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3'])]
            paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
            divs = [div.get_text(strip=True) for div in soup.find_all('div')]

            page_content = headings + paragraphs + divs
            all_content.extend(page_content)

            for link in soup.find_all('a', href=True):
                absolute_url = urljoin(url, link['href'])
                parsed_url = urlparse(absolute_url)
                if parsed_url.netloc == urlparse(url).netloc and absolute_url not in visited_urls:
                    pages_to_visit.append(absolute_url)

            visited_urls.add(current_url)
            time.sleep(delay)  # Rate limiting
        except Exception as e:
            print(f"Skipping {current_url}: {e}")

    return all_content

# Load embeddings model
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = InMemoryVectorStore(embeddings)
llm = OllamaLLM(model="gemma3")

# Scrape and index website content
company_website_url = "https://www.object-automation.com/"
website_texts = scrape_website(company_website_url)

if website_texts:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents = text_splitter.create_documents(website_texts)
    vector_store.add_documents(documents)

# Define request model
class ChatRequest(BaseModel):
    prompt: str

# Define retrieval and response function
def retrieve_and_answer(question):
    retrieved_docs = vector_store.similarity_search(question)
    
    # Ensure the chatbot only answers based on website content
    if not retrieved_docs:
        return "This content is not in the website."
    
    context = "\n".join([doc.page_content for doc in retrieved_docs])

     
    prompt = ChatPromptTemplate.from_template(
        """
        You are an AI assistant that must answer questions using only the provided website content. 
        If the answer cannot be found in the given context, respond strictly with: 'Not available on the website.' 
        Do not generate any external knowledge or assumptions beyond the provided information.
        
        Question: {question} 
        Context: {context} 
        Answer:
        """
    )
    chain = prompt | llm
    response = chain.invoke({"question": question, "context": context})

    # Ensure the response is not hallucinated
    if "This content is not in the website." in response or len(response.strip()) == 0:
        return "This content is not in the website."

    return response

# FastAPI endpoint
@app.post("/chat/")
async def chat(request: ChatRequest):
    try:
        answer = retrieve_and_answer(request.prompt)
        return JSONResponse(content={"response": answer})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)