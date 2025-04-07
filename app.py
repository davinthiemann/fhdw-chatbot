import os
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from bs4 import BeautifulSoup
import requests

load_dotenv(dotenv_path=".env.local")
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)

PDF_FOLDER = "data"

def load_all_pdfs(folder_path):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(path)
            docs = loader.load()
            all_docs.extend(docs)
    return all_docs

pdf_docs = load_all_pdfs(PDF_FOLDER)

def crawl_fhdw(start_url, depth=1, visited=None):
    if visited is None:
        visited = set()
    if depth < 0 or start_url in visited:
        return []

    documents = []
    try:
        response = requests.get(start_url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        documents.append(Document(page_content=text, metadata={"source": start_url}))
        visited.add(start_url)

        for link in soup.find_all("a"):
            href = link.get("href")
            if href and href.startswith("/") and not href.startswith("//"):
                full_url = requests.compat.urljoin(start_url, href)
                if full_url.startswith("https://www.fhdw.de") and full_url not in visited:
                    documents += crawl_fhdw(full_url, depth - 1, visited)
    except Exception as e:
        print(f"Fehler beim Scraping von {start_url}: {e}")
    return documents

web_docs = crawl_fhdw("https://www.fhdw.de", depth=1)
print(f"✅ {len(web_docs)} Webseiten-Seiten geladen.")

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)

all_docs = pdf_docs + web_docs
chunks = split_documents(all_docs)
print(f"📚 Gesamtanzahl Text-Chunks: {len(chunks)}")

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = Chroma.from_documents(chunks, embedding=embeddings, collection_name="fhdw_knowledge")
print("🧠 Vektor-Datenbank erfolgreich erstellt.")

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    memory=memory,
    return_source_documents=True
)

print("🔗 GPT-4o Chain mit Verlauf & Quellen erfolgreich erstellt.")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")

    if not user_message:
        return jsonify({"response": "⚠️ Bitte gib eine Nachricht ein."})

    try:
        result = qa_chain({
            "question": user_message,
            "chat_history": memory.chat_memory
        })

        sources = set()
        for doc in result["source_documents"]:
            if "source" in doc.metadata:
                sources.add(doc.metadata["source"])

        response_text = f"{result['answer']}\n\n📎 Quellen:\n" + ("\n".join(f"🔗 {src}" for src in sources) if sources else "Keine angegeben")

        return jsonify({"response": response_text})

    except Exception as e:
        return jsonify({"response": f"⚠️ Fehler: {str(e)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
