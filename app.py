import os
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from bs4 import BeautifulSoup
import requests

load_dotenv(dotenv_path=".env.local")
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)

def load_pdfs(folder):
    docs = []
    if not os.path.isdir(folder):
        return docs
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            docs.extend(PyPDFLoader(path).load())
    return docs

def crawl(url, depth=1, visited=None):
    visited = visited or set()
    if depth < 0 or url in visited:
        return []
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        docs = [Document(page_content=text, metadata={"source": url})]
        visited.add(url)
        for link in soup.find_all("a"):
            href = link.get("href")
            if href and href.startswith("/") and not href.startswith("//"):
                full_url = requests.compat.urljoin(url, href)
                if full_url.startswith("https://www.fhdw.de") and full_url not in visited:
                    docs.extend(crawl(full_url, depth - 1, visited))
        return docs
    except Exception as e:
        print(f"Fehler beim Crawlen: {url} → {e}")
        return []

def split(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(docs)

pdfs = load_pdfs("data")
web = crawl("https://www.fhdw.de", depth=1)
docs = pdfs + web
chunks = split(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = Chroma.from_documents(chunks, embedding=embeddings, collection_name="fhdw_knowledge")
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1)

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="answer"
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    memory=memory,
    return_source_documents=False
)

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "").strip()
    if not user_msg:
        return jsonify({"response": "Was möchtest du wissen?"})
    try:
        result = qa({"question": user_msg, "chat_history": memory.chat_memory})
        response = result["answer"]

        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": "Da ist leider etwas schiefgelaufen. Versuch es bitte später nochmal."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
