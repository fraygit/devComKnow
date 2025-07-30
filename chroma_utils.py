import os
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import markdown2
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import PyPDF2
import ollama


def load_docs_to_chroma(db_context_number: int, docs):
    if not docs:
        print("No documents to load into Chroma DB.")
        return
    persistent_directory = os.getenv(f"CONTEXT_DB_{db_context_number}_ID", "2")
    print(f"Loading documents into Chroma DB at {persistent_directory}")
    model = os.getenv("MODEL_NAME", "llama3:8b")
    embeddings = OllamaEmbeddings(model=model)
    if os.path.exists(persistent_directory):
        vectordb = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
        vectordb.add_documents(docs)
    else:
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persistent_directory
        )
    print(f"Loaded Chroma DB from {persistent_directory} with model {model}")
    return vectordb


def load_files_to_chroma(directory: str, db_context_number: int):
    if not os.path.exists("documents"):
        os.makedirs("documents", exist_ok=True)
    files_dir = os.path.join(os.path.dirname(__file__), 'documents', directory)
    if not os.path.exists(files_dir):
        return "No repository documents directory found."
    
    for file in os.listdir(files_dir):
        print(f"Processing file: {file}")
        file_path = os.path.join(files_dir, file)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        if file.lower().endswith('.md'):
            header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")])
            all_docs = []
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                header_sections = header_splitter.split_text(content)
                for section in header_sections:
                    section_text = section.page_content
                    html = markdown2.markdown(section_text)
                    soup = BeautifulSoup(html, 'html.parser')
                    parts = []
                    if section.metadata.get('Header 1'):
                        parts.append(section.metadata.get('Header 1'))
                    if section.metadata.get('Header 2'):
                        parts.append(section.metadata.get('Header 2'))
                    if section.metadata.get('Header 3'):
                        parts.append(section.metadata.get('Header 3'))
                    parts.append(soup.get_text())
                    plain_text = '\n'.join(parts)
                    metadata = {
                        "doc_type": "md",
                        "reference": file_name.lower(),
                        "file_path": file_name.lower()
                    }
                    if hasattr(section, 'metadata') and section.metadata:
                        metadata.update({k: v for k, v in section.metadata.items() if v})
                    doc = Document(page_content=plain_text.lower(), metadata=metadata)
                    all_docs.append(doc)
                load_docs_to_chroma(db_context_number, all_docs)
        if file.lower().endswith('.pdf'):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)

            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                text = text.strip().lower()
                chunks = text_splitter.split_text(text)
                if not chunks or len(chunks) == 0:
                    continue
                all_docs = []
                for chunk in chunks:
                    text = chunk.strip()
                    if not text:
                        continue                
                    metadata = {
                        "doc_type": "pdf",
                        "reference": file_path.lower(),
                        "file_path": file_path.lower()
                    }
                    doc = Document(page_content=text, metadata=metadata)
                    all_docs.append(doc)
                    print(f"Extracted {len(text)} characters from {file_name}")
                load_docs_to_chroma(db_context_number, all_docs)

    return f"Loaded files from {directory} into Chroma DB."

def rephrase_prompt_for_chroma_db(prompt: str):
    """
    Rephrase the prompt for Chroma DB search.
    """
    model = os.getenv("MODEL_NAME", "llama3:8b")
    response = ollama.chat(model=model, messages=[
            {"role": "system", "content": "You are a helpful assistant that rephrases the prompt for Chroma DB search."}, 
            {"role": "user", "content": f"Return only the rephrased prompt. Prompt: {prompt}"}])
    return response["message"]["content"]

def search_chroma_db(prompt: str):
    """
    Search Chroma DB for relevant documents based on the prompt.
    """
    # prompt = rephrase_prompt_for_chroma_db(prompt)
    prompt = prompt.strip().lower()
    print(f"prompt rephrased: {prompt}")

    model = os.getenv("MODEL_NAME", "llama3:8b")
    embeddings = OllamaEmbeddings(model=model)
    number_of_context_db = int(os.getenv("NUMBER_OF_CONTEXT_DB", 2))
    relevant_docs = []
    for db_context_number in range(1, number_of_context_db + 1):
        persistent_directory = os.getenv(f"CONTEXT_DB_{db_context_number}_ID", "chroma_db")
        vectordb = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
        if not os.path.exists(persistent_directory):
            print(f"Chroma DB directory {persistent_directory} does not exist for context {db_context_number}.")
            continue

        try:
            results = vectordb.similarity_search_with_score(prompt, k=10)
            # results = vectordb.max_marginal_relevance_search(
            #     prompt, k=5, fetch_k=20, lambda_mult=0.5
            # )            
            relevant_docs.extend(results)
        except Exception as e:
            print(f"Error searching Chroma DB: {e}")

    for i, (doc, score) in enumerate(relevant_docs):
        print(f"Document {i+1} metadata: {doc.metadata}, score: {score}")
    print(f"Found {len(relevant_docs)} relevant documents in Chroma DB.")
    sorted_docs = sorted(relevant_docs, key=lambda x: x[1], reverse=False)
    return [doc for doc, score in sorted_docs if score > 1.3]

def fetch_all_chroma_docs(db_context_number: int):
    """
    Fetch all documents stored in ChromaDB and return as HTML string for Gradio output.
    """
    import html
    persistent_directory = os.getenv(f"CONTEXT_DB_{db_context_number}_ID", 2)

    if not os.path.exists(persistent_directory):
        return "No ChromaDB found."
    try:
        embeddings = OllamaEmbeddings(model="llama3:8b")
        vectordb = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
        # Use get() to retrieve all documents from the collection
        results = vectordb.get()  # This returns a dictionary with 'ids', 'documents', 'metadatas'
        docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(results['documents'], results['metadatas'])
        ]
        print(f"Fetched {len(docs)} documents from ChromaDB.")
        if not docs:
            return "No documents found in ChromaDB."
        html_rows = []
        for i, doc in enumerate(docs, 1):
            content = html.escape(doc.page_content)  # Preview first 300 chars
            meta = doc.metadata
            meta_html = "<br>".join(f"<b>{k}</b>: {html.escape(str(v))}" for k, v in meta.items())
            html_rows.append(f"<div style='margin-bottom:1em;'><b>Doc {i}</b><br>{meta_html}<br><pre style='background:#f8f8f8;padding:8px'>{content}</pre></div>")
        return "<h3>Documents in ChromaDB</h3>" + "".join(html_rows)
    except Exception as e:
        return f"Error fetching documents from ChromaDB: {e}"
