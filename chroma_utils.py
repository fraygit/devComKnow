import os
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import markdown2
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import PyPDF2
import ollama
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddings:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, convert_to_numpy=True).tolist()

def store_documents(folder: str, docs: list):
    embedding_name = os.getenv("NORMAL_EMBEDDING_NAME")
    collection_name = "others"
    if folder.lower() == "technical":
        embedding_name = os.getenv("TECHNICAL_EMBEDDING_NAME")
        collection_name = "technical"
    embeddings = SentenceTransformerEmbeddings(embedding_name)
    collection = Chroma(persist_directory=os.getenv("DB_NAME"), embedding_function=embeddings, collection_name=collection_name)
    collection.add_documents(docs)
    print(f"documents stored in {collection_name} collection.")
    return "Documents stored in Chroma DB."

def search_multi_collection(prompt: str):
    collections = ["others", "technical"]
    all_docs = []
    for collection in collections:
        embedding_name = os.getenv("NORMAL_EMBEDDING_NAME")
        if collection == "technical":
            embedding_name = os.getenv("TECHNICAL_EMBEDDING_NAME")

        embeddings = SentenceTransformerEmbeddings(embedding_name)
        print(f"Searching in collection: {collection} with embedding: {embedding_name}")
        collection = Chroma(persist_directory=os.getenv("DB_NAME"), embedding_function=embeddings, collection_name=collection)
        retrieved_docs = collection.similarity_search_with_score(prompt, k=5)
        # retrieved_docs = collection.max_marginal_relevance_search(
        #     prompt, k=5, fetch_k=100, lambda_mult=0.2
        # )  

        for doc, score in retrieved_docs:
            print(f"Document metadata: {doc.metadata}, score: {score}")
            if score < 500:
                all_docs.append(doc)

        # if collection == "technical":
        #     filtered_docs = [doc for doc, score in retrieved_docs if score < 500]
        #     all_docs.append(filtered_docs)
        # else:
        #     filtered_docs = [doc for doc, score in retrieved_docs if score < 2]
        #     all_docs.append(filtered_docs)

        # for i, (doc, score) in enumerate(retrieved_docs):
        #     print(f"Document {i+1} metadata: {doc.metadata}, score: {score}")  

        # print(f"-----")
        # for i in all_docs:
        #     print(f"{i.metadata} \n")


        # all_docs.extend(retrieved_docs)
    return all_docs
    # sorted_docs = sorted(all_docs, key=lambda x: x[1], reverse=False)
    # return [doc for doc, score in sorted_docs if score > 200]        
    
    # return all_docs

def load_docs_to_chroma(db_context_number: int, docs):
    if not docs:
        print("No documents to load into Chroma DB.")
        return
    persistent_directory = os.getenv(f"CONTEXT_DB_{db_context_number}_ID", "2")
    print(f"Loading documents into Chroma DB at {persistent_directory}")
    model = os.getenv("MODEL_NAME")
    embedding_name = os.getenv("EMBEDDING_NAME")
    embeddings = OllamaEmbeddings(model=embedding_name)

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

def summarize_text(system_prompt: str, user_prompt: str) -> str:
    model = os.getenv("MODEL_NAME")
    print("Summarizing text...")
    response = ollama.chat(model=model, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    print("Text summarized.")
    return response["message"]["content"]

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
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

                html = markdown2.markdown(content)
                soup = BeautifulSoup(html, 'html.parser')
                content_text = soup.get_text()
                chunks = text_splitter.split_text(content_text)
                for chunk in chunks:
                    metadata = {
                        "doc_type": "md",
                        "reference": file_name.lower(),
                        "file_path": file_name.lower()
                    }
                    doc = Document(page_content=chunk, metadata=metadata)
                    all_docs.append(doc)
                

                #load_docs_to_chroma(db_context_number, all_docs)
                store_documents(directory, all_docs)
        if file.lower().endswith('.pdf'):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)

            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""

                all_docs = []
                chunks = text_splitter.split_text(text)
                if not chunks or len(chunks) == 0:
                    continue
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
                #load_docs_to_chroma(db_context_number, all_docs)
                store_documents(directory, all_docs)

    return f"Loaded files from {directory} into Chroma DB."

def rephrase_prompt_for_chroma_db(prompt: str):
    """
    Rephrase the prompt for Chroma DB search.
    """
    model = os.getenv("MODEL_NAME")
    response = ollama.chat(model=model, messages=[
            {"role": "system", "content": "You are a helpful assistant that rephrases the prompt for Chroma DB search."}, 
            {"role": "user", "content": f"Return only the rephrased prompt. Prompt: {prompt}"}])
    return response["message"]["content"]

def search_chroma_db(prompt: str):
    """
    Search Chroma DB for relevant documents based on the prompt.
    """
    # prompt = rephrase_prompt_for_chroma_db(prompt)
    prompt = prompt.strip()
    print(f"prompt rephrased: {prompt}")

    model = os.getenv("MODEL_NAME")
    embedding_name = os.getenv("EMBEDDING_NAME")
    embeddings = OllamaEmbeddings(model=embedding_name)
    relevant_docs = []
    persistent_directory = os.getenv(f"CONTEXT_DB_1_ID", "chroma_db")
    vectordb = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

    try:
        results = vectordb.similarity_search_with_score(prompt, k=25)
        # results = vectordb.max_marginal_relevance_search(
        #     prompt, k=7, fetch_k=100, lambda_mult=0.5
        # )            
        relevant_docs.extend(results)

        # for doc in relevant_docs:
        #     print(f"Document metadata: {doc.metadata}")
    except Exception as e:
        print(f"Error searching Chroma DB: {e}")

    # for i, (doc, score) in enumerate(relevant_docs):
    #     print(f"Document {i+1} metadata: {doc.metadata}, score: {score}")
    # print(f"Found {len(relevant_docs)} relevant documents in Chroma DB.")
    sorted_docs = sorted(relevant_docs, key=lambda x: x[1], reverse=False)
    return [doc for doc, score in sorted_docs if score < 0.7]
    return relevant_docs

def fetch_all_chroma_docs(db_context_number: int):
    """
    Fetch all documents stored in ChromaDB and return as HTML string for Gradio output.
    """
    import html
    persistent_directory = os.getenv(f"CONTEXT_DB_{db_context_number}_ID", 2)

    if not os.path.exists(persistent_directory):
        return "No ChromaDB found."
    try:
        model = os.getenv("MODEL_NAME")
        embedding_name = os.getenv("EMBEDDING_NAME")
        embeddings = OllamaEmbeddings(model=embedding_name)
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


def search_flow_chroma_db(prompt: str):
    """
    Search Chroma DB for relevant documents based on the prompt.
    """
    # prompt = rephrase_prompt_for_chroma_db(prompt)
    prompt = prompt.strip()
    print(f"prompt rephrased: {prompt}")

    model = os.getenv("MODEL_NAME")
    embedding_name = os.getenv("EMBEDDING_NAME")
    embeddings = OllamaEmbeddings(model=embedding_name)

    context_db_1 = os.getenv("CONTEXT_DB_1_ID")
    vectordb = Chroma(persist_directory=context_db_1, embedding_function=embeddings)
    result_find_relevant_docs = vectordb.similarity_search_with_score(prompt, k=20, filter={"doc_type": "summary"})

    for doc, score in result_find_relevant_docs:
        print(f"item: {doc.metadata['file_path']}, score: {score}")

    sorted_docs = sorted(result_find_relevant_docs, key=lambda x: x[1], reverse=False)    
    result_find_relevant_docs = [doc for doc, score in sorted_docs if score < 1.4]
    list_relevant_docs = list(set([doc.metadata["file_path"] for doc in result_find_relevant_docs]))
    for doc in list_relevant_docs:
        print(f"d:{doc}")

    vectordb = Chroma(persist_directory=context_db_1, embedding_function=embeddings)
    if list_relevant_docs and len(list_relevant_docs) > 0:
        result = vectordb.similarity_search_with_score(prompt, k=10, filter={"file_path": {"$in": list_relevant_docs}})
        print(f"result: {result}")
    else:
        result = vectordb.similarity_search_with_score(prompt, k=10)
    # context_db_2 = os.getenv("CONTEXT_DB_2_ID")
    # vectordb = Chroma(persist_directory=context_db_2, embedding_function=embeddings)
    # result_find_relevant_docs = vectordb.similarity_search_with_score(prompt, k=10, filter={"doc_type": "summary"})
    # sorted_docs = sorted(result_find_relevant_docs, key=lambda x: x[1], reverse=False)
    # result_find_relevant_docs = [doc for doc, score in sorted_docs if score > 1.3]
    # for i, (doc, score) in enumerate(result_find_relevant_docs):
    #     print(f"Document {i+1} metadata: {doc.metadata}, score: {score}")


    return result