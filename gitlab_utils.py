import os
import gitlab
import base64
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

def load_docs_to_chroma(docs, persist_directory="chroma_db"):
    # Drop the existing database by deleting the directory if it exists
    import shutil
    import os
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print(f"Dropped existing Chroma DB at {persist_directory}")

    # Create embeddings and vectorstore
    embeddings = OllamaEmbeddings(model="nomic-embed-text")  # or your preferred Ollama model
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"Loaded {len(docs)} documents into new Chroma DB at {persist_directory}")
    return vectordb

def list_group_repos(group_id):
    """List repositories in a GitLab group by group_id or group_path."""
    token = os.getenv("GITLAB_TOKEN")
    url = os.getenv("GITLAB_URL", "https://gitlab.com")
    if not token:
        raise ValueError("GITLAB_TOKEN not set in environment.")
    gl = gitlab.Gitlab(url, private_token=token)
    group = gl.groups.get(group_id)
    projects = group.projects.list(all=True)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    docs = []
    for project in projects:
        proj = gl.projects.get(project.id)
        default_branch = proj.default_branch or 'develop'
        if not default_branch:
            print(f"Skipping {project.name}: No default branch.")
            continue
        try:
            files = proj.repository_tree(recursive=True, all=True, ref=default_branch)
        except gitlab.exceptions.GitlabGetError as e:
            print(f"Skipping {project.name}: {e}")
            continue
        
        for file in files:
            print(f"Processing file: {file['path']} in Project: {project.name}")
            if file['type'] == 'blob' and file['path'].endswith('.md'):
                try:
                    file_obj = proj.files.get(file_path=file['path'], ref=default_branch)
                    content = base64.b64decode(file_obj.content).decode('utf-8')
                    split_chunks = text_splitter.split_text(content)
                    for chunk in split_chunks:
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "doc_type": "markdown",
                                "project_name": project.name,
                                "file_path": file['path']
                            }
                        )
                        docs.append(doc)
                except gitlab.exceptions.GitlabGetError as e:
                    print(f"Could not fetch {file['path']} in {project.name}: {e}")
        
    if docs:
        print(f"Found {len(docs)} markdown documents to load into Chroma.")
        load_docs_to_chroma(docs)
    else:
        print("No markdown documents found to load into Chroma.")    

    return [project.name for project in projects]
