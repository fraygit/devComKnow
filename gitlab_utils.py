import os
import gitlab
import base64
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import time

load_dotenv()

def load_docs_to_chroma(docs, persist_directory="chroma_db"):

    print(f"Preparing to load {len(docs)} documents into Chroma DB at {persist_directory}")
    start_time = time.time()
    try:
        embeddings = OllamaEmbeddings(model="llama3:8b")
        if os.path.exists(persist_directory):
            print("Chroma DB directory exists. Updating existing DB.")
            vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            print("Adding new documents to existing Chroma DB...")
            vectordb.add_documents(docs)
            # Diagnostics: print total docs after adding
            try:
                all_docs = vectordb.get()
                print(f"Total documents in Chroma DB after add: {len(all_docs)}")
            except Exception as diag_e:
                print(f"[Diagnostics] Could not fetch all docs after add: {diag_e}")

        else:
            print("Chroma DB directory does not exist. Creating new DB.")
            vectordb = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=persist_directory
            )
        #vectordb.persist()
        elapsed = time.time() - start_time
        print(f"Loaded/updated {len(docs)} documents in Chroma DB at {persist_directory} in {elapsed:.2f} seconds.")
        return vectordb
    except Exception as e:
        print(f"Error during Chroma DB load/update: {e}")
        raise

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
    
    # Ensure output directory exists
    repo_dir = os.path.join(os.path.dirname(__file__), 'documents', 'repositories')
    os.makedirs(repo_dir, exist_ok=True)

    repo_txt_files = []
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

        repo_txt_path = os.path.join(repo_dir, f"{project.name}.md")
        repo_txt_files.append(repo_txt_path)
        with open(repo_txt_path, "w", encoding="utf-8") as f:
            f.write(f"This is about the repository {project.name}\n\n")
            for file in files:
                print(f"Processing file: {file['path']} in Project: {project.name}")
                if file['type'] == 'blob' and file['path'].endswith('.md'):
                    try:
                        file_obj = proj.files.get(file_path=file['path'], ref=default_branch)
                        content = base64.b64decode(file_obj.content).decode('utf-8')
                        f.write(f"# {file['path']}\n{content}\n\n")
                    except gitlab.exceptions.GitlabGetError as e:
                        print(f"Could not fetch {file['path']} in {project.name}: {e}")

    # After all files are written, split and load
    # docs = []
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # for txt_path in repo_txt_files:
    #     with open(txt_path, "r", encoding="utf-8") as f:
    #         content = f.read()
    #         chunks = text_splitter.split_text(content)
    #         repo_name = os.path.splitext(os.path.basename(txt_path))[0]
    #         for chunk in chunks:
    #             docs.append(Document(
    #                 page_content=chunk,
    #                 metadata={"doc_type": "repo_txt", "project_name": repo_name, "file_path": txt_path}
    #             ))

    # if docs:
    #     print(f"Found {len(docs)} .txt repo documents to load into Chroma.")
    #     load_docs_to_chroma(docs)
    # else:
    #     print("No .txt repo documents found to load into Chroma.")

    return [project.name for project in projects]
