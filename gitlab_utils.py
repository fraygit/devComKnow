import os
import gitlab
import base64
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

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
    
    chunks = []
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
                    chunks.extend(text_splitter.split_text(content))
                    # print(f"File: {file['path']} in Project: {project.name}")
                    # print("Content:\n", content)
                except gitlab.exceptions.GitlabGetError as e:
                    print(f"Could not fetch {file['path']} in {project.name}: {e}")
        
    print(f"chunks: {chunks[7]}")

    return [project.name for project in projects]
