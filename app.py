import gradio as gr
from gitlab_utils import list_group_repos, load_docs_to_chroma
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import markdown2
from bs4 import BeautifulSoup
from langchain_core.documents import Document
import os


def show_group_repos():
    try:
        repos = list_group_repos("ruralco")
        if not repos:
            return "No repositories found."
        return "<br>".join(repos)
    except Exception as e:
        return f"Error: {e}"


from ollama_chat import chat_with_ollama

with gr.Blocks(css="#repo-output { min-height: 200px; }") as demo:
    gr.Markdown("# GitLab Group Repositories: Ruralco")
    output = gr.HTML(elem_id="repo-output")
    btn = gr.Button("Load Repositories")
    btn_chroma = gr.Button("Load Docs to Chroma")

    def on_click():
        output.update(value='<div style="text-align:center;"><span style="font-size:2em;">⏳</span><br>Loading...</div>')
        result = show_group_repos()
        output.update(value=result)

    btn.click(fn=show_group_repos, outputs=output)

    def load_docs_to_chroma_ui():
        repo_dir = os.path.join(os.path.dirname(__file__), 'documents', 'repositories')
        if not os.path.exists(repo_dir):
            return "No repository documents directory found."
        txt_files = [f for f in os.listdir(repo_dir) if f.endswith('.md')]
        if not txt_files:
            return "No .md repository documents found."

        all_docs = []
        # Use MarkdownHeaderTextSplitter for structure-aware splitting
        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")])

        for txt_file in txt_files:
            print(f"Processing file: {txt_file}")
            txt_path = os.path.join(repo_dir, txt_file)
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()

                # Split by markdown headers first
                header_sections = header_splitter.split_text(content)
                repo_name = os.path.splitext(os.path.basename(txt_path))[0]
                for section in header_sections:
                    # Reconstruct text from headers and content
                    section_text = section.page_content
                    html = markdown2.markdown(section_text)
                    soup = BeautifulSoup(html, 'html.parser')
                    plain_text = soup.get_text()
                    
                    # Create Document with enriched metadata
                    metadata = {
                        "doc_type": "repo_md",
                        "project_name": repo_name.lower(),
                        "file_path": txt_path.lower()
                    }
                    if hasattr(section, 'metadata') and section.metadata:
                        metadata.update({k: v for k, v in section.metadata.items() if v})
                    
                    doc = Document(page_content=plain_text.lower(), metadata=metadata)
                    all_docs.append(doc)

        if all_docs:
            print(f"Finished processing all files. Total documents to load: {len(all_docs)}")
            load_docs_to_chroma(all_docs)
            return "Loaded documents into Chroma successfully."
        else:
            return "No documents were processed."

    btn_chroma.click(fn=load_docs_to_chroma_ui, outputs=output)

    # --- Chat UI ---
    gr.Markdown("## AI Chat (llama3:8b)")
    chatbot = gr.Chatbot()
    chat_input = gr.Textbox(placeholder="Ask a question about your repositories...", label="Your Question")
    send_btn = gr.Button("Send")

    def chat_fn(message, history):
        response = chat_with_ollama(message)
        history = history or []
        history.append((message, response))
        return history, ""

    send_btn.click(chat_fn, inputs=[chat_input, chatbot], outputs=[chatbot, chat_input])


demo.launch()
