import gradio as gr
from gitlab_utils import list_group_repos, load_docs_to_chroma
from ollama_chat import chat_using_langchain
from chroma_utils import load_files_to_chroma, fetch_all_chroma_docs


def show_group_repos():
    try:
        repos = list_group_repos("ruralco")
        if not repos:
            return "No repositories found."
        return "<br>".join(repos)
    except Exception as e:
        return f"Error: {e}"



with gr.Blocks(css="#repo-output { min-height: 200px; }") as demo:
    gr.Markdown("# GitLab Group Repositories: Ruralco")
    output = gr.HTML(elem_id="repo-output")
    btn = gr.Button("Load Repositories")
    btn_repo = gr.Button("Load Gitlab markdown files to Chroma")
    btn_show_chroma_repos = gr.Button("Show ChromaDB Repo Documents")
    btn_pdf_chroma = gr.Button("Load PDFs to Chroma")
    btn_show_chroma_docs = gr.Button("Show ChromaDB PDF Documents")

    def on_click():
        output.update(value='<div style="text-align:center;"><span style="font-size:2em;">⏳</span><br>Loading...</div>')
        result = show_group_repos()
        output.update(value=result)

    btn.click(fn=show_group_repos, outputs=output)
    btn_repo.click(fn=lambda: load_files_to_chroma("repositories", db_context_number=1), outputs=output)
    btn_pdf_chroma.click(fn=lambda: load_files_to_chroma("pdf", db_context_number=1), outputs=output)
    btn_show_chroma_repos.click(fn=lambda: fetch_all_chroma_docs(1), outputs=output)   
    btn_show_chroma_docs.click(fn=lambda: fetch_all_chroma_docs(1), outputs=output)

    # --- Chat UI ---
    gr.Markdown("## AI Chat (llama3:8b)")
    chatbot = gr.Chatbot()
    chat_input = gr.Textbox(placeholder="Ask a question about your repositories...", label="Your Question")
    send_btn = gr.Button("Send")

    def chat_fn(message, history):
        response = chat_using_langchain(message, history)
        history = history or []
        history.append((message, response))
        return history, ""

    send_btn.click(chat_fn, inputs=[chat_input, chatbot], outputs=[chatbot, chat_input])


demo.launch()
