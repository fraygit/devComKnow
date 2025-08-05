import gradio as gr
from gitlab_utils import list_group_repos, load_docs_to_chroma
from ollama_chat import chat_using_langchain, chat_with_ollama, chat_with_ollama_2
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
    btn_repo.click(fn=lambda: load_files_to_chroma("technical", db_context_number=2), outputs=output)
    btn_pdf_chroma.click(fn=lambda: load_files_to_chroma("pdf", db_context_number=1), outputs=output)
    btn_show_chroma_repos.click(fn=lambda: fetch_all_chroma_docs(2), outputs=output)   
    btn_show_chroma_docs.click(fn=lambda: fetch_all_chroma_docs(1), outputs=output)

 # --- Chat UI ---
    gr.Markdown("## AI Chat (llama3:8b)")
    chatbot = gr.Chatbot()
    chat_input = gr.Textbox(placeholder="Ask a question about your repositories...", label="Your Question")
    send_btn = gr.Button("Send")


    def chat_fn(message, history):
        history = history or []
        
        # Add user message to history
        history.append((message, ""))  # Start with empty response
        
        # Create a function to update the chat with streaming output
        def update_chat(new_text):
            nonlocal history
            if history and len(history[-1]) > 1:
                # Update the last message in history with new text
                history[-1] = (history[-1][0], new_text)
            return history
        
        # Call chat_with_ollama_2 with streaming
        try:
            # Initialize an empty response
            full_response = ""
            
            # Get the generator from chat_with_ollama_2
            response_generator = chat_with_ollama_2(
                message, 
                callback=lambda x: update_chat(f"{full_response}\n**System**: {x}")
            )
            
            # Stream the response
            for chunk in response_generator:
                if chunk:  # Only process non-empty chunks
                    full_response = chunk
                    update_chat(full_response)
                    yield history, ""
            
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            history[-1] = (message, error_msg)
            yield history, ""

    # Update the click event to use the new chat_fn
    send_btn.click(
        fn=chat_fn,
        inputs=[chat_input, chatbot],
        outputs=[chatbot, chat_input],
        show_progress=False
    )
    
    # Also trigger on Enter key in the chat input
    chat_input.submit(
        fn=chat_fn,
        inputs=[chat_input, chatbot],
        outputs=[chatbot, chat_input],
        show_progress=False
    )


demo.launch()
