
import gradio as gr
from gitlab_utils import list_group_repos


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

    def on_click():
        output.update(value='<div style="text-align:center;"><span style="font-size:2em;">⏳</span><br>Loading...</div>')
        result = show_group_repos()
        output.update(value=result)

    btn.click(fn=show_group_repos, outputs=output)

demo.launch()
