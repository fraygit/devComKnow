# Developer Knowledgebase Chatbot

This project is a knowledge base chatbot designed to help users understand internal specifications, technical documentation, and other markdown documents. It leverages large language models (LLMs) via Ollama and provides a user-friendly chat interface built with Gradio.

## Features
- Chatbot interface for querying internal specs, technical docs, and markdown files
- Uses Ollama for local LLM inference
- Supports searching and referencing markdown documents
- Easy-to-use Gradio web interface

## Requirements
- Python 3.12+
- Ollama (for running LLMs locally)
- Gradio
- Required Python libraries (see below)

## Installation
1. **Clone the repository**
   ```bash
   git clone <this-repo-url>
   cd devComKnow
   ```
2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Install Ollama**
   - Download and install Ollama from [https://ollama.com/download](https://ollama.com/download)
   - Follow the instructions for your OS (Windows, macOS, Linux)

4. **Download Ollama models**
   - Start Ollama and download your preferred model (e.g., llama3, phi3, etc.):
     ```bash
     ollama pull llama3
     ```
   - You can list available models with:
     ```bash
     ollama list
     ```

## Document Folder Structure

1. **Create a folder named `documents` at the project root.**
2. **Inside `documents`, create a subfolder named `technical`.**

The application uses two distinct embedding models to optimize document processing:

### Technical Documents (`/documents/technical/*`)
- Uses the embedding model specified by `TECHNICAL_EMBEDDING_NAME` in your `.env`
- Optimized for technical content like API documentation, code specifications, and technical manuals
- Provides more accurate semantic search for specialized technical terminology

### Normal Documents (Other `/documents/*` folders)
- Uses the embedding model specified by `NORMAL_EMBEDDING_NAME` in your `.env`
- Better suited for general documentation, meeting notes, and non-technical content
- Provides efficient search for regular documentation needs

This dual-embedding approach ensures optimal search performance and accuracy for both technical and non-technical content. The system automatically applies the appropriate embedding model based on the document's location in the folder structure.

## Running the Application
Start the Gradio app with:
```bash
python app.py
```
This will launch a local web server with the chatbot interface.

## Notes
- Ensure Ollama is running before you start the Gradio app.
- You can change the model by updating the `OLLAMA_MODEL` variable in your `.env` file.
- For additional configuration or troubleshooting, refer to the documentation or contact the project maintainers.

---

This tool is intended to help teams and individuals quickly access and understand internal documentation, technical specs, and markdown-based knowledge bases using the latest advances in LLMs.
