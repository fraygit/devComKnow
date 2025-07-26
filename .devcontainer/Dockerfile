# Start from Ollama base image
FROM ollama/ollama:latest

# Install Python 3.12, pip, git, sqlite3, and spatialite (for chromadb compatibility)
RUN apt-get update && \
    apt-get install -y python3.12 python3-pip python3.12-venv git sqlite3 libsqlite3-mod-spatialite && \
    rm -rf /var/lib/apt/lists/*



# Install Python dependencies in a virtual environment
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /venv \
    && /venv/bin/pip install --upgrade pip \
    && /venv/bin/pip install -r /tmp/requirements.txt
ENV PATH="/venv/bin:$PATH"

# RUN ollama serve
# # # Pull llama3:8b model
# RUN ollama pull llama3:8b

# Set workdir
WORKDIR /workspace
