import ollama
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings


def chat_with_ollama(prompt: str, model: str = "llama3:8b", persist_directory: str = "chroma_db") -> str:
    """
    Search Chroma DB for relevant context and pass it to Ollama for better answers.
    """
    # Load embeddings and vectorstore
    embeddings = OllamaEmbeddings(model="llama3:8b")
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    prompt = prompt.lower()
    # Debug: Print embedding for prompt
    try:
        prompt_embedding = embeddings.embed_query(prompt)
        print(f"[DEBUG] Embedding for prompt (first 10 values): {prompt_embedding[:10]}")
    except Exception as e:
        print(f"[DEBUG] Error computing embedding for prompt: {e}")

    # Retrieve top-k relevant documents
    print(f"Searching Chroma DB for prompt: {prompt}")
    # results = vectordb.max_marginal_relevance_search(
    #     prompt, k=4, fetch_k=10, lambda_mult=0.5
    # )
    results = vectordb.similarity_search(prompt, k=4)

    if results:
        print(f"Found {len(results)} relevant documents.")
        for i, doc in enumerate(results):
            print(f"Document {i+1} metadata: {doc.metadata}")
            print(f"Document {i+1} page content: {doc.page_content}")

        context = "\n---\n".join([f"Metadata: {doc.metadata}\nPage Content: {doc.page_content}" for doc in results])
        full_prompt = f"""Answer the User question: {prompt} \n
        If you don't know the answer, just say you don't know. Do not try to make up an answer. 
        If the answer is not in the context, just say you don't know. Return the metadata of the documents used to answer the question.
        \n use the following context to answer: {context}"""
    else:
        print("No relevant documents found.")
        full_prompt = prompt

    response = ollama.chat(model=model, messages=[{"role": "user", "content": full_prompt}])
    return response["message"]["content"]
