import ollama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.memory import ConversationBufferMemory
import os
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain



from chroma_utils import search_chroma_db

def chat_using_langchain(prompt: str, history: list) -> str:    
    model = os.getenv("MODEL_NAME", "llama3:8b")
    llm = ChatOllama(model=model, temperature=0.7)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    embeddings = OllamaEmbeddings(model=model)
    db_name = os.getenv("CONTEXT_DB_1_ID", "chroma_db_repo")
    vectordb = Chroma(persist_directory=db_name, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "rephrase the question to be a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, 
        retriever, 
        contextualize_q_prompt
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context. "
            "if the answer is not in the context, just say you don't know."
            "At the end of the answer, include the meta data where the answer was found.\n\n"
            "Context: {context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain)
    
    response = rag_chain.invoke({
        "input": prompt,
        "chat_history": memory.chat_memory.messages
    })    

    return response["answer"]

def chat_with_ollama(prompt: str, model: str = "llama3:8b", persist_directory: str = "chroma_db") -> str:
    """
    Search Chroma DB for relevant context and pass it to Ollama for better answers.
    """
    # Load embeddings and vectorstore
    # embeddings = OllamaEmbeddings(model="llama3:8b")
    # vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    # prompt = prompt.lower()
    # Debug: Print embedding for prompt
    # try:
    #     prompt_embedding = embeddings.embed_query(prompt)
    #     print(f"[DEBUG] Embedding for prompt (first 10 values): {prompt_embedding[:10]}")
    # except Exception as e:
    #     print(f"[DEBUG] Error computing embedding for prompt: {e}")

    # Retrieve top-k relevant documents
    print(f"Searching Chroma DB for prompt: {prompt}")
    # results = vectordb.max_marginal_relevance_search(
    #     prompt, k=4, fetch_k=10, lambda_mult=0.5
    # )
    #results = vectordb.similarity_search(prompt, k=15)

    results = search_chroma_db(prompt)

    if results:
        print(f"Found {len(results)} relevant documents.")
        for i, doc in enumerate(results):
            print(f"Document {i+1} metadata: {doc.metadata}")

        context = "\n---\n".join([f"Metadata: {doc.metadata}\nPage Content: {doc.page_content}" for doc in results])

        full_prompt = f"""Answer the User question: {prompt} \n
        If you don't know the answer, just say you don't know. Do not try to make up an answer. 
        If the answer is not in the context, just say you don't know. At the end of the answer, include the metadata of the documents used to answer the question.
        \n use the following context to answer: {context[10:]}"""
    else:
        print("No relevant documents found.")
        full_prompt = prompt

    response = ollama.chat(model=model, messages=[{"role": "user", "content": full_prompt}])
    return response["message"]["content"]
