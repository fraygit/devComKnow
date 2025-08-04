import ollama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.memory import ConversationBufferMemory
import os
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from chroma_utils import search_chroma_db, search_flow_chroma_db, search_multi_collection

def chat_using_langchain(prompt: str, history: list) -> str:    
    model = os.getenv("MODEL_NAME")
    llm = ChatOllama(model=model, temperature=0.3)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    embedding_name = os.getenv("EMBEDDING_NAME")
    embeddings = OllamaEmbeddings(model=embedding_name)
    db_name = os.getenv("CONTEXT_DB_1_ID")
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
            "At the end of the answer, include the meta data where the answer was found. Include the meta data from the context at the end.\n\n"
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
    model = os.getenv("MODEL_NAME")
    # Load embeddings and vectorstore
    # model = os.getenv("MODEL_NAME")
    # embeddings = OllamaEmbeddings(model=model)
    # context_db_1 = os.getenv("CONTEXT_DB_1_ID")
    # vectordb = Chroma(persist_directory=context_db_1, embedding_function=embeddings)
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
    # results = vectordb.max_marginal_relevance_search(
    #     prompt, k=8, fetch_k=10, lambda_mult=0.5)
    #results = vectordb.similarity_search(prompt, k=15)
    # print(f"Found {len(results)} relevant documents.")

    # results = search_chroma_db(prompt)
    # results = search_flow_chroma_db(prompt)
    results = search_multi_collection(prompt)

    if results:
        print(f"Found {len(results)} relevant documents.")
        # for i, doc in enumerate(results):
        #     print(f"Document {i+1} metadata: {doc.metadata}")

        # print(results)

        context = "\n---\n".join([f"Metadata: {doc.metadata}\nPage Content: {doc.page_content}" for doc in results])
        context = "\n---\n".join([f"Document: {doc}" for doc in results])

        full_prompt = f"""Answer the User question: {prompt} \n
        If you don't know the answer, just say you don't know. Do not try to make up an answer. 
        If the answer is not in the context, just say you don't know. At the end of the answer, include the metadata of the documents used to answer the question.
        \n use the following context to answer: {context}"""
    else:
        print("No relevant documents found.")
        full_prompt = prompt
    try:
        response = ollama.chat(model=model, messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Understand the context: {results}"}
            ])
        
        response2 = ollama.chat(model=model, messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Understand the question: {prompt}"}
        ])

        response3 = ollama.chat(model=model, messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Answer the question: {prompt}"}
        ])
    except Exception as e:
        print(f"Error model: {e}")
    return response3["message"]["content"]


def chat_with_ollama_2(prompt: str, model: str = "llama3:8b", persist_directory: str = "chroma_db") -> str:
    """
    Search Chroma DB for relevant context and implement a chain of thoughts pattern
    to provide better answers using Ollama.
    """
    model = os.getenv("MODEL_NAME")
    
    # Retrieve relevant documents
    print(f"Searching Chroma DB for prompt: {prompt}")
    results = search_multi_collection(prompt)

    if not results:
        print("No relevant documents found.")
        context = "No relevant context found."
    else:
        print(f"Found {len(results)} relevant documents.")
        # context = "\n---\n".join([f"Document {i+1}:\nMetadata: {doc.metadata}\nContent: {doc.page_content}" 
        #                        for i, doc in enumerate(results)])
        context = "\n---\n".join([f" {doc.page_content}" for doc in results])

    try:
        # Step 1: Understand the context
        system_prompt = ("You are an AI assistant that helps answer questions based on the provided context. "
                        "Think step by step and explain your reasoning.")
        
        # First, analyze the context
        analysis_prompt = f"""Analyze the following context and identify key information that might be relevant 
        to answering questions. Pay attention to:
        1. Main topics and themes
        2. Important facts or data points
        3. Any relationships between different pieces of information

        Context:
        {context}
        """
        
        print(f"Analysing context...")
        analysis = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": analysis_prompt}
            ]
        )
        print(f"Analysis: {analysis['message']['content']}")
        
        # Step 2: Understand and refine the question
        print(f"Refining question...")
        question_analysis = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": analysis["message"]["content"]},
                {"role": "assistant", "content": analysis["message"]["content"]},
                {"role": "user", "content": f"""
                Based on the context analysis, let's analyze the user's question to better understand what they're asking.
                
                User's question: {prompt}
                
                Please:
                1. Rephrase the question in your own words
                2. Identify any ambiguities or missing information
                3. Break down the question into sub-questions if needed
                4. Determine what information from the context is most relevant
                """}
            ]
        )
        print(f"Question analysis: {question_analysis['message']['content']}")
        
        # Step 3: Generate the final answer using chain of thought
        print(f"Generating final answer...")
        final_response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": analysis["message"]["content"]},
                {"role": "assistant", "content": analysis["message"]["content"]},
                {"role": "user", "content": question_analysis["message"]["content"]},
                {"role": "assistant", "content": question_analysis["message"]["content"]},
                {"role": "user", "content": f"""
                Based on the context analysis and question breakdown, please provide a detailed answer to:
                {prompt}
                
                Structure your response with:
                1. A brief summary of the question
                2. Step-by-step reasoning using the context
                3. The final answer
                4. Sources from the context that support your answer
                
                If the answer cannot be determined from the context, please state that clearly.

                Here is the detailed context again for reference:
                {context}
                """}
            ]
        )
        
        return final_response["message"]["content"]
        
    except Exception as e:
        print(f"Error in chat_with_ollama: {e}")
        return "I encountered an error while processing your request. Please try again later."
