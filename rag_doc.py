import json
from collections import defaultdict
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain



# Initialize the LLM
llm = OllamaLLM(model="llama2")

# Define the prompt template
raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a technical assistant good at searching documents. If you do not have an answer from the provided information, say so. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)

# Initialize embeddings and database folder
embedding = FastEmbedEmbeddings()
folder_path = "db"

# Set up text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100, length_function=len, is_separator_regex=False)

# Load and process the PDF
local_pdf_path = r'C:\Users\karsh\OneDrive\Desktop\documents\RAG-implementation\A_First_Look_at_Generating_Website_Fingerprinting (1)[1].pdf'
loader = PDFPlumberLoader(local_pdf_path)
docs = loader.load_and_split()

chunks = text_splitter.split_documents(docs)

# Initialize vector store and populate with document chunks
vector_store = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=folder_path)

# Define retriever
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 20,
        "score_threshold": 0.25,
    }
)

document_chain = create_stuff_documents_chain(llm, raw_prompt)

chain = create_retrieval_chain(retriever, document_chain)

# File for saving user sessions
session_file = "user_sessions.json"

# Load user sessions from file
try:
    with open(session_file, "r") as f:
        user_sessions = json.load(f)
except FileNotFoundError:
    user_sessions = defaultdict(lambda: {"history": [], "context": ""})

def save_sessions():
    """Save the user_sessions dictionary to a JSON file."""
    with open(session_file, "w") as f:
        json.dump(user_sessions, f)

def process_query(user_id, query):
    """
    Process the query for a given user and maintain session history and context.
    
    Args:
        user_id (str): Unique identifier for the user.
        query (str): The user's query.

    Returns:
        str: The response from the chain or user-specific information.
    """
    session = user_sessions.setdefault(user_id, {"history": [], "context": ""})
    history = session["history"]
    context = session["context"]

    # Check if the user requested history
    if query.lower() == "history":
        if not history:
            return "No history found for this user."
        history_display = "\n".join(
            [f"Q: {item['query']}\nA: {item['response']}" for item in history]
        )
        return f"Chat History:\n{history_display}"

    # Append the current query to the context
    context += f"\nUser: {query}"

    # Invoke the chain with query and context
    result = chain.invoke({"input": query, "context": context})

    # Parse the response
    if isinstance(result, dict):
        answer_content = result.get('answer', "No answer provided.")
    else:
        answer_content = result or "Unexpected response format."

    # Update session history and context
    session["history"].append({"query": query, "response": answer_content})
    session["context"] += f"\nAssistant: {answer_content}"

    # Keep only the last 5 messages in history for context
    if len(session["history"]) > 5:
        session["history"] = session["history"][-5:]
        session["context"] = "\n".join(
            [f"User: {item['query']}\nAssistant: {item['response']}" for item in session["history"]]
        )

    # Save the updated sessions
    save_sessions()

    return answer_content

# Example usage with history and exit options
if __name__ == "__main__":
    print("Type 'exit' to quit or 'history' to view your chat history.")
    while True:
        user_id = input("Enter your user ID: ").strip()
        query = input("Enter your query: ").strip()
        
        if query.lower() == "exit":
            print("Exiting the chat. Goodbye!")
            break

        response = process_query(user_id, query)
        print("Response:\n", response)
