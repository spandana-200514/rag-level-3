import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


# ============================================================
# EMBEDDING MODEL
# ============================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)


# ============================================================
# LOAD PDF
# ============================================================

def load_pdf(pdf_path):

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    return documents


# ============================================================
# CHUNKING
# ============================================================

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    return chunks


# ============================================================
# CREATE VECTOR STORE
# ============================================================

def create_vector_store(chunks):

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="chroma_db"
    )

    return vectorstore


# ============================================================
# LOAD EXISTING VECTOR STORE
# ============================================================

def load_vector_store():

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )

    return vectorstore


# ============================================================
# RAG ANSWER
# ============================================================

def generate_answer(question, documents, chat_history):

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    history_text = ""

    for message in chat_history:

        history_text += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )


    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful PDF question-answering assistant.

Use the provided context to answer the question.

Conversation History:
{history}

Context:
{context}

Question:
{question}

Instructions:
- Answer using the context.
- Use conversation history when necessary.
- If the answer is not available in the context, say:
  "I could not find the answer in the provided PDF."
- Do not make up information.

Answer:
"""
    )


    formatted_prompt = prompt.format(
        history=history_text,
        context=context,
        question=question
    )


    response = llm.invoke(formatted_prompt)

    return response.content