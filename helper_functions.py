import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import pinecone
from langchain.vectorstores.pinecone import Pinecone
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms.huggingface_hub import HuggingFaceHub


import pinecone


load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
pinecone_index = os.getenv("PINECONE_INDEX")
openai_key = os.getenv("OPENAI_API_KEY")

pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
index = pinecone.Index(pinecone_index)



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    
    # embeddings = HuggingFaceInstructEmbeddings(
    #    model_name="hkunlp/instructor-xl")
    # vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    
    #Pinecone.from_texts(text_chunks,embedding=embeddings,index_name=index)
    # vectorstore = Pinecone(index, embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm=ChatOpenAI()
    # llm = HuggingFaceHub(repo_id = "google/flan-t5-xxl", model_kwargs={"temperature":0.2, "max_length":502})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain