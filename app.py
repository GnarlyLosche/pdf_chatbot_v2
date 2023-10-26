#NOTES
#CONSIDER SWITCHING TO PDFPLUMBER
# NOTICE THAT I NEED TO ADD LLAMA INDEX HERE TO BETTER DELINEATE/COMPARE DOCUMENTS
# Consider using Emberv1 Embeddings from huggingface or similar - using instructor


import streamlit as st
from dotenv import load_dotenv

#CONSIDER SWITCHING TO PDFPLUMBER
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import pinecone
from langchain.llms.huggingface_hub import HuggingFaceHub

from helper_functions import get_pdf_text, get_text_chunks, get_vectorstore, get_conversation_chain

from htmlTemplates import css, bot_template, user_template

def handle_userinput(user_query):
    response = st.session_state.conversation({'question': user_query})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():

    load_dotenv()

    st.set_page_config(page_title='Analyze Multiple PDFs',page_icon="/Users/charlielosche/Documents/Docs/Coding/python_pdf_chat/assets/DALL·E 2023-10-18 19.51.23 - Sourcer_Icon.png")

    st.write(css, unsafe_allow_html=True)
    
    # If I'm going to use st.session_state to maintain any variables, I need to initialize them on load
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.image("/Users/charlielosche/Documents/Docs/Coding/python_pdf_chat/assets/DALL·E 2023-10-18 19.51.23 - Sourcer_Logo.png")
    st.header("Analyze Multiple PDFs :books:")

    user_question = st.text_input("Ask a question about your documents")
    if user_question:
        handle_userinput(user_question)

    st.write(user_template.replace("{{MSG}}", "Ask the chatbot any question about your document."), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello, what document do you need to review?"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("your documents")
        pdf_docs = st.file_uploader("Upload Your PDFs here click 'Run'", accept_multiple_files=True)
        if st.button("Run"):
            with st.spinner("Processing..."):
                # Get pdf text to just pull the raw contents from the pdfs
                # NOTICE THAT I NEED TO ADD LLAMA INDEX HERE TO BETTER DELINEATE/COMPARE DOCUMENTS
                raw_text = get_pdf_text(pdf_docs)
                
                # Get the pdf text segments
                text_chunks = get_text_chunks(raw_text)
                
                # Create vector store
                vectorstore = get_vectorstore(text_chunks)

                #Create instance of conversation chain
                # use st.session_state.converstion if I want the variable to remain throughout the session
                # If I left st.session_state.conversation in here, it would be stuck in the sidebar
                st.session_state.conversation = get_conversation_chain(vectorstore)

    

if __name__ == '__main__':
    main()