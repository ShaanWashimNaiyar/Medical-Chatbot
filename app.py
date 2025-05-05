import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub
from htmltemplate import css, bot_template, user_template
from sentence_transformers import SentenceTransformer
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_groq import ChatGroq


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    # Load Groq API key
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    # Initialize Groq LLM
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",  # Replaced mixtral-8x7b-32768
        temperature=0.3,                      # Consistent with original
        max_tokens=200                        # Shorter responses for demo
    )
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process your medical report PDFs first!")
        return
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="SRM MediBOT", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        
    st.header("Chat about your health with SRM MediBOT ⚕️")
    user_question = st.text_input("Ask a question about your medical report or prescription:")
    if user_question:
        handle_userinput(user_question)

    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your medical report or prescription", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):#That spinner icon is visible to let the users that app is processing and not frozen
                #get pdf text
                raw_text = get_pdf_text(pdf_docs)
                
                #get text chunks
                text_chunks = get_text_chunks(raw_text)
                
                #create vector store
                vectorstore = get_vectorstore(text_chunks)
                
                #create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
        
        
if __name__ == "__main__":
    main()