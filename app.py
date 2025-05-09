import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables (.env file)
load_dotenv()


# Extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text


# Split text into manageable chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ".", "!", "?", " "])
    chunks = splitter.split_text(text)
    return chunks


# Convert chunks to vector embeddings and store in FAISS
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# Create a RAG conversation chain with HuggingFace LLM
def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


# Handle question & display answer
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for message in st.session_state.chat_history:
        role = "User" if message.type == "human" else "Bot"
        st.write(f"**{role}:** {message.content}")


# Main Streamlit App
def main():
    st.set_page_config(page_title="MultiPDF Chatbot", page_icon="books")

    st.title("Chat with Multiple PDFs (RAG)")
    st.markdown("Upload PDFs, ask questions, and get accurate answers using a local RAG pipeline.")

    # Session state init
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # User question input
    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    # Sidebar for PDF Upload
    with st.sidebar:
        st.subheader("Your PDFs")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

        if st.button("Process"):
            with st.spinner("Processing documents..."):
                os.makedirs("temp", exist_ok=True)

                # Extract text
                raw_text = get_pdf_text(pdf_docs)

                # Split into chunks
                text_chunks = get_text_chunks(raw_text)

                # Generate vector embeddings
                vectorstore = get_vectorstore(text_chunks)

                # Create RAG chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

                st.success("Documents processed successfully!")


if __name__ == '__main__':
    main()
