import os
import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # Load .env if present

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = os.environ.get("HF_TOKEN")

# 🔐 Validate token
if not HF_TOKEN:
    st.error("Missing HF_TOKEN in environment. Please set it in a .env file or system variable.")
    st.stop()

# Load FAISS vectorstore
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Setup LLM endpoint
def load_llm(repo_id, token):
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        task="conversational",  # ✅ Required for Mistral
        huggingfacehub_api_token=token,
        temperature=0.5,
        max_new_tokens=512
    )

# Set prompt
def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Prompt string
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don’t know the answer, just say that you don’t know. Don’t try to make up an answer.
Don’t provide anything outside the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

# Streamlit UI
def main():
    st.title("💬 Ask MediBot!")

    # Session state for chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # Input box
    user_prompt = st.chat_input("Ask your medical question here...")

    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        with st.spinner("Thinking..."):
            try:
                vectorstore = get_vectorstore()

                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=False,  # 🔁 Turned off to skip showing sources
                    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                response = qa_chain.invoke({"query": user_prompt})
                result = response["result"]

                st.chat_message("assistant").markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})

            except Exception as e:
                st.error(f"💥 Error: {str(e)}")

if __name__ == "__main__":
    main()
