import streamlit as st
import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from langchain_groq import ChatGroq   # GROQ LLM

from dotenv import load_dotenv
load_dotenv(override=True)

# Path to FAISS vector database
DB_FAISS_PATH = "vectorstore/db_faiss"


# Cache the vectorstore so it's not reloaded on every run
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])


# -----------------------------
# Load Groq LLM
# -----------------------------
def load_llm():
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY is missing. Add it to your .env file.")
        return None

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",  # ✅ Correct Groq Model
        temperature=0.2,
        max_tokens=512
    )
    return llm


# -----------------------------
# Main App
# -----------------------------

def is_context_relevant(docs, min_length=40):
    total = "".join([d.page_content for d in docs])
    return len(total.strip()) >= min_length

def main():
    st.title("Ask Medibot (Powered by Groq)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # Get user input
    prompt = st.chat_input("Ask your Question here ")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        If the context contains relevant information, use ONLY that to answer.
        If the context does NOT contain the answer, ignore the context and answer from your own medical knowledge safely.

        Context: {context}
        Question: {question}

        Begin your answer:
        """


        try:
            vectorstore = get_vectorstore()

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
            retrieved_docs = retriever.get_relevant_documents(prompt)

            llm = load_llm()

            # ----------------------------
            # HYBRID RAG DECISION
            # ----------------------------
            if is_context_relevant(retrieved_docs):
                # Use RAG answer
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )
                response = qa_chain.invoke({"query": prompt})

                result = response["result"]
                source_documents = response["source_documents"]

                short_sources = "\n".join([
                    f"- {doc.metadata.get('source')} (p.{doc.metadata.get('page')})"
                    for doc in source_documents
                ])

                final_answer = result + "\n\n**Sources:**\n" + short_sources

            else:
                # NO context → use general LLM knowledge
                final_answer = llm.invoke(prompt).content


            # Short, clean sources
            short_sources = "\n".join([
                f"- {doc.metadata.get('source', '')} (p.{doc.metadata.get('page', '')})"
                for doc in source_documents
            ])

            # Do not show sources when LLM says "I don't know"
            if "I don't know" in result or "I do not know" in result:
                final_answer = result
            else:
                final_answer = result + "\n\n**Sources:**\n" + short_sources

            st.chat_message("assistant").markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
