
import streamlit as st
import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

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
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
    base_llm = HuggingFaceEndpoint(
        huggingfacehub_api_token=HF_TOKEN,
        repo_id=huggingface_repo_id,
        temperature=0.5,
        max_new_tokens=512,
        task="conversational",
        model_kwargs={}
    )
    chat_llm = ChatHuggingFace(llm=base_llm)
    return chat_llm


def main():
    st.title("Ask Medibot")

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
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        Don't provide anything out of the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        HUGGINGFACE_REPO_ID =   'meta-llama/Meta-Llama-3-70B-Instruct'                 #"mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        if not HF_TOKEN:
            st.error("HF_TOKEN is not set. Please set it in your environment or .env file.")
            return

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load vectorstore")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})

            result = response['result']
            source_documents = response['source_documents']

            # result_to_show = result + "\n\n**Sources:**\n" + str(source_documents)


            short_sources = "\n".join([f"- {doc.metadata.get('source', '')} (p.{doc.metadata.get('page', '')})"
                                    for doc in source_documents])

            result_to_show = result + "\n\n**Sources:**\n" + short_sources




            st.chat_message("assistant").markdown(result_to_show)
            st.session_state.messages.append({"role": "assistant", "content": result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
