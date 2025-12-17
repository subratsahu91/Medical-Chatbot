# step-1: Load raw PDF
# step-2: Create chunks
# Step-3: Create Vector Embadding
# step-4: Store Embadding into FAISS
# install : pipenv install langchain langchain_community langchain_huggingface faiss-cpu
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv(override=True)


# step-1: Load raw PDF

DATA_PATH = "data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents
documents =load_pdf_files(data=DATA_PATH)
#print('Length of PDF :',len(documents))

# step-2: Create chunks

def create_chunks(extrscted_data):
    text_splitter =RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extrscted_data)
    return text_chunks

text_chunks=create_chunks(extrscted_data=documents)
# print("length of Text Chunks:",len(text_chunks))

# Step-3: Create Vector Embadding

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return  embedding_model

embedding_model=get_embedding_model()

# step-4: Store Embadding into FAISS

DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks,embedding_model)
db.save_local(DB_FAISS_PATH)
