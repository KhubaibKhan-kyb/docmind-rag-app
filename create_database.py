#load pdf
#Split pdf into chunks
#Create embeddings
#Store in vector database 

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

data = PyPDFLoader("DocumentLoaders/biomimetics-08-00235-v2.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

embedding_model = OpenAIEmbeddings()

vector_store = Chroma.from_documents(chunks, embedding_model, persist_directory="chroma_db")