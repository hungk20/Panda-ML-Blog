from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

DATA_STORE_DIR = "data_store"

loader = TextLoader("./born_pink_world_tour.md")
docs = loader.load()

chunk_size = 150
chunk_overlap = 0
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""],
)
split_docs = text_splitter.split_documents(docs)

embedding = OpenAIEmbeddings()
vectordb = FAISS.from_documents(
    documents=split_docs,
    embedding=embedding,
)

vectordb.save_local(DATA_STORE_DIR)
