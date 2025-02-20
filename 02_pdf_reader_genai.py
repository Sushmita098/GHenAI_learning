from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings, AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
import os

# Set credentials for the EMBEDDING model (text-embedding-ada-002)
EMBEDDING_MODEL_NAME = "ADA"  # Ensure this matches your deployment name
EMBEDDING_API_KEY = "8229f1a35942488481fce09e030427c0"  # Replace with your actual API key for embeddings
EMBEDDING_ENDPOINT = "https://c27c3f15-3cb4-479b-876f-48d4a417b03b.openai.azure.com"
EMBEDDING_API_VERSION = "2024-02-01"

# Set credentials for the CHAT model (GPT-4o)
CHAT_MODEL_NAME = "gpt-4o"  # Ensure this matches your deployment name
CHAT_API_KEY = "c698e76d02f1499094d3d97b0d592543"  # Replace with your actual API key for chat
CHAT_ENDPOINT = "https://jllazureopenaigpt42.openai.azure.com/"
CHAT_API_VERSION = "2024-02-01"

# Load Azure OpenAI Embeddings (for document search)
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=EMBEDDING_MODEL_NAME,
    azure_endpoint=EMBEDDING_ENDPOINT,
    api_key=EMBEDDING_API_KEY,
    api_version=EMBEDDING_API_VERSION
)

# Load Azure OpenAI Chat Model (for answering questions)
llm = AzureChatOpenAI(
    azure_deployment=CHAT_MODEL_NAME,
    azure_endpoint=CHAT_ENDPOINT,
    api_key=CHAT_API_KEY,
    api_version=CHAT_API_VERSION,
    temperature=0
)

# Read PDF
pdf_path = r"C:\Users\SushmitaSen\Documents\sandbox\AC_invoice.pdf"
pdfreader = PdfReader(pdf_path)

raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

# Split text into chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

# Create FAISS vector store
document_search = FAISS.from_texts(texts, embeddings)

# Define retrieval & response generation chain
prompt_template = PromptTemplate(
    input_variables=["context", "input"],
    template="Based on the following document context: {context}, answer the question: {input}"
)

combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
retrieval_chain = create_retrieval_chain(document_search.as_retriever(), combine_docs_chain)

# Query the system
while True:
    query = input("Ask a question (type 'exit' to quit): ")
    
    if query.lower() == "exit":
        print("Exiting the system. Goodbye!")
        break
    
    response = retrieval_chain.invoke({"input": query})
    print("Answer:", response["answer"])

