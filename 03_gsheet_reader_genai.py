import pandas as pd

from langchain_openai import OpenAIEmbeddings, AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
import streamlit as st

st.title("GSheet Q&A Chatbot")
# Set credentials for the EMBEDDING model (text-embedding-ada-002)
EMBEDDING_MODEL_NAME = "TES"  # Ensure this matches your deployment name
EMBEDDING_API_KEY = ""  # Replace with your actual API key for embeddings
EMBEDDING_ENDPOINT = ""
EMBEDDING_API_VERSION = "2024-02-01"

# Set credentials for the CHAT model (GPT-4o)
CHAT_MODEL_NAME = "gpt-4"  # Ensure this matches your deployment name
CHAT_API_KEY = ""  # Replace with your actual API key for chat
CHAT_ENDPOINT = ""
CHAT_API_VERSION = "2024-02-01"


# embeddings = AzureOpenAIEmbeddings(
#     azure_deployment=EMBEDDING_MODEL_NAME,
#     azure_endpoint=EMBEDDING_ENDPOINT,
#     api_key=EMBEDDING_API_KEY,
#     api_version=EMBEDDING_API_VERSION
# )

# # Load Azure OpenAI Chat Model (for answering questions)
# llm = AzureChatOpenAI(
#     azure_deployment=CHAT_MODEL_NAME,
#     azure_endpoint=CHAT_ENDPOINT,
#     api_key=CHAT_API_KEY,
#     api_version=CHAT_API_VERSION,
#     temperature=0
# )

sheet_id='18OCY9e6L2kN2Abwvn4Sj_Il4-Vn3rezEHQBqdaO1_Xw'


@st.cache_data(show_spinner=False)
def load_text_chunks():
    df = pd.read_csv(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv")
    text_data = df.to_csv(index=False)
    
    # Split text
    text_splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size=800, 
        chunk_overlap=200
    )
    text_chunks = text_splitter.split_text(text_data)
    
    # Convert into Document objects
    docs = [Document(page_content=chunk) for chunk in text_chunks]
    return docs

# 2) Build FAISS vector store with @st.cache_resource 
#    (FAISS is not picklable, so we can't use st.cache_data)
@st.cache_resource(show_spinner=False)
def build_vector_store():
    docs = load_text_chunks()
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=EMBEDDING_MODEL_NAME,
        azure_endpoint=EMBEDDING_ENDPOINT,
        api_key=EMBEDDING_API_KEY,
        api_version=EMBEDDING_API_VERSION
    )
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# 3) Build retrieval chain with @st.cache_resource
@st.cache_resource(show_spinner=False)
def load_chain():
    llm = AzureChatOpenAI(
        azure_deployment=CHAT_MODEL_NAME,
        azure_endpoint=CHAT_ENDPOINT,
        api_key=CHAT_API_KEY,
        api_version=CHAT_API_VERSION,
        temperature=0
    )
    prompt_template = PromptTemplate(
        input_variables=["context", "input"],
        template=(
            "You are a helpful assistant. Use only the following document context to "
            "answer the user question. If you are not sure, say you don't know.\n\n"
            "Document context:\n{context}\n\nQuestion: {input}"
        )
    )
    combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)

    vector_store = build_vector_store()
    retrieval_chain = create_retrieval_chain(
        vector_store.as_retriever(), 
        combine_docs_chain
    )
    return retrieval_chain

def get_response(user_query: str) -> str:
    chain = load_chain()
    response = chain.invoke({"input": user_query})
    return response["answer"]

# 4) Setup session state for chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# 5) Display chat history so far
for role, content in st.session_state["chat_history"]:
    with st.chat_message(role):
        st.write(content)

# 6) Chat input
user_input = st.chat_input("Ask a question about the data...")
if user_input:
    # a) Add user message to the chat history
    st.session_state["chat_history"].append(("user", user_input))
    with st.chat_message("user"):
        st.write(user_input)

    # b) Generate the response
    with st.spinner("Thinking..."):
        answer = get_response(user_input)

    # c) Add assistant answer to the chat history
    st.session_state["chat_history"].append(("assistant", answer))
    with st.chat_message("assistant"):
        st.write(answer)