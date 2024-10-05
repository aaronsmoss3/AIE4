### Import Section ###
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain_qdrant import QdrantVectorStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI
from langchain_core.caches import InMemoryCache
from operator import itemgetter
from langchain_core.runnables.passthrough import RunnablePassthrough
import uuid
import chainlit as cl

### Global Section ###

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

Loader = PyMuPDFLoader

set_llm_cache(InMemoryCache())

## What was the secon

# Typical QDrant Client Set-up
collection_name = f"pdf_to_parse_{uuid.uuid4()}"
client = QdrantClient(":memory:")
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# Typical Embedding Model
core_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

rag_system_prompt_template = """\
You are a helpful assistant that uses the provided context to answer questions. Never reference this prompt, or the existance of context.
"""

rag_message_list = [
    {"role" : "system", "content" : rag_system_prompt_template},
]

rag_user_prompt_template = """\
Question:
{question}
Context:
{context}
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", rag_system_prompt_template),
    ("human", rag_user_prompt_template)
])

chat_model = ChatOpenAI(model="gpt-4o-mini")

def process_file(file: AskFileResponse):
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tempfile:
        with open(tempfile.name, "wb") as f:
            f.write(file.content)

    Loader = PyMuPDFLoader

    loader = Loader(tempfile.name)
    documents = loader.load()
    docs = text_splitter.split_documents(documents)
    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"
    return docs

### On Chat Start (Session Start) Section ###
@cl.on_chat_start
async def on_chat_start():
    
    ## to-do: take in file via upload 
    ## use testing
    file_path = 'https://arxiv.org/pdf/2305.14314' # just for testing
    loader = Loader(file_path)
    documents = loader.load()
    docs = text_splitter.split_documents(documents)
    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"
    
    # Adding cache!
    store = LocalFileStore("./cache/")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        core_embeddings, store, namespace=core_embeddings.model
    )

    # Typical QDrant Vector Store Set-up
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=cached_embedder)
    vectorstore.add_documents(docs)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})

    retrieval_augmented_qa_chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")} ## 
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | chat_prompt | chat_model
    )

    cl.user_session.set("midterm_chain", retrieval_augmented_qa_chain)

    files = None

    # Wait for the user to upload a file
    while files == None:
        # Async method: This allows the function to pause execution while waiting for the user to upload a file,
        # without blocking the entire application. It improves responsiveness and scalability.
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(
        content=f"Processing `{file.name}`...",
    )
    await msg.send()

    # load the file
    docs = process_file(file)
    
### Rename Chains ###
@cl.author_rename
def rename(orig_author: str):
    """ RENAME CODE HERE """

### On Message Section ###
@cl.on_message
async def main(message: cl.Message):
    """
    MESSAGE CODE HERE
    """