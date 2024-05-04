# import dependencies
import os
import requests
from operator import itemgetter
import chainlit as cl
from llama_parse import LlamaParse
from llama_parse.utils import ResultType
from llama_index.core import SimpleDirectoryReader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient

# load keys and variables
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv("LLAMA_CLOUD_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# pdf url
URL = "https://d18rn0p25nwr6d.cloudfront.net/CIK-0001326801/c7318154-f6ae-4866-89fa-f0c589f2ee3d.pdf"

# parse instructions
PARSING_INSTRUCTION = """The provided document is an annual report filed by Meta Platforms, Inc. with the Securities and Exchange Commission (SEC). 
This form provides detailed financial information about the company's performance for a specific year. 
It includes financial statements, management discussion and analysis, and other relevant disclosures required by the SEC.
It contains many tables and some signature pages.

Replace the signatures with tables containing the headers for each element.
"""

# rag prompt
RAG_PROMPT = """
CONTEXT:
{context}

QUERY:
{question}

The provided context is an annual report filed by Meta Platforms, Inc. with the Securities and Exchange Commission (SEC).
This form provides detailed financial information about the company's performance for a specific year.
It includes financial statements, management discussion and analysis, and other relevant disclosures required by the SEC.
It contains many tables and some signature pages. All members of the board need to sign the document.

Answer the query above only using the context provided. If you don't know the answer, simply say 'I don't know'.
"""


def create_new_vectorstore(collection_name, embeddings) -> Qdrant:
    data_file = "./data/output.md"
    # Check if the file exists
    if os.path.exists(data_file):
        print("The 10-K form has been parsed already. Using the cached version.")
    else:
        print("Cache is empty. Parsing will begin.")
        parse_10K_file()

    # load the document
    loader = DirectoryLoader(path="data/", glob="**/*.md", show_progress=True)
    documents = loader.load()

    # split the document into chunks

    # split markdown headers
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    md_text_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )

    md_splits = md_text_splitter.split_text(documents[0].page_content)

    # recursive character text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(md_splits)

    # create the vectorstore
    qdrant_vector_store = Qdrant.from_documents(
        documents=docs,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=collection_name,
    )

    return qdrant_vector_store


def get_vectorstore(client, collection_name, embeddings) -> Qdrant:

    try:
        vector_store = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings,
        )
        print(vector_store.embeddings)
    except:
        # create the vectorstore
        vector_store = create_new_vectorstore(collection_name, embeddings)

    return vector_store


def parse_10K_file() -> None:

    # Create the data directory if it doesn't exist
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # Check if the file already exists
    file_path = os.path.join(data_dir, "Meta_10k.pdf")
    if not os.path.exists(file_path):
        # Download the file
        url = URL
        response = requests.get(url)

        # Save the file to the data directory
        with open(file_path, "wb") as file:
            file.write(response.content)
        print("File downloaded successfully.")
    else:
        print("File already exists. Skipping download.")

    # setup parser
    parser = LlamaParse(
        result_type=ResultType.MD, parsing_instruction=PARSING_INSTRUCTION
    )

    # load and parse the documet
    file_extractor = {".pdf": parser}
    llama_parse_documents = SimpleDirectoryReader(
        input_files=["data/Meta_10k.pdf"], file_extractor=file_extractor
    ).load_data()

    # save markdown file
    data_file = "./data/output.md"
    with open(data_file, "a") as f:
        for doc in llama_parse_documents:
            f.write(doc.text + "\n")


@cl.on_chat_start
async def start() -> None:
    # instantiate embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # get vectorstore
    client = QdrantClient(
        QDRANT_URL,
        api_key=QDRANT_API_KEY,  # For Qdrant Cloud, None for local instance
    )
    collection_name = "meta_10k"
    qdrant_vectorstore = get_vectorstore(client, collection_name, embeddings)

    # setup our retriever
    qdrant_retriever = qdrant_vectorstore.as_retriever()

    # setup rag prompt
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

    # setup the rag chain
    chat_model = ChatOpenAI(model="gpt-3.5-turbo")

    rag_chain = (
        {
            "question": itemgetter("question"),
            "context": itemgetter("question") | qdrant_retriever,
        }
        | RunnablePassthrough().assign(context=itemgetter("context"))
        | {
            "response": rag_prompt | chat_model | StrOutputParser(),
            "context": itemgetter("context"),
        }
    )

    cl.user_session.set("rag_chain", rag_chain)
    await cl.Message(
        author="Assistant", content="Hi, I'm your Meta 10K Assistant! How can I help?"
    ).send()


@cl.on_message
async def main(message: cl.Message) -> None:
    chain = cl.user_session.get("rag_chain")
    cb = cl.AsyncLangchainCallbackHandler()
    cb.answer_reached = True
    # res=await chain.acall(message, callbacks=[cb])
    res = await chain.ainvoke(
        {"question": message.content}, config=RunnableConfig(callbacks=[cb])
    )
    print(f"response: {res}")
    answer = res["response"]

    await cl.Message(author="Assistant", content=answer).send()
