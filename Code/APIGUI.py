import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import os
import random
from sentence_transformers import CrossEncoder
from langchain.docstore.document import Document as LangchainDocument
from typing import Optional, List, Tuple
from transformers import AutoTokenizer
from transformers import AutoModel
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
import os
def document_parsing(file_path, chunk_size, MARKDOWN_SEPARATORS):
    """
    Document parser, processes uploaded document and splits text into chunks for a given chunksize.
    """
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()
    RAW_KNOWLEDGE_BASE = [
        LangchainDocument(page_content=doc.page_content, metadata={"source": doc.metadata["source"]}) for doc in pages
    ]
    

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # The maximum number of characters in a chunk: we selected this value arbitrarily
        chunk_overlap=chunk_size/10,  # The number of characters to overlap between chunks, 10%
        add_start_index=False,  # If `True`, includes chunk's start index in metadata
        strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in RAW_KNOWLEDGE_BASE:
        docs_processed += text_splitter.split_documents([doc])    
    
    return docs_processed

def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: str,
    ) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique

def embed_documents(EMBEDDING_MODEL_NAME, docs_processed):
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        #model_kwargs={"device": "mps"},
        encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
    )

    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )
    return KNOWLEDGE_VECTOR_DATABASE

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    MARKDOWN_SEPARATORS = [
        "\n#{1,6} ",
        "```\n",
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        "\n\n",
        "\n",
        " ",
        "",
    ]
    path = '../Data/content/pearce-et-al-2016-singing-together-or-apart-the-effect-of-competitive-and-cooperative-singing-on-social-bonding-within.pdf'
    EMBEDDING_MODEL_NAME = "thenlper/gte-small"
    chunksize = 512
    splitted_doc = document_parsing(path, chunk_size=chunksize, MARKDOWN_SEPARATORS=MARKDOWN_SEPARATORS)
    processed = split_documents(chunk_size=chunksize, knowledge_base=splitted_doc, tokenizer_name=EMBEDDING_MODEL_NAME)
    KNOWLEDGE_VECTOR_DATABASE = embed_documents(EMBEDDING_MODEL_NAME=EMBEDDING_MODEL_NAME, docs_processed=processed)