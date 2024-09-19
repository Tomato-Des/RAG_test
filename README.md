# RAG Question-Answering System

This Python project implements a Retrieval-Augmented Generation (RAG) question-answering system. 
It uses the `langchain` library along with the `groq` API and `FAISS` vector store to retrieve and answer questions based on text documents.

## Overview

The script leverages a variety of libraries to load, split, and embed documents, and then uses a language model to generate answers based on the retrieved context. It supports:

- Loading documents from a specified directory
- Embedding documents and queries using `SentenceTransformer`
- Storing embeddings using `FAISS`
- Generating answers using the `ChatGroq` language model
- Interactive question-answering through a command-line interface

## Prerequisites

Before running the script, ensure you have the following dependencies installed. You can install them using pip install -r requirements.txt
pip install python-dotenv sentence-transformers langchain-groq langchain-community langchain-core
