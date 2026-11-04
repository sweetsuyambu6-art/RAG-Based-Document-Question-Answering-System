# RAG-Based-Document-Question-Answering-System

This project is a Retrieval-Augmented Generation (RAG) application that allows users to ask questions about information stored in indexed documents. It uses OpenAI Embeddings to convert document content into vector representations and ChromaDB to store and retrieve relevant information. LangChain's MMR retrieval technique selects relevant and diverse document chunks, which are passed to the Mistral Small language model. Using prompt engineering, the system generates answers based only on the retrieved document context, helping reduce unsupported or irrelevant responses.

Technologies: Python, LangChain, RAG, OpenAI Embeddings, ChromaDB, Mistral Small, Prompt Engineering.
