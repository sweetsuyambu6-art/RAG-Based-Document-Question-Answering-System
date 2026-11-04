{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPN8EvK/KNUomIgaCBFor33",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/sweetsuyambu6-art/RAG-Based-Document-Question-Answering-System/blob/main/RAG_Based_Document_Question_Answering_System_py.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "i-hIkVgNp3u_"
      },
      "outputs": [],
      "source": [
        "from dotenv import load_dotenv\n",
        "from langchain_huggingface import HuggingFaceEmbeddings\n",
        "from langchain_community.vectorstores import Chroma\n",
        "from langchain_mistralai import ChatMistralAI\n",
        "from langchain_core.prompts import ChatPromptTemplate\n",
        "\n",
        "load_dotenv()\n",
        "\n",
        "embedding_model = HuggingFaceEmbeddings(model = \"all-MiniLM-L6-v2\")\n",
        "\n",
        "vectorstore = Chroma(\n",
        "    persist_directory= \"chroma_db\",\n",
        "    embedding_function=embedding_model\n",
        ")\n",
        "\n",
        "retriever = vectorstore.as_retriever(\n",
        "    search_type = \"mmr\",\n",
        "    search_kwargs = {\n",
        "        \"k\" : 4,\n",
        "        \"fetch_k\":10,\n",
        "        \"lambda_mult\" :0.5\n",
        "    }\n",
        ")\n",
        "\n",
        "llm = ChatMistralAI(model = \"mistral-small-2506\")\n",
        "\n",
        "#prompt template\n",
        "prompt = ChatPromptTemplate.from_messages(\n",
        "    [\n",
        "        (\n",
        "            \"system\",\n",
        "            \"\"\"You are a helpful AI assistant.\n",
        "\n",
        "Use ONLY the provided context to answer the question.\n",
        "\n",
        "If the answer is not present in the context,\n",
        "say: \"I could not find the answer in the document.\"\n",
        "\"\"\"\n",
        "        ),\n",
        "        (\n",
        "            \"human\",\n",
        "            \"\"\"Context:\n",
        "{context}\n",
        "\n",
        "Question:\n",
        "{question}\n",
        "\"\"\"\n",
        "        )\n",
        "    ]\n",
        ")\n",
        "\n",
        "print(\"Rag system created \")\n",
        "\n",
        "print(\"press 0 to exit \")\n",
        "\n",
        "while True:\n",
        "    query = input(\"You : \")\n",
        "    if query == \"0\":\n",
        "        break\n",
        "\n",
        "    docs = retriever.invoke(query)\n",
        "\n",
        "    context = \"\\n\\n\".join(\n",
        "        [doc.page_content for doc in docs]\n",
        "    )\n",
        "\n",
        "    final_prompt = prompt.invoke({\n",
        "        \"context\" :context,\n",
        "        \"question\": query\n",
        "    })\n",
        "\n",
        "    response = llm.invoke(final_prompt)\n",
        "\n",
        "    print(f\"\\n AI: {response.content}\")\n",
        ""
      ]
    }
  ]
}
