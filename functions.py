import getpass
import os
import pprint
import requests
import sys

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from icecream import ic
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import MongoDBAtlasVectorSearch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer


#initialize mongo client
def mongo_client_init(mongo_uri):
    mclient = MongoClient(mongo_uri, server_api=ServerApi('1'))
    return mclient



# Input: string.
# Output: list of floats.


def get_embedding(text):
    embedding_model = SentenceTransformer("thenlper/gte-small")
    if not text.strip():
        print("Attempted to get embedding for empty text.")
        return []

    embedding = embedding_model.encode(text)
    return list(embedding)

def vector_search(prompt,collection):
    """
    Perform a vector search in the MongoDB collection based on the user query.

    Args:
    user_query (str): The user's query string.
    collection (MongoCollection): The MongoDB collection to search.

    Returns:
    list: A list of matching documents.
    """

    # Generate embedding for the user query.
    query_embedding = get_embedding(prompt)

    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    # Define the vector search MongoDB aggregation pipeline.
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 5,  # Number of candidate documents to consider.
                "limit": 4,  # Return top 4 matches.
            }
        },
    ]

    # Execute the vector search MongoDB aggregation pipeline.
    results = collection.aggregate(pipeline)
    return list(results)


def get_search_result(query, collection):
    get_knowledge = vector_search(query, collection)
    search_result = ""
    for result in get_knowledge:
        search_result += f"{result.get('id', 'N/A')}\n"
        
    return search_result


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

