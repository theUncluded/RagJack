# Import libraries.
import getpass
import gradio as gr
import os
import pprint
import requests
import sys
import weaviate


from gradio.themes.base import Base
from icecream import ic
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Weaviate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from weaviate.embedded import EmbeddedOptions

os.environ["OPEN_API_KEY"] = getpass.getpass()

directory_path = "txts"
sys.path.append(directory_path)
print(sys.path)

txts = os.listdir(directory_path)

print(txts)