from flask import Flask
from flask import Flask, jsonify
from flask_cors import CORS
from flask_cors import cross_origin
import os
from flask import request

from langchain.chat_models import ChatOllama

app = Flask(__name__)
# Allow requests from your Angular app's domain (replace 'http://localhost:4200' with your Angular app's URL)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:4200"}})
# import torch

# import gc
# torch.cuda.empty_cache()
# gc.collect()

# Load web page
import argparse
from langchain.document_loaders import DirectoryLoader

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embed and store
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.embeddings import OllamaEmbeddings  # We can also try Ollama embeddings

from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import OpenAIEmbeddings

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain import hub
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ChatMessageHistory
from threading import Thread
from queue import SimpleQueue

import requests

__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


class Ollama_:
    def __init__(self, port, model):
        self.port = port
        self.model = model
        self.llm = None
        self.qa_chain = None
        self.vectorstore = None

    def download_model(self):

        print('------------------------')

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
        }
        data = '{\n  "name": "' + self.model + '"\n}'
        r = requests.post('http://localhost:11430/api/pull', headers=headers, data=data)



# ollama = Ollama("11430", "llama2")
ollama = Ollama_("11434", "llama2")


@app.route('/', methods=['GET'])
@cross_origin()
def test():
    return '<h1> App Running <h1>'


@app.route('/download', methods=['GET', 'POST'])
@cross_origin()
def download_model():
    msg_queue = SimpleQueue()
    daemon = Thread(target=ollama.download_model(), args=(msg_queue,), daemon=True, name='Background')
    daemon.start()

    #ollama.download_model()
    return jsonify({"data": 'Model Download'})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
