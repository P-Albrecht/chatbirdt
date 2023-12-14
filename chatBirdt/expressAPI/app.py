from flask import Flask, jsonify, request
from flask_cors import cross_origin
from flask_cors import CORS
import requests

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import sys
import os
app = Flask(__name__)
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
#CORS(app, origins=['http://localhost:4200/'])

@app.route('/')
@cross_origin()
def index():
    return jsonify({"data": 'Web App with Python Flask???'})


@app.route('/xxx', methods=['GET', 'POST'])
@cross_origin()
def chat_document():
    return requests.get('http://localhost:11434/api/tags').content


@app.route('/yyy', methods=['GET', 'POST'])
@cross_origin()
def yyy():
    return requests.get('http://localhost:5000/').content

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)