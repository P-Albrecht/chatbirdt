__import__('pysqlite3')
import sys
import os
import requests

from flask import Flask, jsonify
from flask_cors import CORS
from flask_cors import cross_origin
from flask import request
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
from threading import Thread
from queue import SimpleQueue


sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
app = Flask(__name__)
CORS(app)

lock = False
ollamaIP = '172.21.0.5'




@app.route('/', methods=['GET'])
@cross_origin()
def test_page():
    return '<h1> App Running !!!!! <h1>'


@app.route('/test', methods=['GET'])
@cross_origin()
def test():
    return jsonify({"data": 'connection working'})

@app.route('/download', methods=['GET', 'POST'])
@cross_origin()
def download_model():
    #msg_queue = SimpleQueue()
    #daemon = Thread(target=ollama.download_model(), args=(msg_queue,), daemon=True, name='Background')
    #daemon.start()
    return jsonify({"data": 'Model Download'})


@app.route('/definedQuestion/<input_>', methods=['GET', 'POST'])
@cross_origin()
def defined_question(input_):
    return jsonify({"data": "data"})


@app.route('/question/<input_>', methods=['GET', 'POST'])
@cross_origin()
def question(input_):
    return jsonify({"data": "data"})


@app.route('/questionDocument/<input_>', methods=['GET', 'POST'])
@cross_origin()
def question_document(input_):
    return jsonify({"data": "data"})


@app.route('/chatDocument/<input_>', methods=['GET', 'POST'])
@cross_origin()
def chat_document(input_):
    texts = request.args.getlist('texts[]')
    from_ai = request.args.getlist('fromAI[]')
    return jsonify({"data": "data"})
    #return jsonify({"data": requests.get('http://172.21.0.5:11434/api/tags').json()})


@app.route('/chatDocument2', methods=['GET', 'POST'])
@cross_origin()
def chat_document2():

    llm = Ollama(base_url='http://localhost:11434',
                model="llama2",
                verbose=True,
                temperature=0.3,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

    prompt = PromptTemplate(
        input_variables=["topic"],
        template="{topic}",
    )
    chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
    return chain.run('name three car brands')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
