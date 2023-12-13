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

class Ollama_:
    def __init__(self, port, model):
        self.port = port
        self.model = model
        self.llm = None
        self.qa_chain = None
        self.vectorstore = None

    def set_llm(self):

        r = requests.get('http://' + ollamaIP + ':' + self.port + '/api/tags').json()
        has_llama = False
        for x in r['models']:
            print(x['name'])
            if x['name'] == self.model + ':latest':
                has_llama = True

        print(has_llama)

        if has_llama:
            loader = DirectoryLoader('./Folder', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
            data = loader.load()

            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            all_splits = text_splitter.split_documents(data)
            print(f"Split into {len(all_splits)} chunks")

            # create the open-source embedding function
            embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

            # load it into Chroma
            self.vectorstore = Chroma.from_documents(all_splits, embedding_function)

            print(f"Loaded {len(data)} documents")

            # LLM
            self.llm = Ollama(base_url='http://' + ollamaIP + ':' + self.port,
                              model="llama2",
                              verbose=True,
                              temperature=0.3,
                              callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

        else:
            print('model not found !!!!!!!!!!!!!!!!!!!!!!!!!!! -->> Download')
            msg_queue = SimpleQueue()
            daemon = Thread(target=self.download_model(), args=(msg_queue,), daemon=True, name='Background')
            daemon.start()

    def download_model(self):
        lock = True
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
        }
        data = '{\n  "name": "' + self.model + '"\n}'
        r = requests.post('http://' + ollamaIP + ':' + self.port + '/api/pull', headers=headers, data=data)
        lock = False

    def defined_question(self, input):
        if lock:
            return 'downloading'
        elif self.llm is None:
            self.set_llm()
            return 'model not found'
        else:
            prompt = PromptTemplate(
                input_variables=["topic"],
                template="Give me 2 interesting facts about {topic}?",
            )
            chain = LLMChain(llm=self.llm, prompt=prompt, verbose=False)
            return chain.run(input)

    def question(self, input):
        if lock:
            return 'downloading'
        elif self.llm is None:
            self.set_llm()
            return 'model not found'
        else:
            prompt = PromptTemplate(
                input_variables=["topic"],
                template="{topic}",
            )
            chain = LLMChain(llm=self.llm, prompt=prompt, verbose=False)
            return chain.run(input)

    def question_document(self, input):
        if lock:
            return 'downloading'
        elif self.llm is None:
            self.set_llm()
            return 'model not found'
        else:
            qa_chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff",
                                                   retriever=self.vectorstore.as_retriever())
            result = qa_chain({"query": input})
            return result['result']

    def chat_document(self, input, texts, from_ai):
        if lock:
            return 'downloading'
        elif self.llm is None:
            self.set_llm()
            return 'model not found'
        else:
            qa_chain = ConversationalRetrievalChain.from_llm(llm=self.llm, chain_type="stuff",  retriever=self.vectorstore.as_retriever())

            print(texts)
            print(from_ai)

            chat_history = []
            h_input = ''
            for i, x in enumerate(texts):
                # print(from_ai[i])
                if from_ai[i] == 'response':
                    chat_history.append((h_input, x))
                    h_input = ''

                else:
                    if (h_input != ''):
                        chat_history.append((h_input, ''))
                    # elif i == (len(texts)-1):
                    #    chat_history.append((x, ''))
                    h_input = x

            print(chat_history)
            result = qa_chain({"question": input, "chat_history": chat_history})

            return result['answer']


ollama = Ollama_("11434", "llama2")


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
    msg_queue = SimpleQueue()
    daemon = Thread(target=ollama.download_model(), args=(msg_queue,), daemon=True, name='Background')
    daemon.start()
    return jsonify({"data": 'Model Download'})


@app.route('/definedQuestion/<input_>', methods=['GET', 'POST'])
@cross_origin()
def defined_question(input_):
    return jsonify({"data": ollama.defined_question(input_)})


@app.route('/question/<input_>', methods=['GET', 'POST'])
@cross_origin()
def question(input_):
    return jsonify({"data": ollama.question(input_)})


@app.route('/questionDocument/<input_>', methods=['GET', 'POST'])
@cross_origin()
def question_document(input_):
    return jsonify({"data": ollama.question_document(input_)})


@app.route('/chatDocument/<input_>', methods=['GET', 'POST'])
@cross_origin()
def chat_document(input_):
    texts = request.args.getlist('texts[]')
    from_ai = request.args.getlist('fromAI[]')
    return jsonify({"data": ollama.chat_document(input_, texts, from_ai)})
    #return jsonify({"data": requests.get('http://172.21.0.5:11434/api/tags').json()})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
