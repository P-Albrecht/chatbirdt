import requests
from flask import Flask, jsonify, request
from flask_cors import cross_origin

from flask_cors import cross_origin

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

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

app = Flask(__name__)





ollamaIP = 'http://localhost:11434'
#ollamaIP = 'http://host.docker.internal:11434'

class Ollama_:
    def __init__(self):
        self.llm = None
        self.qa_chain = None
        self.vectorstore = None

    def set_llm(self):

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
        self.llm = Ollama(base_url= ollamaIP,
                          model="llama2",
                          verbose=True,
                          temperature=0.3,
                          callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

    def download_model(self):
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
        }
        data = '{\n  "name": "' + self.model + '"\n}'
        r = request.post(ollamaIP + '/api/pull', headers=headers, data=data)

    def defined_question(self, input):
        if self.llm is None:
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
        if self.llm is None:
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
        if self.llm is None:
            self.set_llm()
            return 'model not found'
        else:
            qa_chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff",
                                                   retriever=self.vectorstore.as_retriever())
            result = qa_chain({"query": input})
            return result['result']

    def chat_document(self, input, texts, from_ai):
        if self.llm is None:
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

ollama = Ollama_()
ollama.set_llm()








@app.route('/')
def index():
    return 'Web App with Python Flask!'


@app.route('/download', methods=['GET', 'POST'])
@cross_origin()
def download_model():
    ollama.download_model()
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



app.run(host='0.0.0.0', port=5000)
