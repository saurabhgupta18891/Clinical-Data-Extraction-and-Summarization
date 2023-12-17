from flask import Flask, request, jsonify
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os
from langchain.chains import RetrievalQA
import pinecone
import openai
from werkzeug.utils import secure_filename
from transformers import  TFAutoModelForSeq2SeqLM
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline

app = Flask(__name__)

# Set your Pinecone and OpenAI API keys here
os.environ["PINECONE_API_KEY"] = ""
os.environ["PINECONE_ENV"] = "gcp-starter"
os.environ["OPENAI_API_KEY"] = ""

# Specify the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'docx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the 'uploads' directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Load the language model, embeddings, and set up Pinecone
def setup():
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    index_name = "frontera-poc"

    # initialize pinecone
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.getenv("PINECONE_ENV"),  # next to api key in console
    )

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, metric="cosine", dimension=1536)

    # pinecone.delete_index("frontera-poc")

    return embeddings, index_name

# Function for Named Entity Recognition using OpenAI GPT-3.5 Turbo
def ner_extraction(document_content):
    SYSTEM_PROMPT = ("You are a smart and intelligent Medical Named Entity Recognition (NER) system."
                     "I will provide you the medical documents, and you have to extract all medical entities present in the document."
                     "Only extract medical entities; if no medical entity is present, please respond that there are no medical entities present in the document."
                     "Please only provide a list of medical entities; no additional text or notes. Entities should be extracted in the following format::"
                     "Category of entity: Name of entity, e.g., 1. Medicine: Name of medicine, 2. Illness: Name of illness, 3.Procedure: Name of procedure, 4.Symptom: Name of Symptom")

    USER_PROMPT = f" Question: Please extract medical entities in the given document - {document_content}, Answer:"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ]
    )

    return response['choices'][0]['message']['content'].strip()

# Function for document summarization
def summarize_document(document_content):
    model_name = 'Saurabh91/medical_summarization-finetuned-starmpccAsclepius-Synthetic-Clinical-Notes'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("summarization", model=model, tokenizer=tokenizer, max_new_tokens=128)
    hf = HuggingFacePipeline(pipeline=pipe)
    summary = hf(document_content)

    return summary


# Load the language model and set up the QA chain
def load_qa_chain(texts,embeddings,index_name):
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
    prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    retriever = Pinecone.from_documents(texts, embeddings, index_name=index_name).as_retriever(search_kwargs={'k': 2})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever,
                                           return_source_documents=True, chain_type_kwargs={'prompt': prompt})

    return qa_chain


# Endpoint for processing queries
@app.route('/answer', methods=['GET','POST'])
def get_answer():
    try:
        if 'file' not in request.files or 'query' not in request.form:
            return jsonify({'error': 'Please provide both file and query'})

        file = request.files['file']
        query = request.form['query']

        # Check if the file is allowed
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Load and split the document
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            document_content = docs[0].page_content
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len,
                                                           add_start_index=True)
            texts = text_splitter.split_documents(docs)

            # Set up embeddings and Pinecone
            embeddings, index_name = setup()

            # Ingest documents into Pinecone
            db = Pinecone.from_documents(texts, embeddings, index_name=index_name)

            # Load the QA chain
            qa_chain = load_qa_chain(texts,embeddings,index_name)

            # Get the answer
            response = qa_chain({'query': query})
            answer_qa = response["result"]

            # Perform Named Entity Recognition using OpenAI GPT-3.5 Turbo
            ner_result = ner_extraction(document_content)
            ner_result = "Medical Entities Extracted:\n" + ner_result

            summary_result = summarize_document(document_content)
            # return jsonify({'answer': answer})
            return jsonify({'answer_qa': answer_qa, 'ner_result': ner_result, 'summary_result': summary_result})
            # return jsonify({'summary_result': summary_result})

        else:
            return jsonify({'error': 'Invalid file extension'})

    except Exception as e:
            return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
