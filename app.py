from flask import Flask, request, jsonify, Response, session, make_response
from flask_cors import CORS
import time
import boto3
import json
import urllib
import cohere
from botocore.exceptions import NoCredentialsError
#import pypdf
# Embedding of a document
from langchain.embeddings import CohereEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader, PyPDFLoader
from langchain.vectorstores import Qdrant
from werkzeug.utils import secure_filename
from langchain.document_loaders import TextLoader
from langchain.document_loaders import TextLoader
from qdrant_client import QdrantClient
from langchain.embeddings import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
import cohere
from qdrant_client.http.models import Batch
from qdrant_client.http import models
import qdrant_client
import os
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.utilities import ApifyWrapper
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain.embeddings import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from datetime import datetime
import os
import qdrant_client
import cohere
import qdrant_client

from qdrant_client.http.models import Batch
from qdrant_client.http import models
import os

from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env file

# Loading environment variables
openai_api_key = os.getenv('openai_api_key')
cohere_api_key = os.getenv('cohere_api_key')
qdrant_url = os.getenv('qdrant_url')
qdrant_api_key = os.getenv('qdrant_api_key')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

qdrant_client = qdrant_client.QdrantClient(
    url=os.getenv('qdrant_url'),
    prefer_grpc=True,
    api_key=os.getenv('qdrant_api_key')
)



#https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/embeddings/cohere.py
cohere = CohereEmbeddings(
                model="multilingual-22-12", cohere_api_key=os.getenv('cohere_api_key')
  )

def process_ocr(input_pdf_file, collection_name, action='create'):
    # AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    # AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

    s3_client = boto3.client('s3', 
                      aws_access_key_id=AWS_ACCESS_KEY_ID, 
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY, 
                      region_name='us-east-1')
    textract_client = boto3.client('textract', 
                            aws_access_key_id=AWS_ACCESS_KEY_ID, 
                            aws_secret_access_key=AWS_SECRET_ACCESS_KEY, 
                            region_name='us-east-1')
    history = []
    default_bucket_name = "myflash" 
    output_file = "output.txt"
   
    key = 'input-pdf-files/{}'.format(os.path.basename(input_pdf_file.filename))

    try:
        input_pdf_file.save(input_pdf_file.filename)
        s3_client.upload_file(input_pdf_file.filename, default_bucket_name, key)
        os.remove(input_pdf_file.filename)
    except NoCredentialsError as e:
        print("Error uploading file to S3:", e)
        return history, "Error uploading file to S3: {}".format(e), "None"
        
    s3_object = {'Bucket': default_bucket_name, 'Name': key}
    
    response = textract_client.start_document_analysis(
        DocumentLocation={'S3Object': s3_object},
        FeatureTypes=['TABLES', 'FORMS']
    )
    job_id = response['JobId']
    
    while True:
        response = textract_client.get_document_analysis(JobId=job_id)
        status = response['JobStatus']
        if status in ['SUCCEEDED', 'FAILED']:
            break
        time.sleep(5)

    if status == 'SUCCEEDED':
        with open(output_file, 'w') as output_file_io:
            for block in response['Blocks']:
                if block['BlockType'] in ['LINE', 'WORD']:
                    output_file_io.write(block['Text'] + '\n')

        with open(output_file, "r") as file:
            first_512_chars = file.read(512).replace("\n", "").replace("\r", "").replace("[", "").replace("]", "") + " [...]"
            history.append(("Document conversion", first_512_chars))

        loader = TextLoader(output_file)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 500,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
            )

        chunks = text_splitter.split_documents(docs)

            # Generate embeddings
        # embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
        # Recreate the collection with the new vector store
        vector_store = Qdrant(
        client=qdrant_client, collection_name=collection_name,
        embeddings=cohere, vector_name=collection_name
            )
        vectors_config = {
            collection_name: models.VectorParams(size=768, #collection_name as custom vector name
        distance=models.Distance.COSINE),
        }
        if action == 'create':
            qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
            )
        elif action == 'update':
            qdrant_client.update_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
            )
        else:
            print(f"Invalid action: {action}")
            return
        print('Loading a new vector store...')

        vector_store.add_documents(chunks)
        print('Upserting finished.')

        embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
        # qdrant = Qdrant.from_documents(chunks, embeddings, url=qdrant_url, collection_name=collection_name, prefer_grpc=True, api_key=qdrant_api_key, force_recreate=False)
        os.remove(output_file) # Delete downloaded file
        # collection_name = qdrant.collection_name
        return history, first_512_chars, collection_name

    return "Failed to convert document", "None"


#Flask config
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')  # set the SECRET_KEY environment variable before running your app
CORS(app)

# Test default route h
@app.route('/')
def hello_world():
    return {"Hello":"World"}

@app.route('/upload_ocr/<collection_name>', methods=['POST'])  # recreate or update
def update_endpoint(collection_name):
    # Check if file is included in request
    if 'input_pdf_file' not in request.files:
        return jsonify({"error": "input_pdf_file is required"}), 400
    input_pdf_file = request.files['input_pdf_file']

    # Check if we have a file in the request
    if not input_pdf_file:
        return jsonify({"error": "input_pdf_file is required"}), 400
    print(input_pdf_file.filename)

    history, texts, collection_name = process_ocr(input_pdf_file, collection_name, 'update')

    if collection_name == 'None':
        return jsonify({"error": texts}), 400

    return jsonify({"Updated an existing collection": collection_name, "texts": texts}), 200


@app.route('/upload_ocr', methods=['POST'])  # recreate or update
def ocr_endpoint():
    # Check if file is included in request
    if 'input_pdf_file' not in request.files:
        return jsonify({"error": "input_pdf_file is required"}), 400
    input_pdf_file = request.files['input_pdf_file']

    # Check if we have a file in the request
    if not input_pdf_file:
        return jsonify({"error": "input_pdf_file is required"}), 400
    print(input_pdf_file.filename)
    # Extract the collection name from the file name
    collection_name = os.path.splitext(os.path.basename(input_pdf_file.filename))[0].replace(" ", "_")

    # history = request.form.get('history') 
    # history = []

    # call the function and get the collection name
    history, texts, collection_name = process_ocr(input_pdf_file, collection_name)

    if collection_name == 'None':
        return jsonify({"error": texts}), 400

    return jsonify({"Created a new document": collection_name, "texts": texts}), 200



# Embedding of a document
from langchain.embeddings import CohereEmbeddings
from langchain.document_loaders import UnstructuredFileLoader, PyPDFLoader
from langchain.vectorstores import Qdrant
from io import BytesIO

@app.route('/upload_anydoc', methods=['POST'])
def upload_pdf():

    if 'input_pdf_file' not in request.files:
        return jsonify({"error": "input_pdf_file is required"}), 400
    input_pdf_file = request.files['input_pdf_file']

    # Check if we have a file in the request
    if not input_pdf_file:
        return jsonify({"error": "input_pdf_file is required"}), 400

        # Extract the collection name from the file name
    collection_name = os.path.splitext(os.path.basename(input_pdf_file.filename))[0].replace(" ", "_")
    print(collection_name)
    # write the uploaded file to a local file

    if not collection_name:
        #  Assign a default name
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        collection_name = "document" + current_time
    print(collection_name)
    file_path = os.path.join('./', input_pdf_file.filename)

    # write the uploaded file to a local file
    file_path = os.path.join('./', input_pdf_file.filename)
    input_pdf_file.save(file_path)
    print(file_path)

    # Checking filetype for document parsing, PyPDF is a lot faster than Unstructured for pdfs.
    import mimetypes
    mime_type = mimetypes.guess_type(input_pdf_file.filename)[0]
    print(mime_type)
    # now you have a local file to process using the loaders
    if mime_type == 'application/pdf':
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    else:
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 500,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )

    chunks = text_splitter.split_documents(docs)
    # Generate embeddings
    embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
    # Recreate the collection with the new vector store
    vector_store = Qdrant(
        client=qdrant_client, collection_name=collection_name,
        embeddings=cohere, vector_name=collection_name
            )
    vectors_config = {
          collection_name: models.VectorParams(size=768, #collection_name as custom vector name
      distance=models.Distance.COSINE),
      }
    
    qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
    )

    
    print('Loading a new vector store...')

    # texts = [chunk.page_content for chunk in chunks]

    vector_store.add_documents(chunks)
    print('Upserting finished.')
    # qdrant = Qdrant.from_documents(chunks, embeddings, url=qdrant_url, collection_name=collection_name, vector_name="custom_vector", prefer_grpc=True, api_key=qdrant_api_key, force_recreate=True)
    os.remove(file_path) # Delete downloaded file
    return {"Created a new collection ":collection_name}


@app.route('/update_anydoc/<collection_name>', methods=['POST'])
def update_pdf(collection_name):

    if 'input_pdf_file' not in request.files:
        return jsonify({"error": "input_pdf_file is required"}), 400
    input_pdf_file = request.files['input_pdf_file']

    # Check if we have a file in the request
    if not input_pdf_file:
        return jsonify({"error": "input_pdf_file is required"}), 400

        # Extract the collection name from the file name
    # collection_name = os.path.splitext(os.path.basename(input_pdf_file.filename))[0].replace(" ", "_")
    print(collection_name)
    # write the uploaded file to a local file

    if not collection_name:
        #  Assign a default name
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        collection_name = "document" + current_time
    print(collection_name)
    file_path = os.path.join('./', input_pdf_file.filename)

    # write the uploaded file to a local file
    file_path = os.path.join('./', input_pdf_file.filename)
    input_pdf_file.save(file_path)
    print(file_path)

    # Checking filetype for document parsing, PyPDF is a lot faster than Unstructured for pdfs.
    import mimetypes
    mime_type = mimetypes.guess_type(input_pdf_file.filename)[0]
    print(mime_type)
    # now you have a local file to process using the loaders
    if mime_type == 'application/pdf':
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    else:
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 500,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )

    chunks = text_splitter.split_documents(docs)
    # Generate embeddings
    embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
    # Recreate the collection with the new vector store
    vector_store = Qdrant(
        client=qdrant_client, collection_name=collection_name,
        embeddings=cohere, vector_name=collection_name
            )
    vectors_config = {
          collection_name: models.VectorParams(size=768, #collection_name as custom vector name
      distance=models.Distance.COSINE),
      }
    
    qdrant_client.update_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
    )

    print('Loading a new vector store...')

    # texts = [chunk.page_content for chunk in chunks]

    vector_store.add_documents(chunks)
    print('Upserting finished.')
    # qdrant = Qdrant.from_documents(chunks, embeddings, url=qdrant_url, collection_name=collection_name, vector_name="custom_vector", prefer_grpc=True, api_key=qdrant_api_key, force_recreate=True)
    os.remove(file_path) # Delete downloaded file
    return {"Updated an existing collection ":collection_name}

@app.route('/list_documents', methods=['POST'])
def list_documents():
    collection_data = qdrant_client.get_collections()
    document_names = [collection.name for collection in collection_data.collections]
    return jsonify(document_names)


@app.route('/delete_document', methods=['DELETE'])
def delete_collection():
    collection_name = request.json.get("collection_name")

    qdrant_client.delete_collection(collection_name=collection_name)

    return {"This following document has been deleted": collection_name}



# Retrieve information from a collection
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from qdrant_client import QdrantClient
from langchain.embeddings import CohereEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate

openai_api_key = os.getenv('openai_api_key')
cohere_api_key = os.getenv('cohere_api_key')
qdrant_url = os.getenv('qdrant_url')
qdrant_api_key = os.getenv('qdrant_api_key')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

@app.route('/retriever', methods=['POST'])
def retrieve_info(chat_history=[]):
    collection_name = request.json.get("collection_name")
    print(collection_name)
    query = request.json.get("query")

    if not collection_name or len(collection_name) > 255:
        return {"error": "Invalid collection name"}
    
    # Get the chat history for this collection from the session
    chat_history = session.get(collection_name, [])

    vector_store = Qdrant(
        client=qdrant_client, collection_name=collection_name,
        embeddings=cohere, vector_name=collection_name
            )
    custom_template = """You are a nice assistant having a conversation with a human.Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. If you do not know the answer reply with 'I am sorry, I dont have this answer'.
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""
    custom_prompt = PromptTemplate.from_template(custom_template)

    llm = ChatOpenAI(temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':5})

    crc = ConversationalRetrievalChain.from_llm(llm, retriever, condense_question_prompt=custom_prompt)
    result = crc({'question': query, 'chat_history': chat_history})
    answer = result['answer']
    chat_history.append((query, result['answer']))
     # After retrieving and modifying `chat_history`
    # chat_history.append((query, result['answer']))

    # Save the updated chat history back to the session
    session[collection_name] = chat_history

    # print(result,chat_history)
    # return result, chat_history
    data = {'question': query, 'answer': answer, 'chat_history': chat_history}
    json_data = json.dumps(data, ensure_ascii=False).encode('utf8')

    return Response(json_data, mimetype='application/json; charset=utf-8')

@app.route('/chathistory/<collection_name>', methods=['GET'])
def get_collection_history(collection_name):
    # Get the chat history for this collection from the session
    chat_history = session.get(collection_name, [])

    # Convert to JSON serializable format
    chat_history_serializable = [{'query': ch[0], 'response': ch[1]} for ch in chat_history]

    # Return the chat history as JSON
    return jsonify({
        'chat_history': chat_history_serializable
    })

@app.route('/clearhistory/<collection_name>', methods=['POST'])
def clear_collection_history(collection_name):
    # Remove the chat history for this collection from the session
    session.pop(collection_name, None)

    # Return a success message 
    return jsonify({'message': f'Chat history for collection {collection_name} cleared'})