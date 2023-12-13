from flask import Flask, request, jsonify, Response, session, make_response
from flask.sessions import SecureCookieSessionInterface
from flask_cors import CORS, cross_origin
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
from langchain.document_loaders import AmazonTextractPDFLoader
from urllib.parse import urlparse
from langchain.prompts.prompt import PromptTemplate
import os
import qdrant_client
import cohere
import qdrant_client
from urllib.parse import urlsplit, urlunsplit
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
APIFY_API_TOKEN = os.getenv('APIFY_API_TOKEN')

# openai_api_key = os.getenv('openai_api_key')
# cohere_api_key = os.getenv('cohere_api_key')
# qdrant_url = os.getenv('qdrant_url')
# qdrant_api_key = os.getenv('qdrant_api_key')
# AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
# AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

qdrant_client = qdrant_client.QdrantClient(
    url=os.getenv('qdrant_url'),
    prefer_grpc=True,
    api_key=os.getenv('qdrant_api_key')
)

#https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/embeddings/cohere.py
cohere = CohereEmbeddings(
                model="multilingual-22-12", cohere_api_key=os.getenv('cohere_api_key')
  )



custom_template = """You are a multilingual document assistant to help a human answer all their questions regarding this document. Given the following conversation and a follow-up question, rephrase the follow-up question to make it a stand-alone question. If you don't know the answer, respond with 'I'm sorry, I don't have that answer.'
    Chat history:{chat_history}
     Follow Up Input: {question}
     Standalone Question : """

custom_prompt = PromptTemplate.from_template(custom_template)

llm = ChatOpenAI(temperature=1)

# retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':5})

# crc = ConversationalRetrievalChain.from_llm(llm, retriever, condense_question_prompt=custom_prompt)


#Flask config
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')  # change this to a secure random string
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.secret_key = os.getenv('SECRET_KEY')  # set the SECRET_KEY environment variable before running your app
CORS(app)
# CORS(app, resources={r"/*": {"origins": "https://askmydocument.onrender.com", "supports_credentials": True}})

# Test default route h
@app.route('/')
def hello_world():
    return {"Hello":"World"}

@app.route('/upload_ocr/<collection_name>', methods=['POST'])  # recreate or update
@cross_origin(supports_credentials=True)  # Apply CORS to this specific route
def update_endpoint(collection_name):
    # Check if file is included in request
    if 'input_pdf_file' not in request.files:
        return jsonify({"error": "input_pdf_file is required"}), 400
    input_pdf_file = request.files['input_pdf_file']

    # Check if we have a file in the request
    if not input_pdf_file:
        return jsonify({"error": "input_pdf_file is required"}), 400
    print(input_pdf_file.filename)

# Prepare S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id = AWS_ACCESS_KEY_ID,
        aws_secret_access_key = AWS_SECRET_ACCESS_KEY,
        region_name = 'us-east-1',
    )

    textract_client = boto3.client('textract', 
                            aws_access_key_id=AWS_ACCESS_KEY_ID, 
                            aws_secret_access_key=AWS_SECRET_ACCESS_KEY, 
                            region_name='us-east-1')

    default_bucket_name = "myflash2" 
    key = 'input-pdf-files/{}'.format(os.path.basename(input_pdf_file.filename))
    
    try:
        # Upload the file to S3 and get the response
        response = s3_client.upload_fileobj(input_pdf_file, default_bucket_name, key)
        s3_url = f"s3://{default_bucket_name}/{key}"

        print(s3_url)
    except NoCredentialsError as e:
        print("Error uploading file to S3:", e)
        return jsonify({"error": "Error uploading file to S3", "details": str(e)}), 500

    # Then, use s3_url with AmazonTextractPDFLoader
    loader = AmazonTextractPDFLoader(s3_url, client=textract_client)
    docs = loader.load()
    print(len(docs))

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

    # print(result,chat_history)
    return {"Updated an existing collection ":collection_name}


@app.route('/upload_ocr', methods=['POST'])  # recreate or update
@cross_origin(supports_credentials=True)  # Apply CORS to this specific route
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

    # Prepare S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id = AWS_ACCESS_KEY_ID,
        aws_secret_access_key = AWS_SECRET_ACCESS_KEY,
        region_name = 'us-east-1',
    )

    textract_client = boto3.client('textract', 
                            aws_access_key_id=AWS_ACCESS_KEY_ID, 
                            aws_secret_access_key=AWS_SECRET_ACCESS_KEY, 
                            region_name='us-east-1')

    default_bucket_name = "myflash2" 
    key = 'input-pdf-files/{}'.format(os.path.basename(input_pdf_file.filename))
    
    try:
        # Upload the file to S3 and get the response
        response = s3_client.upload_fileobj(input_pdf_file, default_bucket_name, key)
        s3_url = f"s3://{default_bucket_name}/{key}"

        print(s3_url)
    except NoCredentialsError as e:
        print("Error uploading file to S3:", e)
        return jsonify({"error": "Error uploading file to S3", "details": str(e)}), 500

    # Then, use s3_url with AmazonTextractPDFLoader
    loader = AmazonTextractPDFLoader(s3_url, client=textract_client)
    docs = loader.load()
    print(len(docs))

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

            # Get the chat history for this collection from the session
    chat_history = session.get(collection_name, [])
    query = "What this document is about, then list 3 possible questions someone might ask you about the document. Then encourage the person to ask the question. "
        
    vector_store = Qdrant(
            client=qdrant_client, collection_name=collection_name,
            embeddings=cohere, vector_name=collection_name
                )
    custom_template = """Start with a polite greeting and mention that you are a multilingual document assistant to answer any questions you may have regarding the uploaded document. Be polite and respectful while maintaining a professional tone of conversation.
                 If you don’t know the answer, respond with “I’m sorry, I don’t have that answer.
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

    # Save the updated chat history back to the session
    session[collection_name] = chat_history

    # print(result,chat_history)
    # return result, chat_history
    data = {"Created a new collection ":collection_name, 'answer': answer}
    json_data = json.dumps(data, ensure_ascii=False).encode('utf8')
    # qdrant = Qdrant.from_documents(chunks, embeddings, url=qdrant_url, collection_name=collection_name, vector_name="custom_vector", prefer_grpc=True, api_key=qdrant_api_key, force_recreate=True)

    return Response(json_data, mimetype='application/json; charset=utf-8')



# Embedding of a document
from langchain.embeddings import CohereEmbeddings
from langchain.document_loaders import UnstructuredFileLoader, PyPDFLoader
from langchain.vectorstores import Qdrant
from io import BytesIO

@app.route('/upload_anydoc', methods=['POST'])
@cross_origin(supports_credentials=True)  # Apply CORS to this specific route
def upload_pdf():
    if 'input_pdf_file' not in request.files:
        return jsonify({"error": "input_pdf_file is required"}), 400
    
    input_pdf_file = request.files['input_pdf_file']

    # Check if we have a file in the request
    if not input_pdf_file:
        return jsonify({"error": "input_pdf_file is required"}), 400

    # Extract the collection name from the file name
    collection_name = os.path.splitext(os.path.basename(input_pdf_file.filename))[0].replace(" ", "_")
    
    # Prepare S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id = AWS_ACCESS_KEY_ID,
        aws_secret_access_key = AWS_SECRET_ACCESS_KEY,
        region_name = 'us-east-1',
    )

    default_bucket_name = "myflash2" 
    key = 'input-pdf-files/{}'.format(os.path.basename(input_pdf_file.filename))
    
    try:
        # Save the file temporarily and upload it to S3
        # input_pdf_file.save(input_pdf_file.filename)
        # s3_client.upload_file(input_pdf_file.filename, default_bucket_name, key)
        s3_client.upload_fileobj(input_pdf_file, default_bucket_name, key)
        
        # Get the URL of the uploaded file
        generated_url = s3_client.generate_presigned_url('get_object', Params={
            'Bucket': default_bucket_name, 
            'Key': key
        })

        # os.remove(input_pdf_file.filename)

        # Parse the URL and rebuild it without params
        url_parts = list(urlsplit(generated_url))
        url_parts[3] = ""  # Parameters are at index 3
        file_url = urlunsplit(url_parts)

        print(file_url)
        
    except NoCredentialsError as e:
        print("Error uploading file to S3:", e)
        return jsonify({"error": "Error uploading file to S3", "details": str(e)}), 500

    # Checking filetype for document parsing. PyPDF is a lot faster than Unstructured for pdfs.
    import mimetypes
    mime_type = mimetypes.guess_type(file_url)[0]
    print(mime_type)
    # Load the file from the Byte Stream
    # byte_stream.seek(0)  # Make sure to seek to the start of the file
    if mime_type == 'application/pdf':
        loader = PyPDFLoader(file_url)
        docs = loader.load()
    else:
        loader = UnstructuredFileLoader(file_url)
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

            # Get the chat history for this collection from the session
    chat_history = session.get(collection_name, [])
    query = " What this document is about, then list 3 possible questions someone might ask you about the document. Then encourage the person to ask the question."
        
    vector_store = Qdrant(
            client=qdrant_client, collection_name=collection_name,
            embeddings=cohere, vector_name=collection_name
                )
    custom_template = """Start with a polite greeting and mention that you are a multilingual document assistant to answer any questions you may have regarding the uploaded document. Be polite and respectful while maintaining a professional tone of conversation.
                 If you don't know the answer, respond with 'I'm sorry, I don't have that answer'.
             Chat history:{chat_history}
            Follow Up Input: {question}
           Standalone Question :"""
    custom_prompt = PromptTemplate.from_template(custom_template)

    llm = ChatOpenAI(temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':5})

    crc = ConversationalRetrievalChain.from_llm(llm, retriever, condense_question_prompt=custom_prompt)
    result = crc({'question': query, 'chat_history': chat_history})
    answer = result['answer']
    chat_history.append((query, result['answer']))
     # After retrieving and modifying `chat_history`

    # Save the updated chat history back to the session
    session[collection_name] = chat_history

    # print(result,chat_history)
    # return result, chat_history
    data = {"Created a new collection ":collection_name, 'answer': answer}
    json_data = json.dumps(data, ensure_ascii=False).encode('utf8')
    # qdrant = Qdrant.from_documents(chunks, embeddings, url=qdrant_url, collection_name=collection_name, vector_name="custom_vector", prefer_grpc=True, api_key=qdrant_api_key, force_recreate=True)
    # os.remove(file_path) # Delete downloaded file
    # return {"Created a new collection ":collection_name}
    return Response(json_data, mimetype='application/json; charset=utf-8')


@app.route('/update_anydoc/<collection_name>', methods=['POST'])
@cross_origin(supports_credentials=True)  # Apply CORS to this specific route
def update_pdf(collection_name):

    if 'input_pdf_file' not in request.files:
        return jsonify({"error": "input_pdf_file is required"}), 400
    input_pdf_file = request.files['input_pdf_file']

    # Check if we have a file in the request
    if not input_pdf_file:
        return jsonify({"error": "input_pdf_file is required"}), 400

# Prepare S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id = AWS_ACCESS_KEY_ID,
        aws_secret_access_key = AWS_SECRET_ACCESS_KEY,
        region_name = 'us-east-1',
    )

    default_bucket_name = "myflash2" 
    key = 'input-pdf-files/{}'.format(os.path.basename(input_pdf_file.filename))
    
    try:
        # Save the file temporarily and upload it to S3
        # input_pdf_file.save(input_pdf_file.filename)
        # s3_client.upload_file(input_pdf_file.filename, default_bucket_name, key)
        s3_client.upload_fileobj(input_pdf_file, default_bucket_name, key)
        
        # Get the URL of the uploaded file
        generated_url = s3_client.generate_presigned_url('get_object', Params={
            'Bucket': default_bucket_name, 
            'Key': key
        })

        # os.remove(input_pdf_file.filename)

        # Parse the URL and rebuild it without params
        url_parts = list(urlsplit(generated_url))
        url_parts[3] = ""  # Parameters are at index 3
        file_url = urlunsplit(url_parts)

        print(file_url)
        
    except NoCredentialsError as e:
        print("Error uploading file to S3:", e)
        return jsonify({"error": "Error uploading file to S3", "details": str(e)}), 500

    # Checking filetype for document parsing. PyPDF is a lot faster than Unstructured for pdfs.
    import mimetypes
    mime_type = mimetypes.guess_type(file_url)[0]
    print(mime_type)
    # Load the file from the Byte Stream
    # byte_stream.seek(0)  # Make sure to seek to the start of the file
    if mime_type == 'application/pdf':
        loader = PyPDFLoader(file_url, extract_images=True)
        docs = loader.load()
    else:
        loader = UnstructuredFileLoader(file_url)
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
    return {"Updated an existing collection ":collection_name}

@app.route('/list_documents', methods=['GET'])
@cross_origin(supports_credentials=True)  # Apply CORS to this specific route
def list_documents():
    collection_data = qdrant_client.get_collections()
    document_names = [collection.name for collection in collection_data.collections]
    return jsonify(document_names)


@app.route('/delete_document', methods=['DELETE'])
@cross_origin(supports_credentials=True)  # Apply CORS to this specific route
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



@app.route('/retriever', methods=['POST'])
@cross_origin(supports_credentials=True)  # Apply CORS to this specific route
def retrieve_in(chat_history=[]):
    collection_name = request.json.get("collection_name")
    # print(collection_name)
    query = request.json.get("query")

    if not collection_name or len(collection_name) > 255:
        return {"error": "Invalid collection name"}

    # Get the chat history for this collection from the session
    chat_history = session.get(collection_name, [])
    print(chat_history)

    vector_store = Qdrant(
    client=qdrant_client, collection_name=collection_name,
    embeddings=cohere, vector_name=collection_name
    )

    # vector_store = Qdrant(
    #     client=qdrant_client, collection_name=collection_name,
    #     embeddings=cohere, vector_name=collection_name
    #         )
    # custom_template = """You are a multilingual document assistant here to help a human with any questions he/she may have regarding the document uploaded having. Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. If you do not know the answer reply with 'I am sorry, I dont have this answer'.
    #     Chat History:
    #     {chat_history}
    #     Follow Up Input: {question}
    #     Standalone question:"""
    custom_prompt = PromptTemplate.from_template(custom_template)

    # llm = ChatOpenAI(temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':5})

    crc = ConversationalRetrievalChain.from_llm(llm, retriever, condense_question_prompt=custom_prompt)
    result = crc({'question': query, 'chat_history': chat_history})
    answer = result['answer']
    print(answer)
    chat_history.append((query, result['answer']))
     # After retrieving and modifying `chat_history`
    # chat_history.append((query, result['answer']))

    # Save the updated chat history back to the session
    session[collection_name] = chat_history

    # print(result,chat_history)
    # return result, chat_history
    data = {'question': query, 'answer': answer, 'chat_history': chat_history}
    # print(data)
    # json_data = json.dumps(data, ensure_ascii=False).encode('utf8')

    # return Response(json_data, mimetype='application/json; charset=utf-8')
    # print(data)
    return jsonify(data)   # Use Flask's jsonify method

@app.route('/enter_url/<collection_name>', methods=['POST'])  # recreate or update
@cross_origin(supports_credentials=True)  # Apply CORS to this specific route
def update_url(collection_name):
    # Check if file is included in request
    collection_name = collection_name
    print(collection_name)

    url = request.json.get("url")
    

    
    if not collection_name or len(collection_name) > 255:
        return {"error": "Invalid collection name"}
    # query = request.json.get("query")
    apify = ApifyWrapper()

    # Call the Actor to obtain text from the crawled webpages
    loader = apify.call_actor(
        actor_id="apify/website-content-crawler",
        run_input={
            "startUrls": [
                {
                    "url": url
                }
            ]
            },
        dataset_mapping_function=lambda item: Document(
            page_content=item["text"] or "", metadata={"source": item["url"]}
        ),
    )
    docs = loader.load()
    # docs = loader.load()
    print(len(docs))

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

    # print(result,chat_history)
    return {"Updated an existing collection ":collection_name}


@app.route('/enter_url', methods=['POST'])
@cross_origin(supports_credentials=True)  # Apply CORS to this specific route
def website_query(chat_history=[]):
    url = request.json.get("url")
    parsed_url = urlparse(url)
    # Extract the domain name as the collection name
    collection_name = parsed_url.netloc
    print(collection_name)

    if not collection_name or len(collection_name) > 255:
        return {"error": "Invalid collection name"}
    # query = request.json.get("query")
    apify = ApifyWrapper()

    # Call the Actor to obtain text from the crawled webpages
    loader = apify.call_actor(
        actor_id="apify/website-content-crawler",
        run_input={
            "startUrls": [
                {
                    "url": url
                }
            ]
            },
        dataset_mapping_function=lambda item: Document(
            page_content=item["text"] or "", metadata={"source": item["url"]}
        ),
    )
    docs = loader.load()
    # docs = loader.load()
    print(len(docs))

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

            # Get the chat history for this collection from the session
    chat_history = session.get(collection_name, [])
    query = "What this document is about, then list 3 possible questions someone might ask you about the document. Then encourage the person to ask the question. "
        
    vector_store = Qdrant(
            client=qdrant_client, collection_name=collection_name,
            embeddings=cohere, vector_name=collection_name
                )
    custom_template = """Start with a polite greeting and mention that you are a multilingual document assistant to answer any questions you may have regarding the uploaded document. Be polite and respectful while maintaining a professional tone of conversation.
                 If you don't know the answer, respond with 'I'm sorry, I don't have that answer'.
             Chat history:
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

    # Save the updated chat history back to the session
    session[collection_name] = chat_history

    # print(result,chat_history)
    # return result, chat_history
    data = {"Created a new collection ":collection_name, 'answer': answer}
    json_data = json.dumps(data, ensure_ascii=False).encode('utf8')
    # qdrant = Qdrant.from_documents(chunks, embeddings, url=qdrant_url, collection_name=collection_name, vector_name="custom_vector", prefer_grpc=True, api_key=qdrant_api_key, force_recreate=True)

    return Response(json_data, mimetype='application/json; charset=utf-8')

@app.route('/summarize', methods=['POST'])
@cross_origin(supports_credentials=True)  # Apply CORS to this specific route
def retrieve_summary(chat_history=[]):
    collection_name = request.json.get("collection_name")
    print(collection_name)
    # query = request.json.get("query")
    query = "Summarize the entire document in a detailed manner. List out all the key points"

    if not collection_name or len(collection_name) > 255:
        return {"error": "Invalid collection name"}
    
    # Get the chat history for this collection from the session
    # chat_history = session.get(collection_name, [])

    vector_store = Qdrant(
        client=qdrant_client, collection_name=collection_name,
        embeddings=cohere, vector_name=collection_name
            )
    custom_template = """Here you are a multilingual document assistant to help a human answer all their questions regarding the downloaded document. Given the following conversation and a follow-up question, rephrase the follow-up question to make it a stand-alone question. If you don't know the answer, respond with "I'm sorry, I don't have that answer."'.
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""
    custom_prompt = PromptTemplate.from_template(custom_template)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':5})

    crc = ConversationalRetrievalChain.from_llm(llm, retriever, condense_question_prompt=custom_prompt)
    result = crc({'question': query, 'chat_history': chat_history})
    answer = result['answer']

    # return result, chat_history
    data = {'answer': answer}

    return jsonify(data)


@app.route('/chathistory/<collection_name>', methods=['GET'])
@cross_origin(supports_credentials=True)  # Apply CORS to this specific route
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
@cross_origin(supports_credentials=True)  # Apply CORS to this specific route
def clear_collection_history(collection_name):
    # Remove the chat history for this collection from the session
    session.pop(collection_name, None)

    # Return a success message 
    return jsonify({'message': f'Chat history for collection {collection_name} cleared'})
