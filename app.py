from flask import Flask, request, jsonify
from flask_cors import CORS
import urllib.request
from langchain.docstore.document import Document
import time
import boto3
import os
from botocore.exceptions import NoCredentialsError
# Embedding of a document
from langchain.embeddings import CohereEmbeddings
from langchain.document_loaders import UnstructuredFileLoader, PyPDFLoader
from langchain.vectorstores import Qdrant
from werkzeug.utils import secure_filename

# Loading environment variables
openai_api_key = os.environ.get('openai_api_key')
cohere_api_key = os.environ.get('cohere_api_key')
qdrant_url = os.environ.get('qdrant_url')
qdrant_api_key = os.environ.get('qdrant_api_key')
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')


#Flask config
app = Flask(__name__)
CORS(app)

# Test default route
@app.route('/')
def hello_world():
    return {"Hello":"World"}


@app.route('/ocr', methods=['POST'])
def ocr_route():
    data = request.form.to_dict()
    collection_name = data['collection_name']
    bucket_name = data['bucket_name']
    file_path = data['file_path']

    # Get the file from the request
    file = request.files['file']
    local_file_path = secure_filename(file.filename)
    file.save(local_file_path)

    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

    s3 = boto3.client('s3', 
                      aws_access_key_id=AWS_ACCESS_KEY_ID, 
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY, 
                      region_name='us-east-1')

    # Upload file to S3
    try:
        s3.upload_file(local_file_path, bucket_name, file_path)
        print(f'Successfully uploaded {file_path} to {bucket_name}')
        os.remove(local_file_path) # remove local file after upload
    except FileNotFoundError:
        return jsonify({'error': 'The file was not found'}), 400
    except NoCredentialsError:
        return jsonify({'error': 'Credentials not available'}), 400

    textract = boto3.client('textract', 
                            aws_access_key_id=AWS_ACCESS_KEY_ID, 
                            aws_secret_access_key=AWS_SECRET_ACCESS_KEY, 
                            region_name='us-east-1')

    response = textract.start_document_text_detection(
        DocumentLocation={'S3Object': {'Bucket': bucket_name, 'Name': file_path}}
    )

    print('Processing...')
    
    while True:
        response_get = textract.get_document_text_detection(
            JobId=response['JobId']
        )
        
        status = response_get['JobStatus']
        print('Job status: {}'.format(status))

        if status in ['SUCCEEDED', 'FAILED']:
            print('Finished')
            break
            
        print('Waiting for the job to complete...')
        time.sleep(5)

    text = ''
    for result_page in response_get['Blocks']:
        if 'Text' in result_page:
            text += result_page['Text'] + '\n'

    # Create document
    doc = Document(page_content=text)

    # Generate embeddings
    embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
    qdrant = Qdrant.from_documents(doc, embeddings, url=qdrant_url, collection_name=collection_name, prefer_grpc=True, api_key=qdrant_api_key)
    # os.remove(file_path) # Delete downloaded file
    return {"collection_name":qdrant.collection_name}



# Embedding of a document
from langchain.embeddings import CohereEmbeddings
from langchain.document_loaders import UnstructuredFileLoader, PyPDFLoader
from langchain.vectorstores import Qdrant

@app.route('/embed', methods=['POST'])
def embed_pdf():
    collection_name = request.json.get("collection_name")
    file_url = request.json.get("file_url")

    # Download the file from the url provided
    folder_path = f'./'
    os.makedirs(folder_path, exist_ok=True) # Create the folder if it doesn't exist
    filename = file_url.split('/')[-1] # Filename for the downloaded file
    file_path = os.path.join(folder_path, filename) # Full path to the downloaded file
    
    import ssl # not the best for production use to not verify ssl, but fine for testing
    ssl._create_default_https_context = ssl._create_unverified_context

    urllib.request.urlretrieve(file_url, file_path) # Download the file and save it to the local folder

    # Checking filetype for document parsing, PyPDF is a lot faster than Unstructured for pdfs.
    import mimetypes
    mime_type = mimetypes.guess_type(file_path)[0]
    if mime_type == 'application/pdf':
        loader = PyPDFLoader(file_path)
        docs = loader.load_and_split()
    else:
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load()
    # Generate embeddings
    embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
    qdrant = Qdrant.from_documents(docs, embeddings, url=qdrant_url, collection_name=collection_name, prefer_grpc=True, api_key=qdrant_api_key)
    os.remove(file_path) # Delete downloaded file
    return {"collection_name":qdrant.collection_name}

# Retrieve information from a collection
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from qdrant_client import QdrantClient

@app.route('/retrieve', methods=['POST'])
def retrieve_info():
    collection_name = request.json.get("collection_name")
    query = request.json.get("query")

    client = QdrantClient(url=qdrant_url, prefer_grpc=True, api_key=qdrant_api_key)

    embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
    qdrant = Qdrant(client=client, collection_name=collection_name, embedding_function=embeddings.embed_query)
    search_results = qdrant.similarity_search(query, k=2)
    chain = load_qa_chain(OpenAI(openai_api_key=openai_api_key,temperature=0.2), chain_type="stuff")
    results = chain({"input_documents": search_results, "question": query}, return_only_outputs=True)
    
    return {"results":results["output_text"]}