from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import boto3
import os
from botocore.exceptions import NoCredentialsError
# Embedding of a document
from langchain.embeddings import CohereEmbeddings
from langchain.document_loaders import UnstructuredFileLoader, PyPDFLoader
from langchain.vectorstores import Qdrant
from werkzeug.utils import secure_filename
from langchain.document_loaders import TextLoader

# Loading environment variables
openai_api_key = os.environ.get('openai_api_key')
cohere_api_key = os.environ.get('cohere_api_key')
qdrant_url = os.environ.get('qdrant_url')
qdrant_api_key = os.environ.get('qdrant_api_key')
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')


def pdf_2_text(input_pdf_file, history, collection_name):
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

    s3_client = boto3.client('s3', 
                      aws_access_key_id=AWS_ACCESS_KEY_ID, 
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY, 
                      region_name='us-east-1')
    textract_client = boto3.client('textract', 
                            aws_access_key_id=AWS_ACCESS_KEY_ID, 
                            aws_secret_access_key=AWS_SECRET_ACCESS_KEY, 
                            region_name='us-east-1')
    history = history or []
    default_bucket_name = "myflash" 
    output_file = "output.txt"
   
    key = 'input-pdf-files/{}'.format(os.path.basename(input_pdf_file))
    
    try:
        response = s3_client.upload_file(input_pdf_file, default_bucket_name, key)
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

        embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
        qdrant = Qdrant.from_documents(docs, embeddings, url=qdrant_url, collection_name=collection_name, prefer_grpc=True, api_key=qdrant_api_key)
        os.remove(output_file) # Delete downloaded file
        collection_name = qdrant.collection_name
        return history, first_512_chars, collection_name

    return history, "Failed to convert document", "None"


#Flask config
app = Flask(__name__)
CORS(app)

# Test default route
@app.route('/')
def hello_world():
    return {"Hello":"World"}


@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    # data = request.get_json()
    # input_pdf_file = data.get('input_pdf_file', None)
    # history = data.get('history', None)
    # collection_name = data.get('collection_name', None)
    if 'input_pdf_file' not in request.files:
        return jsonify({"error": "input_pdf_file is required"}), 400
    input_pdf_file = request.files['input_pdf_file']
    history = request.form.get('history')
    collection_name = request.form.get('collection_name')
    
    if not input_pdf_file:
        return jsonify({"error": "input_pdf_file is required"}), 400

    history, texts, collection_name = pdf_2_text(input_pdf_file, history, collection_name)
    
    if collection_name == 'None':
        return jsonify({"error": texts}), 400

    return jsonify({"collection_name": collection_name, "history": history, "texts": texts}), 200



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