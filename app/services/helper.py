import json
import os
import mimetypes
import openai
from openai import OpenAI
import uuid
from pinecone import Pinecone, ServerlessSpec
from app import app
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
import boto3

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

if os.getenv("PINECONE_INDEX") not in pc.list_indexes().names():
    pc.create_index(
        name=os.getenv("PINECONE_INDEX"),
        dimension=1536,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def upload_s3(file, file_name):
    bucket_name = os.environ.get("AWS_S3_BUCKET_NAME")
    region = os.environ.get("AWS_REGION")
    access_key = os.environ.get("AWS_ACCESS_KEY")
    secret_key = os.environ.get("AWS_SECRET_KEY")

    s3_client = boto3.client(
        service_name='s3',
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    response = s3_client.upload_file(file, bucket_name, file_name)

    print(f'upload_log_to_aws response: {response}')


def extract_metadata(metadata_text, text):
    # Try-except block to handle invalid JSON input gracefully
    try:
        # Parse the JSON string into a dictionary
        metadata_dict = json.loads(metadata_text)
        print(metadata_text)
        print(metadata_dict)
    except json.JSONDecodeError:
        # Handle the error (e.g., return None or a default value)
        return None
    text_new = {'text': str(text)}
    metadata_dict.update(text_new)
    return metadata_dict


def get_answer(question, context):
    prompt = f"Here is the context: {context}, and Question: {question} "
    f"Utilizing the given context and your extensive knowledge, provide an accurate "
    f"and detailed answer that encompasses information from both sources."
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}], )
    content = response.choices[0].message.content
    return content


def upload_chunks_db(chunks):
    # Initialize Pinecone Index and Embedding Model
    index = pc.Index(os.getenv("PINECONE_INDEX"))
    embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Get embeddings for each chunk of text
    embeddings_arrays = embeddings_model.embed_documents(
        [chunk["text"] for chunk in chunks]
    )

    # Set the batch size and initialize the batch list
    batch_size = 100
    batch = []

    # Iterate over chunks and create vectors
    for idx, chunk in enumerate(chunks):
        metadata = {}
        vector = {
            "values": embeddings_arrays[idx],
            "metadata": metadata,
        }
        batch.append(vector)

        # When batch is full, or it's the last item, upsert the vectors
        if len(batch) == batch_size or idx == len(chunks) - 1:
            index.upsert(vectors=batch)

            # Empty the batch for the next round
            batch = []


def upload_documents(docs, meta_data, doc_name, is_rfi):
    try:
        if is_rfi == "true":
            index = pc.Index(os.getenv("PINECONE_INDEX_RFI"))
        else:
            index = pc.Index(os.getenv("PINECONE_INDEX"))
        for doc in docs:
            filename_with_extension = os.path.basename(doc.metadata["source"])
            docName, _ = os.path.splitext(filename_with_extension)
            text = doc.page_content

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=768)

            chunks = text_splitter.create_documents([text])
            embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

            embeddings_arrays = embeddings_model.embed_documents(
                [chunk.page_content.replace("\n", " ") for chunk in chunks]
            )

            batch_size = 100
            batch = []

            for idx in range(len(chunks)):
                chunk = chunks[idx]
                metadata = {
                    "text": chunk.page_content,
                    "type": meta_data.strip('\"'),
                    "doc_name": doc_name
                }
                vector = {
                    "id": str(uuid.uuid4()),
                    "values": embeddings_arrays[idx],
                    "metadata": metadata,
                }
                batch.append(vector)

                # When batch is full, or it's the last item, upsert the vectors
                if len(batch) == batch_size or idx == len(chunks) - 1:
                    index.upsert(vectors=batch)

                    # Empty the batch
                    batch = []
    except Exception as e:  # Catch generic exceptions for broader handling
        print(f"An error occurred while processing documents: {e}")


def upload_txt(meta_data, doc_name):
    # use the uploads_dir in your DirectoryLoader
    loader = DirectoryLoader(app.config['UPLOAD_FOLDER'], glob="*.txt")
    docs = loader.load()
    upload_documents(docs, meta_data, doc_name)


def upload_pdf(meta_data, doc_name):
    # use the uploads_dir in your DirectoryLoader
    loader = PyPDFDirectoryLoader(app.config['UPLOAD_FOLDER'])
    docs = loader.load()
    upload_documents(docs, meta_data, doc_name)


def upload_doc(meta_data, doc_name):
    # use the uploads_dir in your DirectoryLoader
    loader = DirectoryLoader(app.config['UPLOAD_FOLDER'], glob="*.doc")
    docs = loader.load()
    upload_documents(docs, meta_data, doc_name)


def upload_csv(meta_data, doc_name):
    # use the uploads_dir in your DirectoryLoader
    loader = DirectoryLoader(app.config['UPLOAD_FOLDER'], glob="*.csv")
    docs = loader.load()
    upload_documents(docs, meta_data, doc_name)


def upload_pptx(meta_data, doc_name):
    # use the uploads_dir in your DirectoryLoader
    loader = DirectoryLoader(app.config['UPLOAD_FOLDER'], glob="*.pptx")
    docs = loader.load()
    upload_documents(docs, meta_data, doc_name)


def get_top_matches(text):
    context_response = get_pinecone_similarities(text)
    return context_response


def get_pinecone_similarities(text):
    index = pc.Index(os.getenv("PINECONE_INDEX"))
    embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    embedded_query = embeddings_model.embed_query(text)

    response = index.query(
        vector=embedded_query,
        top_k=1,
        include_metadata=True
    )

    processed_matches = [
        {
            'text': match['metadata']['text'].replace('\n', ' ')
        }
        for match in response['matches']
    ]
    return processed_matches[0]["text"]


def get_filter_pinecone_similarities(text, filter_data, index):
    if index == "rfi":
        index = pc.Index(os.getenv("PINECONE_INDEX_RFI"))
    else:
        index = pc.Index(os.getenv("PINECONE_INDEX"))
    embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    embedded_query = embeddings_model.embed_query(text)

    response = index.query(
        vector=embedded_query,
        filter={
            'type': {"$in": filter_data}
        },
        top_k=1,
        include_metadata=True
    )

    processed_matches = [
        {
            'text': match['metadata']['text'].replace('\n', ' ')
        }
        for match in response['matches']
    ]
    return processed_matches[0]["text"]


def get_top8_similarities(text):
    index = pc.Index(os.getenv("PINECONE_INDEX"))
    embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    embedded_query = embeddings_model.embed_query(text)

    response = index.query(
        vector=embedded_query,
        top_k=8,
        include_metadata=True
    )

    processed_matches = [
        {
            'text': match['metadata']['text'].replace('\n', ' ')
        }
        for match in response['matches']
    ]
    return processed_matches


def get_top8_filter_similarities(text, filter_data):
    index = pc.Index(os.getenv("PINECONE_INDEX"))
    embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    embedded_query = embeddings_model.embed_query(text)

    response = index.query(
        vector=embedded_query,
        filter={
            'type': {"$in": filter_data}
        },
        top_k=8,
        include_metadata=True
    )

    processed_matches = [
        {
            'text': match['metadata']['text'].replace('\n', ' '),
            'docs': match['metadata']['doc_name']
        }
        for match in response['matches']
    ]
    return processed_matches


def query_pinecone(text):
    context_response = get_pinecone_similarities(text)
    response = get_answer(question=text, context=context_response)
    return response


def query_filter_pinecone(text, filter_data, index):
    context_response = get_filter_pinecone_similarities(text, filter_data, index)
    response = get_answer(question=text, context=context_response)
    return response


def process_file_based_on_mime(file_path, meta_data, doc_name):
    """Process files based on the file's MIME type."""
    mime_type = mimetypes.guess_type(file_path)[0]

    if mime_type == 'text/plain':
        upload_txt(meta_data, doc_name)
    elif mime_type == 'application/msword':
        upload_doc(meta_data, doc_name)
    elif mime_type == 'application/pdf':
        upload_pdf(meta_data, doc_name)
    elif mime_type == 'text/csv':
        upload_csv(meta_data, doc_name)
    elif mime_type == 'application/vnd.openxmlformats-officedocument.presentationml.presentation':
        upload_pptx(meta_data, doc_name)
    # Add more MIME type handling as needed
    else:
        # Log unsupported file type and remove it
        print(f"Unsupported file type: {mime_type}")
        os.remove(file_path)
