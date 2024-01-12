import os
import mimetypes
import openai
from openai import OpenAI
import uuid
import pinecone
from app import app
import spacy
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import SpacyTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


spacy.cli.download("en_core_web_sm")


def get_answer(question, context):
    prompt = f"Here is the context: {context}, and Question: {question} "
    f"Utilizing the given context and your extensive knowledge, provide an accurate "
    f"and detailed answer that encompasses information from both sources."
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],)
    content = response.choices[0].message.content
    return content


def upload_chunks_db(chunks):
    # Initialize Pinecone Index and Embedding Model
    index = pinecone.Index(os.getenv("PINECONE_INDEX"))
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
        metadata = {
            "text": chunk["text"]  # Include the original text in the metadata
        }
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


def upload_documents(docs):
    index = pinecone.Index(os.getenv("PINECONE_INDEX"))
    for doc in docs:
        filename_with_extension = os.path.basename(doc.metadata["source"])
        docName, _ = os.path.splitext(filename_with_extension)
        text = doc.page_content

        text_splitter = SpacyTextSplitter(chunk_size=768)

        chunks = text_splitter.create_documents([text])
        embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

        embeddings_arrays = embeddings_model.embed_documents(
            [chunk.page_content.replace("\n", " ") for chunk in chunks]
        )

        batch_size = 100
        batch = []

        for idx in range(len(chunks)):
            chunk = chunks[idx]
            vector = {
                "id": str(uuid.uuid4()),
                "values": embeddings_arrays[idx],
                "metadata": {
                    **chunk.metadata,
                    "text": chunk.page_content,
                },
            }
            batch.append(vector)

            # When batch is full, or it's the last item, upsert the vectors
            if len(batch) == batch_size or idx == len(chunks) - 1:
                index.upsert(vectors=batch)

                # Empty the batch
                batch = []


def upload_txt():
    # use the uploads_dir in your DirectoryLoader
    loader = DirectoryLoader(app.config['UPLOAD_FOLDER'], glob="*.txt")
    docs = loader.load()
    upload_documents(docs)


def upload_pdf():
    # use the uploads_dir in your DirectoryLoader
    loader = PyPDFDirectoryLoader(app.config['UPLOAD_FOLDER'])
    docs = loader.load()
    upload_documents(docs)


def upload_doc():
    # use the uploads_dir in your DirectoryLoader
    loader = DirectoryLoader(app.config['UPLOAD_FOLDER'], glob="*.doc")
    docs = loader.load()
    upload_documents(docs)


def upload_csv():
    # use the uploads_dir in your DirectoryLoader
    loader = DirectoryLoader(app.config['UPLOAD_FOLDER'], glob="*.csv")
    docs = loader.load()
    upload_documents(docs)


def upload_pptx():
    # use the uploads_dir in your DirectoryLoader
    loader = DirectoryLoader(app.config['UPLOAD_FOLDER'], glob="*.csv")
    docs = loader.load()
    upload_documents(docs)


def get_top_matches(text):
    context_response = get_pinecone_similarities(text)
    return context_response


def get_pinecone_similarities(text):
    index = pinecone.Index(os.getenv("PINECONE_INDEX"))
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


def query_pinecone(text):
    context_response = get_pinecone_similarities(text)
    response = get_answer(question=text, context=context_response)
    return response


def process_file_based_on_mime(file_path):
    """Process files based on the file's MIME type."""
    mime_type = mimetypes.guess_type(file_path)[0]

    if mime_type == 'text/plain':
        upload_txt()
    elif mime_type == 'application/msword':
        upload_doc()
    elif mime_type == 'application/pdf':
        upload_pdf()
    elif mime_type == 'text/csv':
        upload_csv()
    elif mime_type == 'application/vnd.openxmlformats-officedocument.presentationml.presentation':
        upload_pptx()
    # Add more MIME type handling as needed
    else:
        # Log unsupported file type and remove it
        print(f"Unsupported file type: {mime_type}")
        os.remove(file_path)
