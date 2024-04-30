import json
import os
from app import app
from flask import jsonify, request
from app.services.helper import (get_top8_similarities, upload_chunks_db, query_pinecone, upload_txt, upload_pdf,
                                 upload_s3,
                                 upload_doc, upload_csv, get_top8_filter_similarities, query_filter_pinecone,
                                 process_file_based_on_mime)
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain_openai import OpenAI

# Create a directory for storing uploaded files within the app context
UPLOAD_CSV_FOLDER = os.path.join(app.root_path, 'csv')
os.makedirs(UPLOAD_CSV_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = './uploads'  # Make sure to change this to your desired upload folder path


# Define a route for the "Hello World" endpoint
@app.route('/')
def hello_world():
    return 'Hello, World!'


# Define a route for the "asking question" endpoint
@app.route('/api/query', methods=['POST'])
def merge_text():
    try:
        data = request.get_json()
        text = data.get('text')

        if not text or not isinstance(text, str):
            return jsonify({'error': str('text is required and must be a non-empty string')}), 400
        else:
            response = query_pinecone(text)
            return jsonify({'answer': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/query_filter', methods=['POST'])
def filter_text():
    try:
        data = request.get_json()
        text = data.get('text')
        filter_data = data.get('filter')
        index = data.get('index')

        if not text or not isinstance(text, str):
            return jsonify({'error': str('text is required and must be a non-empty string')}), 400
        else:
            response = query_filter_pinecone(text, filter_data, index)
            return jsonify({'answer': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Define a route for the "tok_k similarities from DB" endpoint
@app.route('/api/get_similarities', methods=['POST'])
def query():
    try:
        data = request.get_json()
        text = data.get('text')

        if not text or not isinstance(text, str):
            return jsonify({'error': str('text is required and must be a non-empty string')}), 400
        else:
            similarities = get_top8_similarities(text)
            if similarities is None:
                # Handle the case where no similarities were found or an error occurred
                return jsonify({'error': 'Failed to retrieve similarities'}), 500

            response = {
                'similarities': similarities
            }

            return jsonify(response), 200

    except Exception as e:
        print({'error': str(e)})
        return jsonify({'error': str(e)}), 500


@app.route('/api/get_filter_similarities', methods=['POST'])
def query_filter():
    try:
        data = request.get_json()
        text = data.get('text')
        filter_data = data.get('filter')

        if not text or not isinstance(text, str):
            return jsonify({'error': str('text is required and must be a non-empty string')}), 400
        else:
            similarities = get_top8_filter_similarities(text, filter_data)
            if similarities is None:
                # Handle the case where no similarities were found or an error occurred
                return jsonify({'error': 'Failed to retrieve similarities'}), 500

            response = {
                'similarities': similarities
            }

            return jsonify(response), 200

    except Exception as e:
        print({'error': str(e)})
        return jsonify({'error': str(e)}), 500


# Define a route for the "Uploading chunks directly" endpoint
@app.route('/api/upload_chunks', methods=['POST'])
def upload_chunks():
    try:
        data = request.get_json()
        # Validate chunks
        received_chunks = data.get('chunks')
        if not received_chunks or not isinstance(received_chunks, list):
            return jsonify({'error': str('chunks are required and must be a non-empty list')}), 422

        upload_chunks_db(received_chunks)
        return jsonify(message='chunks uploaded and processed.'), 200
    except Exception as e:
        print({'error': str(e)})
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if request.method != 'POST':
        # Method Not Allowed
        return jsonify(error='Method not allowed'), 405

    uploaded_files = request.files.getlist('files')
    metadata = request.form.get('type')
    rfi = request.form.get('rfi')
    print(rfi)
    # Assuming metadata is sent as JSON, parse it
    if not metadata:
        # No files part
        return jsonify(error='Type required!'), 400
    # Assuming metadata is sent as JSON, parse it
    if not uploaded_files:
        # No files part
        return jsonify(error='No files part in the request'), 400

    files_path = []
    try:
        # Ensure the upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        for f in uploaded_files:
            # Define the path for each file
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
            # upload file to s3
            s3_path = upload_s3(f)
            # Save the file
            f.save(upload_path)
            files_path.append({"path": upload_path, "name": s3_path})
            # Here you might want to call your processing functions e.g., process_file(upload_path)

        # Process each file according to its MIME type
        for file in files_path:
            process_file_based_on_mime(file["path"], metadata, file["name"], rfi)

        # Remove files after processing
        for file in files_path:
            os.remove(file["path"])

        return jsonify(message='Files uploaded and processed successfully.')

    except Exception as e:
        # Clean up any files that were saved before the error occurred
        for file in files_path:
            if os.path.exists(file["path"]):
                os.remove(file["path"])
        return jsonify(error=str(e)), 500


@app.route('/chat_csv', methods=['POST'])
def chat_csv():
    # Load the OpenAI API key from the environment variable
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        return jsonify({"error": "OPENAI_API_KEY is not set"}), 500
    query = request.form.get('query')
    csv_file = request.files.get('csv_file')
    print("csv_file", csv_file)
    if csv_file is None:
        return jsonify({"error": "No CSV file provided"}), 400

    # Get the original file name
    original_filename = csv_file.filename

    # Create a path for saving the uploaded file
    file_path = os.path.join(UPLOAD_CSV_FOLDER, original_filename)

    # Save the uploaded file with the original name
    csv_file.save(file_path)

    agent = create_csv_agent(
        OpenAI(temperature=0), file_path, verbose=True)

    prompt = query  # "Which product line had the lowest average price"

    if prompt is None or prompt == "":
        return jsonify({"error": "No user question provided"}), 400

    response = agent.run(prompt)

    # You can format the response as needed, e.g., convert to JSON
    response_json = {"answer": response}

    return jsonify(response_json), 200
