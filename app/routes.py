import json
import io
from prisma.models import Assistants
from .database import db
import requests
import os
from app import app
from flask import jsonify, request
from app.services.helper import (layer)
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain_openai import OpenAI

# Define a route for the "Hello World" endpoint
@app.route('/')
def hello_world():
    return 'Hello, World!'


# Define a route for the "asking question" endpoint
@app.route('/api/query', methods=['POST'])
def query_text():
    try:
        data = request.get_json()
        text = data.get('text')

        if not text or not isinstance(text, str):
            return jsonify({'error': str('text is required and must be a non-empty string')}), 400
        else:
            response = layer(text)
            return jsonify({'answer': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500