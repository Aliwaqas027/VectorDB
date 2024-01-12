from flask import Flask

# Create a Flask app instance
app = Flask(__name__)

# Import the routes from routes.py
from app import routes
