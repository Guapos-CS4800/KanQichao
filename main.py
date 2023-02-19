from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/hello')
def hello_world():
    return "hello world!"

app.run(host = "0.0.0.0")
