from flask import Flask, render_template
from flask_cors import CORS

app = Flask(__name__, static_url_path='/static')
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/draw')
def canvas():
    return render_template('demoCanvas.html')

app.run(host = "0.0.0.0")