from flask import Flask, render_template, request
from flask_cors import CORS
import subprocess
import base64

import classifier.classifyimage as classify


app = Flask(__name__, static_url_path='/static')
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/draw', methods=['GET', 'POST'])
def draw():

    if request.method == 'POST':
        print("Sent Back Data")
    
    else:
        return render_template('demoCanvas.html')

@app.route('/submit', methods=['GET','POST'])
def cheetos():
    if request.method =="GET":
        print('SUBMIT URL: IN GET METHOD')

        f = open("classifier/predictedkanji.txt", "r", encoding='utf-8')
        line_list = f.readlines()
        res = ""
        for line in line_list:
            res += line
        return res

    if request.method =='POST':
        img = request.get_json()
        img = img['testing']
        
        FILE_WRITE = open("user_drawings/draw.txt", 'w')
        FILE_WRITE.write(img)
        FILE_WRITE.close()

        #TODO: Make decoder for IMG URL to IMG
        with open('sample.jpg', 'wb') as f:
            f.write(base64.decodebytes(img.split(',')[1].encode()))
        
        classify.main()
        
        f = open("classifier/predictedkanji.txt", "r", encoding='utf-8')
        line_list = f.readlines()
        for line in line_list:
            print('List of Kanji: ' + line)

        return "OK"


if __name__ == '__main__':
    app.run(host = "0.0.0.0")