from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import subprocess
import base64

import json


import testJisho as informer
import classifier.classifyimage as classify


app = Flask(__name__, template_folder='Templates', static_url_path='/static')
CORS(app)


#Home Page
@app.route('/')
def index():
    return render_template('index.html')


#Use Route to Debug CI CD
@app.route('/dong')
def dong():
    return 'Hello World'


#Main Functionality Page
#Contains the cavas drawing element
@app.route('/draw', methods=['GET', 'POST'])
def draw():

    if request.method == 'POST':
        print("Sent Back Data")
    else:
        return render_template('demoCanvas.html')

'''
Submit relates to when the user draws on the canvas
On mouse release all drawing data is sent to backend to be processed
Immediately after a GET request is sent to determine the results
'''
@app.route('/submit', methods=['GET','POST'])
def cheetos():
    if request.method =="GET":

        #Send the predicted characters back to front end
        #Read the results from the file
        f = open("classifier/predictedkanji.txt", "r", encoding='utf-8')
        line_list = f.readlines()
        res = ""
        for line in line_list:
            res += line

        f.close()

        return res

    if request.method =='POST':
        #Retreive Image information
        img = request.get_json()
        img = img['testing']
        
        #Write Encoded Image into a text file for processing later
        FILE_WRITE = open("user_drawings/draw.txt", 'w')
        FILE_WRITE.write(img)
        FILE_WRITE.close()

        #Sanity Check
        #Decode Image to verify it is the same as the drawing
        with open('sample.jpg', 'wb') as f:
            f.write(base64.decodebytes(img.split(',')[1].encode()))
        
        #Given the image, predict the top ten associated kanji
        #Results are put into a file predictedkanji.txt
        classify.main()
        
        #Sanity Check
        #Read the top ten kanji
        f = open("classifier/predictedkanji.txt", "r", encoding='utf-8')
        line_list = f.readlines()
        f.close()

        return "OK"

'''
Route Display is all functionality relating to pushing a button
If the buttons for any of the predicted Kanji are clicked these methods
are called.
Main purpose is to generate relevant information to a specific kanji.
'''
@app.route('/display', methods=['GET','POST'])
def displayKanjiInformation():
    if request.method == "POST":
        #Get the character to process
        sentKanji = request.get_json()
        #Jisho Webscrape
        res = informer.getInfo(sentKanji['kanji'])

        json_obj = json.dumps(res, indent=4)

        #Write the information to a file to grab later
        fileWrite = open("info.json", "w", encoding='utf-8')
        fileWrite.write(json_obj)
        fileWrite.close()

        return 'dong'
    if request.method == 'GET':
        #Read the data, send back to front end
        fileRead = open("info.json", "r", encoding="utf-8")
        json_obj = json.load(fileRead)
        fileRead.close()
        
        return jsonify(json_obj)


if __name__ == '__main__':
    app.run(host = "0.0.0.0")