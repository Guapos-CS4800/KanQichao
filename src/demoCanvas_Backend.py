from flask import Flask, render_template, request
from flask_cors import CORS

app = Flask(__name__, static_url_path='/static')
CORS(app)

@app.route('/draw')
def index():
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def draw():

    if request.method == 'POST':
        print("Sent Back Data")
        

    else:
        return render_template('demoCanvas.html')

@app.route('/submit', methods=['GET','POST'])
def cheetos():
    if request.method =='POST':

        img = request.form.get('testing')
        print(img)

        FILE_WRITE = open("draw.txt", 'w')
        FILE_WRITE.write("sadas")
        FILE_WRITE.close()

        return "POST RECIEVED"


app.run(host = "0.0.0.0")