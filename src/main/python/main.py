from flask import Flask
from flask_cors import CORS


def main():
    app = Flask(__name__)
    CORS(app)

    @app.route('/')
    def hello_world():
        return "hello world!"

    @app.route('/dong')
    def goodbye_world():
        return "Our messiah."


    @app.route('/qichao')
    def brandonTest():
        return "Our Savior."

    app.run(host = "0.0.0.0")



if __name__ == '__main__':
    main()


