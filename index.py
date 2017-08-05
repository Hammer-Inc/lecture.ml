from flask import Flask
app = Flask(__name__)

@app.route('/get' method = ['GET'])
def get():
	return "hello world"


