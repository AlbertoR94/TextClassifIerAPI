from flask import Flask, request

from preprocessing import *
from model import *

app = Flask(__name__)

@app.route("/")
def index():
    return "Welcome to Text Classification API"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    my_dataset = myDataset(data, text_transform)
    response_list = make_predictions(convClassifier, my_dataset)
    return {"data": response_list}

@app.route("/predict_one", methods=["POST"])
def predict_one():
    data = request.json
    sentence = data["text"]
    # print(type(sentence))
    if isinstance(sentence, str):
        pred = predict_sentence(convClassifier, sentence)
    else:
        raise ValueError("Invalid input format")
    return pred
