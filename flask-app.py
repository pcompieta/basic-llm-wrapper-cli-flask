from flask import Flask, request
import simple_score


app = Flask(__name__)


@app.route("/")
def running():
    return "The llm-score microservice is running!"


@app.route("/score", methods=["POST"])
def score():
    payload = request.get_json()
    prompt = payload['prompt']
    parameters = payload['parameters']

    json_result = simple_score.process(prompt, parameters)
    return json_result
