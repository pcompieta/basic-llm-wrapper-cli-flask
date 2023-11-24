import json

from flask import Flask, request
import simple_score


app = Flask(__name__)

with open('simple_config.json') as config_file:
    configs = json.load(config_file)

model_ready = simple_score.init(configs['model_path'])


@app.route("/")
def running():
    return "The llm-score microservice is running!"


@app.route("/score", methods=["POST"])
def score():
    payload = request.get_json()
    prompt = payload['prompt']
    parameters = payload['parameters']

    json_result = simple_score.process(model_ready["model"], model_ready["tokenizer"], prompt, parameters)
    return json_result
