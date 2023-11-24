import simple_score

import argparse  # https://docs.python.org/dev/library/argparse.html

parser = argparse.ArgumentParser(description='Engage with the model via CLI')
parser.add_argument('model_path', help='the absolute path to the model folder')
args = parser.parse_args()

model_ready = simple_score.init(args.model_path)

while True:
    prompt = input("Ask: ")
    json_result = simple_score.process(model_ready["model"], model_ready["tokenizer"], prompt)
    print(json_result)
