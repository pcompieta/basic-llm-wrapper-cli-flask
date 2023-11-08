import simple_score

while True:
    prompt = input("Ask: ")

    parameters = {
        "max_length": 4000,
        "repetition_penalty": 1.2,
        "max_new_tokens": 200,
        "temperature": 0.1,
        "top_p": 0.95
    }

    json_result = simple_score.process(prompt, parameters)

    print(json_result)
