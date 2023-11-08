import json
import transformers

# INITIALIZATION -- LONG OP -- START

with open('simple_config.json') as config_file:
    configs = json.load(config_file)

model_uri = configs['model_uri']

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_uri,
    low_cpu_mem_usage=True,
    return_dict=True,
    device_map="auto",
    offload_folder="offload_folder"
)

# Reload tokenizer to save it
tokenizer = transformers.AutoTokenizer.from_pretrained(model_uri)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# INITIALIZATION -- LONG OP -- END


def process(prompt: str, parameters):
    """
    CHAT COMPLETION: https://github.com/facebookresearch/llama-recipes/blob/main/examples/chat_completion/chat_completion.py
    GENERIC INFERENCE: https://github.com/facebookresearch/llama-recipes/blob/main/examples/inference.py
    """

    # TODO assess model.generate(...) in place of transformers.pipeline(...)(prompt)
    pipe = transformers.pipeline(task="text-generation", model=model, tokenizer=tokenizer,
                                 max_new_tokens=parameters['max_new_tokens'],  # TODO what's the default ?
                                 max_length=parameters['max_length'],  # max length of output, default=4096
                                 return_full_text=False,  # to not repeat the question, default=True
                                 # num_return_sequences=1,  # TODO what's this for
                                 # top_k=10,  # default=10
                                 top_p=parameters['top_p'],  # default=0.9
                                 temperature=parameters['temperature'],  # default=0
                                 repetition_penalty=parameters['repetition_penalty'],  # TODO check whether honoured or not; and, what's the default ?
                                 )

    json_result = pipe(prompt)
    return json_result
