import transformers


def init(model_uri: str) -> dict:
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

    return {"model": model, "tokenizer": tokenizer}


default_parameters = {
        "max_length": 4000,
        "repetition_penalty": 1.2,
        "max_new_tokens": 200,
        "temperature": 0.1,
        "top_p": 0.95,
    }


def process(model, tokenizer, prompt: str, parameters: dict = None):
    """
    CHAT COMPLETION: https://github.com/facebookresearch/llama-recipes/blob/main/examples/chat_completion/chat_completion.py
    GENERIC INFERENCE: https://github.com/facebookresearch/llama-recipes/blob/main/examples/inference.py
    """

    if parameters is None:
        parameters = default_parameters

    # TODO assess model.generate(...) in place of transformers.pipeline(...)(prompt)
    pipe = transformers.pipeline(task="text-generation", model=model, tokenizer=tokenizer,
                                 max_new_tokens=parameters['max_new_tokens'],  # wins over max_length TODO what default?
                                 # max_length=parameters['max_length'],  # max length of output, default=4096
                                 return_full_text=False,  # to not repeat the question, default=True
                                 repetition_penalty=parameters['repetition_penalty'],
                                 # TODO check whether honoured or not; and, what's the default ?
                                 # num_return_sequences=1,  # TODO what's this for
                                 # top_k=10,  # default=10
                                 # top_p=parameters['top_p'],  # default=0.9 TODO UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.95` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
                                 # temperature=parameters['temperature'],  # default=0 TODO UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.1` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
                                 )

    json_result = pipe(prompt)
    return json_result
