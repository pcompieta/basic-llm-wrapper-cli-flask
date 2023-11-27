# Introduction
Python REST wrapper for consuming LLAMA and other models running on VM.

Takes inspiration from: https://github.com/facebookresearch/llama/tree/main

# Download Models

Via git clone (with LFS):
```shell
export HF_USERNAME=username
export HF_TOKEN=token # taken from https://huggingface.co/settings/tokens
export HF_MODELOWNER="meta-llama"
export HF_MODEL_REPO="Llama-2-7b-chat-hf" 
GIT_LFS_SKIP_SMUDGE=1 git clone https://$HF_USERNAME:$HF_TOKEN@huggingface.co/$HF_MODELOWNER/$HF_MODEL_REPO  # light git-clone
cd $HF_MODEL_REPO
git lfs pull  # using LFS to download is faster (parallel, with resume)
```

List of models of interests (can do `git clone` on the below)
* Tiny (for dev purposes): https://huggingface.co/Maykeye/TinyLLama-v0
* LLAMA 2.0 7B: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
* LLAMA 2.0 13B: https://huggingface.co/meta-llama/Llama-2-13b-chat-hf

# Build
IDE: PyCharm or IntelliJ Idea recommended, VS Code should be also working. You may want to install all possible plugins for managing venvs, requirements.txt, and flask. 

Create a Virtual Env
```shell
python3 -m venv venv
. venv/bin/activate
```

Install all deps
```shell
# pip uninstall -y -r <(pip freeze) # cleans the VENV uninstalling all libs
pip install -r requirements.txt
```

# Launch

Please launch entry point as below:
```shell
source ./venv/bin/activate
python -m flask --app ./flask-app.py run --host 0.0.0.0 --port 5003
```

Please note this takes 1-2 minutes to load Llama libs & models into memory.


# Score

Once main program is up and running, it can be invoked like below.

## Simple format
```shell
curl -X POST http://127.0.0.1:5000/score --header 'Content-Type: application/json' -d '
  {
    "prompt" : "How are you?",
    "parameters" : {
        "repetition_penalty": 1.2,
        "max_new_tokens": 200,
        "temperature": 0.1,
        "top_p": 0.95
    }
  }'
```

## Advanced CHAT LLAMA format

```shell
curl -X POST http://127.0.0.1:5003/score --header 'Content-Type: application/json' -d '
  {
    "prompt" : "[INST]<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible. Your answers should only answer the question once and not have any text after the answer is done.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you dont know the answer  to a question, please dont share false information. Answer must be in detail Answer should have formatted  list. Do not mention about text formatting in response. Do not use \"*\" or asterisk  symbol in text formatting in response answer\n<</SYS>>\n\nQUESTION:/n/n What is a Request for Proposal?[/INST]\nHelpful Answer:",
    "parameters" : {
        "max_length": 4000,
        "repetition_penalty": 1.2,
        "max_new_tokens": 200,
        "temperature": 0.1,
        "top_p": 0.95
    }
  }'
```
# Load test

To see help on how to launch load testing, open a shell with the current VirtualEnv and execute the below:
```shell
python loadtest.py -h
```

Example invocations (note: dry-run does not use LLM but rather a simple routing keeping CPU actively busy)
```shell
python loadtest.py /path/to/model/TinyLLama-v0 --many 12 --delay 1 --dryrun --busy_cpu_sec 20
python loadtest.py /path/to/model/TinyLLama-v0 --many 32 --question "What is the best recipe for pancakes?"
```