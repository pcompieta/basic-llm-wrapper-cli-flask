import argparse  # https://docs.python.org/dev/library/argparse.html
import time
from concurrent.futures import ProcessPoolExecutor, wait, ThreadPoolExecutor
from multiprocessing import Pool
from time import perf_counter
import datetime
import simple_score

parser = argparse.ArgumentParser(description='Load test the LLM, firing many concurrent questions')
parser.add_argument('model_path', help='the absolute path to the model folder')
parser.add_argument('--many', dest='many', default=2, type=int, help='how many concurrent calls to be fired')
parser.add_argument('--delay', dest='delay', default=2, type=int, help='seconds of delay between 1 call and the next')
parser.add_argument('--question', dest='question', default="what is AI Navigator?")
parser.add_argument('--dryrun', dest='dryrun', type=bool, default=False)

args = parser.parse_args()
print(args)

if not args.dryrun:
    model_ready = simple_score.init(args.model_path)


def waste_seconds():
    """
    taking approx 0.7-0.9 sec
    """
    import hashlib
    import os

    def hash_function(inner_data):
        hasher = hashlib.sha256()
        hasher.update(inner_data)
        return hasher.hexdigest()

    data = os.urandom(120 * 1024 * 1024)  # Generate a large block of data: 1 MB
    hash_function(data)


def process_single(responses, question: str, dryrun: bool):
    t_start = perf_counter()
    start_time = datetime.datetime.now()

    if not dryrun:
        json_result = simple_score.process(model_ready["model"], model_ready["tokenizer"], question)
        print(json_result)
    else:
        waste_seconds()

    t_stop = perf_counter()
    end_time = datetime.datetime.now()
    responses.append({"elapsed": t_stop - t_start, "start_time": start_time, "end_time": end_time})


def sequential_test(many: int, question: str, dryrun: bool):
    responses = []
    t_begin = perf_counter()

    for _ in range(many):
        process_single(responses, question, dryrun)

    t_end = perf_counter()
    elapsed = t_end-t_begin
    print(f"Sequential test, total elapsed: {elapsed}")
    print_df_as_table(responses)


def multithread_test(many: int, question: str, dryrun: bool):
    responses = []
    t_begin = perf_counter()

    with ThreadPoolExecutor(max_workers=many) as executor:
        futures = [executor.submit(process_single, responses, question, dryrun) for _ in range(many)]
        wait(futures)

    t_end = perf_counter()
    elapsed = t_end - t_begin
    print(f"Thread-pool test, total elapsed: {elapsed}")
    print_df_as_table(responses)


def multiprocess_test(many: int, question: str, dryrun: bool):
    responses = []
    t_begin = perf_counter()

    with ProcessPoolExecutor(max_workers=many) as executor:
        futures = [executor.submit(process_single, responses, question, dryrun) for _ in range(many)]
        wait(futures)

    t_end = perf_counter()
    elapsed = t_end - t_begin
    print(f"Process-pool test, total elapsed: {elapsed}")
    print_df_as_table(responses)


def print_df_as_table(responses):
    import pandas as pd
    from tabulate import tabulate
    df = pd.DataFrame(responses)
    print(tabulate(df, headers=["No", "elapsed (sec)", "start_time", "end_time"]))


print("-----------------------------------------------------------------")
sequential_test(args.many, args.question, args.dryrun)
print("-----------------------------------------------------------------")
multithread_test(args.many, args.question, args.dryrun)
print("-----------------------------------------------------------------")
# multiprocess_test(args.many, args.question, args.dryrun)
print("-----------------------------------------------------------------")