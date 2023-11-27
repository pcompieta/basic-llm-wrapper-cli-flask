import argparse  # https://docs.python.org/dev/library/argparse.html
import time
from concurrent.futures import ProcessPoolExecutor, wait, ThreadPoolExecutor
import multiprocessing
from time import perf_counter
import datetime
import simple_score


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

    data = os.urandom(250 * 1024 * 1024)  # Generate a large block of data: 250 MB of data, approx 2s to be hashed
    hash_function(data)


def process_single(responses, question: str, dryrun: bool):
    t_start = perf_counter()
    start_time = datetime.datetime.now().isoformat(sep=" ", timespec='milliseconds')

    if not dryrun:
        json_result = simple_score.process(model_ready["model"], model_ready["tokenizer"], question)
        print(json_result)
    else:
        waste_seconds()

    t_stop = perf_counter()
    end_time = datetime.datetime.now().isoformat(sep=" ", timespec='milliseconds')
    responses.append({"elapsed": t_stop - t_start, "start_time": start_time, "end_time": end_time})


def sequential_test(many: int, question: str, dryrun: bool):
    responses = []
    t_begin = perf_counter()

    for _ in range(many):
        process_single(responses, question, dryrun)

    t_end = perf_counter()
    total_elapsed = t_end - t_begin
    throughput = many / total_elapsed
    print(f"""SEQUENTIAL TEST
          total_elapsed={total_elapsed:.1f}
          throughput={throughput:.1f}
          """)
    print_df_as_table(responses)


def multithread_test(many: int, question: str, dryrun: bool, delay: int = 0):
    responses = []
    t_begin = perf_counter()

    with ThreadPoolExecutor(max_workers=many) as executor:
        futures = []
        for _ in range(many):
            futures.append(executor.submit(process_single, responses, question, dryrun))
            time.sleep(delay)
        wait(futures)

    t_end = perf_counter()
    total_elapsed = t_end - t_begin
    throughput = many / total_elapsed
    print(f"""THREADPOOL TEST
          total_elapsed={total_elapsed:.1f}
          throughput={throughput:.1f}
          """)
    print_df_as_table(responses)


def multiprocess_test(many: int, question: str, dryrun: bool, delay: int = 0):  # NOT WORKING YET
    responses = []
    t_begin = perf_counter()

    futures = []
    with ProcessPoolExecutor() as executor:
        for _ in range(many):
            futures.append(executor.submit(process_single, responses, question, dryrun))
            time.sleep(delay)
        wait(futures)

    t_end = perf_counter()
    total_elapsed = t_end - t_begin
    throughput = many / total_elapsed
    print(f"""PROCESSPOOL TEST
          total_elapsed={total_elapsed:.1f}
          throughput={throughput:.1f}
          """)
    print_df_as_table(responses)


def print_df_as_table(responses):
    import pandas as pd
    from tabulate import tabulate
    df = pd.DataFrame(responses)
    print(tabulate(df, headers=["No", "elapsed (sec)", "start_time", "end_time"]))


if __name__ == '__main__':
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description='Load test the LLM, firing many concurrent questions')
    parser.add_argument('model_path', help='the absolute path to the model folder')
    parser.add_argument('--many', dest='many', default=2, type=int, help='how many concurrent calls to be fired')
    parser.add_argument('--delay', dest='delay', default=1, type=int, choices=range(1, 10), help='seconds of delay between calls')
    parser.add_argument('--question', dest='question', default="what is AI Navigator?")
    parser.add_argument('--dryrun', dest='dryrun', type=bool, default=False)

    args = parser.parse_args()
    print(args)

    if not args.dryrun:
        model_ready = simple_score.init(args.model_path)

    print("=======================================================================")
    sequential_test(args.many, args.question, args.dryrun)
    print("=======================================================================")
    multithread_test(args.many, args.question, args.dryrun, 0)
    print("=======================================================================")
    multithread_test(args.many, args.question, args.dryrun, args.delay)
    # print("=======================================================================")
    # multiprocess_test(args.many, args.question, args.dryrun, 0)
    # print("=======================================================================")
    # multiprocess_test(args.many, args.question, args.dryrun, args.delay)
