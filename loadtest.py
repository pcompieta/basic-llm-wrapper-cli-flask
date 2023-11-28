import argparse  # https://docs.python.org/dev/library/argparse.html
import time
from concurrent.futures import ProcessPoolExecutor, wait, ThreadPoolExecutor
import multiprocessing
from time import perf_counter
import datetime
import simple_score


def parse_args():
    global args
    parser = argparse.ArgumentParser(description='Load test the LLM, firing many concurrent questions')
    parser.add_argument('model_path', help='the absolute path to the model folder')
    parser.add_argument('--many', dest='many', default=2, type=int, help='how many concurrent calls to be fired')
    parser.add_argument('--delay', dest='delay', default=1.0, type=float, choices=range(1, 10), help='seconds of delay between calls')
    parser.add_argument('--question', dest='question', default="what is AI Navigator?")
    parser.add_argument('--dryrun', dest='dryrun', action='store_true')  # False when not provided
    parser.add_argument('--busy_cpu_sec', dest='busy_cpu_sec', default=2.0, type=float, help='secs of compute in case of dryrun')
    args = parser.parse_args()
    print(args)
    return args


def keep_cpu_busy(seconds_of_computation: float = 1.0):
    import hashlib
    import os

    def hash_function(inner_data):
        hasher = hashlib.sha256()
        hasher.update(inner_data)
        return hasher.hexdigest()

    seconds_to_waste_int = int(seconds_of_computation * 125 * 1024 * 1024)
    data = os.urandom(seconds_to_waste_int)  # Generate large block of data: hashing at 125 MB/sec
    hash_function(data)


def process_single(responses, question: str, dryrun: bool, busy_cpu_sec: float):
    t_start = perf_counter()
    start_time = datetime.datetime.now().isoformat(sep=" ", timespec='milliseconds')

    if not dryrun:
        simple_score.process(model_ready["model"], model_ready["tokenizer"], question)
    else:
        keep_cpu_busy(busy_cpu_sec)

    t_stop = perf_counter()
    end_time = datetime.datetime.now().isoformat(sep=" ", timespec='milliseconds')
    responses.append({"elapsed": t_stop - t_start, "start_time": start_time, "end_time": end_time})


def sequential_test(many: int, question: str, dryrun: bool, busy_cpu_sec: float):
    responses = []
    t_begin = perf_counter()

    for _ in range(many):
        process_single(responses, question, dryrun, busy_cpu_sec)

    t_end = perf_counter()
    total_elapsed = t_end - t_begin
    throughput = many / total_elapsed
    print(f"""SEQUENTIAL TEST
          total_elapsed={total_elapsed:.1f}s
          throughput={throughput:.2f} job/s
          """)
    print_df_as_table(responses)


def multithread_test(many: int, question: str, dryrun: bool, delay: int, busy_cpu_sec: float):
    responses = []
    t_begin = perf_counter()

    with ThreadPoolExecutor(max_workers=many) as executor:
        futures = []
        for _ in range(many):
            futures.append(executor.submit(process_single, responses, question, dryrun, busy_cpu_sec))
            time.sleep(delay)
        wait(futures, return_when="FIRST_EXCEPTION")

    t_end = perf_counter()
    total_elapsed = t_end - t_begin
    throughput = many / total_elapsed
    print(f"""THREADPOOL TEST
          total_elapsed = {total_elapsed:.1f}s
          throughput = {throughput:.2f} job/s
          delay = {delay:.1f}s
          """)
    print_df_as_table(responses)


def multiprocess_test(many: int, question: str, dryrun: bool, delay: int, busy_cpu_sec: float):  # NOT WORKING YET
    responses = []
    t_begin = perf_counter()

    futures = []
    with ProcessPoolExecutor() as executor:
        for _ in range(many):
            futures.append(executor.submit(process_single, responses, question, dryrun, busy_cpu_sec))
            time.sleep(delay)
        wait(futures, return_when="FIRST_EXCEPTION")

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
    args = parse_args()

    if not args.dryrun:
        model_ready = simple_score.init(args.model_path)

    print("=======================================================================")
    sequential_test(args.many, args.question, args.dryrun, args.busy_cpu_sec)
    print("=======================================================================")
    multithread_test(args.many, args.question, args.dryrun, 0, args.busy_cpu_sec)
    print("=======================================================================")
    multithread_test(args.many, args.question, args.dryrun, args.delay, args.busy_cpu_sec)
    # print("=======================================================================")
    # multiprocess_test(args.many, args.question, args.dryrun, 0)
    # print("=======================================================================")
    # multiprocess_test(args.many, args.question, args.dryrun, args.delay)
