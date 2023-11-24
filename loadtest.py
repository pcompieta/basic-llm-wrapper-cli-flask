import argparse  # https://docs.python.org/dev/library/argparse.html
from time import perf_counter
import datetime
import simple_score

parser = argparse.ArgumentParser(description='Load test the LLM, firing many concurrent questions')
parser.add_argument('model_path', help='the absolute path to the model folder')
parser.add_argument('--many', dest='many', default=2, type=int, help='how many concurrent calls to be fired')
parser.add_argument('--delay', dest='delay', default=2, type=int, help='seconds of delay between 1 call and the next')
parser.add_argument('--question', dest='question', default="what is AI Navigator?")

args = parser.parse_args()
print(args)

model_ready = simple_score.init(args.model_path)


def sequential_test(many: int):
    responses = []
    t_begin = perf_counter()
    for _ in range(many):
        t_start = perf_counter()
        start_time = datetime.datetime.now()

        json_result = simple_score.process(model_ready["model"], model_ready["tokenizer"], args.question)
        print(json_result)

        t_stop = perf_counter()
        end_time = datetime.datetime.now()
        responses.append({"elapsed": t_stop - t_start, "start_time": start_time, "end_time": end_time})
    t_end = perf_counter()

    print(f"Sequential test, total elapsed: {t_end-t_begin}")
    print_df_as_table(responses)


def print_df_as_table(responses):
    import pandas as pd
    from tabulate import tabulate
    df = pd.DataFrame(responses)
    print(tabulate(df, headers=["No", "elapsed (sec)", "start_time", "end_time"]))


sequential_test(args.many)
