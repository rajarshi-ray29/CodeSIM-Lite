from utils.jsonl import read_jsonl, write_jsonl
from evaluations.func_evaluate import evaluate_io_et
import os


def generate_ep_dataset_human(
    NORMAL_RESULTS_PATH,
    EP_SAMPLES_PATH,
):
    samples = []
    results = read_jsonl(NORMAL_RESULTS_PATH)
    for result in results:
        completion = result["source_codes"][-1]

        if "from typing import *" not in completion:
            completion = "from typing import *\n" + completion

        samples.append(
            {
                "task_id": result["task_id"],
                "solution": completion,
                # "completion": result["solution"]
            }
        )

    write_jsonl(EP_SAMPLES_PATH, samples)


mbpp_not_included_set = set([
    "Mbpp/304", "Mbpp/393", "Mbpp/399", "Mbpp/401", "Mbpp/408",
    "Mbpp/411", "Mbpp/417", "Mbpp/434", "Mbpp/443", "Mbpp/444",
    "Mbpp/452", "Mbpp/464", "Mbpp/584", "Mbpp/617", "Mbpp/625",
    "Mbpp/627", "Mbpp/738", "Mbpp/747", "Mbpp/756", "Mbpp/776",
    "Mbpp/802", "Mbpp/228", "Mbpp/291"
])

def generate_ep_dataset_mbpp(
        NORMAL_RESULTS_PATH,
        EP_SAMPLES_PATH,
):
    samples = []
    results = read_jsonl(NORMAL_RESULTS_PATH)
    for result in results:
        completion = result["source_codes"][-1]
        task_id = "Mbpp/" + result["name"].split("_")[1]
        if task_id in mbpp_not_included_set:
            continue

        if "from typing import *" not in completion:
            completion = "from typing import *\n" + completion

        samples.append(
            {
                "task_id": task_id,
                "solution": completion
            }
        )

    write_jsonl(EP_SAMPLES_PATH, samples)

