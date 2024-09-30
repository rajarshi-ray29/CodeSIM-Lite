import pandas as pd
import json
from utils.jsonl import read_jsonl, write_jsonl


def gen_summary(results_path: str, summary_path: str):
    results = pd.DataFrame(read_jsonl(results_path))

    if "api_calls" not in results:
        results["api_calls"] = 1

    solved = len(results.query("is_solved == True"))
    unsolved = len(results.query("is_solved == False"))

    accuracy = solved / (solved + unsolved)

    normal_solved = len(results.query("is_solved == True & api_calls == 2"))
    our_solved = len(results.query("is_solved == True & api_calls > 2"))

    total_prompt_tokens = sum(results["prompt_tokens"].sum())
    total_completion_tokens = sum(results["completion_tokens"].sum())

    average_prompt_tokens = total_prompt_tokens / len(results)
    average_completion_tokens = total_completion_tokens / len(results)
    
    total_api_calls = results["api_calls"].sum()
    max_api_calls = results["api_calls"].max()
    min_api_calls = results["api_calls"].min()
    average_api_calls = total_api_calls / len(results)

    false_results = results.query("is_solved == False")['api_calls'].value_counts()
    true_results = results.query("is_solved == True")['api_calls'].value_counts()

    with open(summary_path, mode="w", encoding="utf-8") as summary_file:
        # Define a width for alignment
        name_width = 30
        value_width = 10

        summary_file.write(f"{'Accuracy:':<{name_width}} {accuracy*100:>{value_width}.02f}\n")
        summary_file.write(f"{'Solved:':<{name_width}} {solved:>{value_width}}\n")
        summary_file.write(f"{'Unsolved:':<{name_width}} {unsolved:>{value_width}}\n")
        summary_file.write(f"\n")
        summary_file.write(f"{'Normal Solved:':<{name_width}} {normal_solved:>{value_width}}\n")
        summary_file.write(f"{'Our Solved:':<{name_width}} {our_solved:>{value_width}}\n")
        summary_file.write(f"\n")
        summary_file.write(f"\n")
        summary_file.write(f"{'Total Prompt Tokens:':<{name_width}} {total_prompt_tokens:>{value_width}}\n")
        summary_file.write(f"{'Average Prompt Tokens:':<{name_width}} {average_prompt_tokens:>{value_width}.0f}\n")
        summary_file.write(f"\n")
        summary_file.write(f"{'Total Completion Tokens:':<{name_width}} {total_completion_tokens:>{value_width}}\n")
        summary_file.write(f"{'Average Completion Tokens:':<{name_width}} {average_completion_tokens:>{value_width}.0f}\n")
        summary_file.write(f"\n")
        summary_file.write(f"{'Total Api Calls:':<{name_width}} {total_api_calls:>{value_width}}\n")
        summary_file.write(f"{'Max Api Calls:':<{name_width}} {max_api_calls:>{value_width}}\n")
        summary_file.write(f"{'Min Api Calls:':<{name_width}} {min_api_calls:>{value_width}}\n")
        summary_file.write(f"{'Average Api Calls:':<{name_width}} {average_api_calls:>{value_width}.02}\n")
        summary_file.write(f"\n")
        summary_file.write(f"\n")
        summary_file.write(f"{'Solved Api Calls':<{name_width}}\n")
        summary_file.write(f"{'Api calls':<{name_width}} {'Solved':>{value_width}}\n")
        # Printing all keys and their values (Solved)
        for key, value in true_results.items():
            summary_file.write(f"{key:<{name_width}} {value:>{value_width}}\n")
        summary_file.write(f"\n")
        summary_file.write(f"{'Unsolved Api Calls':<{name_width}}\n")
        summary_file.write(f"{'Api calls':<{name_width}} {'Unsolved':>{value_width}}\n")
        # Printing all keys and their values (Unsolved)
        for key, value in false_results.items():
            summary_file.write(f"{key:<{name_width}} {value:>{value_width}}\n")

