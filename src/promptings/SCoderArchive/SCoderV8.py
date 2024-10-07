from typing import List
import tiktoken
import os
import json
import re
import sys
import time

from copy import deepcopy
import xml.etree.ElementTree as ET

from .Base import BaseStrategy
from .Direct import DirectStrategy
from models.Base import BaseModel

from datasets.Dataset import Dataset
from datasets.APPSDataset import APPSDataset
from datasets.MBPPDataset import MBPPDataset
from datasets.XCodeDataset import XCodeDataset
from datasets.HumanEvalDataset import HumanDataset
from datasets.CodeContestDataset import CodeContestDataset

from utils.parse import parse_response


class SCoder(DirectStrategy):
    def __init__(
        self,
        k=3,
        d=5,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        self.d = d

        print("\n\n________________________")
        print(f"Running SCoder with k={self.k}, d={self.d}")
        print("\n", flush=True)


    @staticmethod
    def get_sample_io_str(sample_io: any) -> str:
        if len(sample_io) > 0:
            if type(sample_io[0]) == str:
                return "\n".join(sample_io)
            if type(sample_io[0]) == dict:
                return "\n".join([f"Input:\n{io['input']}\nExpected output:\n{io['output'][0]}" for io in sample_io])
        return sample_io
    

    @staticmethod
    def process_test_log(test_logs: str):
        passed_test_cases = []
        falied_test_cases = []
        for test_log in test_logs.splitlines():
            if test_log.startswith("Passed"):
                passed_test_cases.append(test_log[test_log.index("assert"):])
            if test_log.startswith("Failed"):
                falied_test_cases.append(test_log[test_log.index("assert"):])
        
        return f"Passed Test Cases:\n{"\n".join(passed_test_cases)}\n\nFailed Test Cases:\n{"\n".join(falied_test_cases)}"


    def parse_test_cases(self, test_cases: str):
        return [
            test_case
            for test_case in test_cases.splitlines()
            if len(test_case) > 0 and not test_case.startswith("#")
        ]


    def check(
            self,
            item: dict,
            additional_io: List[str],
            code: str
    ) -> bool:
        passed_sample, test_log_sample = self.data.evaluate_sample_io(
            item,
            code,
            self.language
        )
        
        passed_additional, test_log_additional = self.data.evaluate_additional_io(
            additional_io,
            code,
            self.language
        )

        return passed_sample & passed_additional, self.process_test_log(test_log_sample + test_log_additional)
    

    @staticmethod
    def process_test_log(test_logs: str):
        if len(test_logs) == 0:
            return ""
    
        passed_test_cases = []
        falied_test_cases = []
        for test_log in test_logs.splitlines():
            if test_log.startswith("Passed"):
                passed_test_cases.append(test_log[test_log.index("assert"):])
            if test_log.startswith("Failed"):
                falied_test_cases.append(test_log[test_log.index("assert"):])
        
        return f"Passed Test Cases:\n{"\n".join(passed_test_cases)}\n\nFailed Test Cases:\n{"\n".join(falied_test_cases)}"


    def run_single_pass(self, item: dict):
        print("", flush=True)

        std_input_prompt = ""

        if type(self.data) == APPSDataset or type(self.data) == CodeContestDataset or type(self.data) == XCodeDataset:
            std_input_prompt = "- Strictly follow the input and output format. The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take input using `input()` function then call the function with specified parameters and finally print the output of the function. Do not add extra print statement otherwise it will failed the test cases."

        problem = self.data.get_prompt(item)

        item['api_calls'] = 0
        pr_tok = 0
        com_tok = 0


        additional_io = []

        
        for plan_no in range(1, self.k+1):
            # Planning and Coding Phase
            input_for_planning_coding = [
                {
                    "role": "user",
                    "content": prompt_for_Planning_coding.format(
                        problem=problem,
                        language=self.language,
                        std_input_prompt=std_input_prompt,
                    )
                },
            ]

            print("\n\n________________________")
            print(f"Input for Planning and Coding: {plan_no}")
            print(input_for_planning_coding[0]['content'], flush=True)

            response, pr_tok_1, com_tok_1 = self.gpt_chat(
                processed_input=input_for_planning_coding
            )

            item['api_calls'] += 1
            pr_tok += pr_tok_1
            com_tok += com_tok_1

            print("\n\n________________________")
            print(f"Response from Planning and Coding: {plan_no}")
            print(response, flush=True)

            if problem not in response:
                problem_with_solution = f"# Problem:\n{problem}\n\n{response}"
            else:
                problem_with_solution = f"{response}"

            if "Plan Modification Needed" in response and "No Plan Modification Needed" not in response:
                print("\n\n________________________")
                print(f"Plan Modification Needed. Skipping Debugging.")
                continue

            code = parse_response(response)

            passed = False

            # Debugging Phase
            for debug_no in range(1, self.d + 1):
                passed, test_log = self.check(item, additional_io, code)

                if passed:
                    break

                input_for_improving_code = [
                    {
                        "role": "user",
                        "content": prompt_for_debugging.format(
                            problem_with_solution=problem_with_solution,
                            language=self.language,
                            test_log=test_log,
                            std_input_prompt=std_input_prompt,
                        )
                    }
                ]

                print("\n\n________________________")
                print(f"Input for Debugging: {plan_no}, {debug_no}")
                print(input_for_improving_code[0]['content'], flush=True)

                response, pr_tok_1, com_tok_1 = self.gpt_chat(
                    input_for_improving_code
                )
                item['api_calls'] += 1
                pr_tok += pr_tok_1
                com_tok += com_tok_1


                print("\n\n________________________")
                print(f"Response from Debugging: {plan_no}, {debug_no}")
                print(response, flush=True)


                code = parse_response(response)

                # problem_with_solution = f"{problem_with_solution}\n\n## Debug Attempt {debug_no}:\n\n{response}\n"


            if passed:
                break

        print("________________________\n\n", flush=True)
        return code, pr_tok, com_tok



prompt_for_Planning_coding = """# Instructions
Suppose you are a programmer, your task is to solve a given programming problem in {language} programming language.

For solving a problem:
- Step 1: Think about the problem.
- Step 2: Generate a plan.
- Step 3: According to plan simulation the sample input and check whether it gives desired output or not.
- Step 4: If plan is ok generate code according to the plan.
- Step 5: If plan is not ok do not need to generate code.

General Guideline:
- For precise mathematical calculations use `Decimal` library.

## Problem
```{language}
{problem}
```

### Problem Understanding
TODO: Think about the problem and try to find a similar problem that you already know and formulate a plan to solve this problem.

### Plan
TODO: Write step by step detail plan to solve this problem.

### Simulation
TODO: Simulate the sample input according to plan and generate output to see whether the plan works.

### Need Plan Modification
TODO:
- Write **No Need to Modify Plan** if simulation is ok. Proceed to Code generation.
- Write **Plan Modification Needed** if simulation goes wrong.

**If simulation is ok and no plan modification is needed follow the steps below:**

### Code
```{language}
# TODO: Write code according to your plan. Put comments about the implemented plan with the corresponding code.
```

----------------
Important:
- Strictly follow the instructions.
- The generated {language} code must be inside ``` block.
{std_input_prompt}"""


prompt_for_debugging = """# Instructions
- Suppose you are a programmer.
- You will receive a {language} code that fails to pass some test cases. Try to figure out the mistake and correct it.
- See the plan and identify the deviation in coding and correct it.
- Simulate the failed test cases find out the bug.

General Guidelines:
    - For precise mathematics use `Decimal` library.
    - Try new way if previous way is not working.

{problem_with_solution}

### Test Report:
{test_log}

### Explanation:
TODO: Explain why the generated code fails to solve the problem.

### Modified Code
```{language}
# TODO: Modify the code so that it can solve the problem.
```

----------------
Important:
- Strictly follow the instructions.
- The generated {language} code must be inside ``` block.
{std_input_prompt}"""

