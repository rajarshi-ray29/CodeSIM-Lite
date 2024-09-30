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

from evaluations.func_evaluate import evaluate_io

from utils.parse import parse_response
from utils.verboseType import *

class SCoder(DirectStrategy):
    def __init__(
        self,
        additional_info_run=2,
        max_plan_try=3,
        max_debug_try=5,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        
        self.additional_info_run=additional_info_run
        self.max_plan_try=max_plan_try
        self.max_debug_try=max_debug_try


        if self.verbose >= VERBOSE_FULL:
            print("\n\n" + "_" * 70)
            print(f"Running SCoder with additional_info_run={additional_info_run}, max_plan_try={max_plan_try}, max_debug_try={max_debug_try}")
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
            data_row: dict,
            additional_io: List[str],
            code: str
    ) -> bool:
        passed_sample, test_log_sample = self.data.evaluate_sample_io(
            data_row,
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
        passed_test_cases = []
        falied_test_cases = []
        for test_log in test_logs.splitlines():
            if test_log.startswith("Passed"):
                passed_test_cases.append(test_log[test_log.index("assert"):])
            if test_log.startswith("Failed"):
                falied_test_cases.append(test_log[test_log.index("assert"):])
        
        return f"Passed Test Cases:\n{"\n".join(passed_test_cases)}\n\nFailed Test Cases:\n{"\n".join(falied_test_cases)}"


    def run_single_pass(self, data_row: dict):
        print("", flush=True)

        std_input_prompt = ""

        if type(self.data) == APPSDataset or \
            type(self.data) == CodeContestDataset or \
            type(self.data) == XCodeDataset:
            std_input_prompt = "- Strictly follow the input and output format. The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take input using `input()` function then call the function with specified parameters and finally print the output of the function. Do not add extra print statement otherwise it will failed the test cases."

        problem = self.data.get_prompt(data_row)

        additional_io = None

        # Additional IO collection
        for idx in range(1, self.additional_info_run + 1):
            # Additional IO
            additional_io_generation_input = [
                {
                    "role": "user",
                    "content": prompt_for_additional_io.format(
                        problem=problem
                    ),
                },
            ]

            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print(f"Input for Additional IO Generation: {idx}")
                print(additional_io_generation_input[0]['content'], flush=True)

            response = self.gpt_chat(
                processed_input=additional_io_generation_input,
                frequency_penalty=0.2
            )

            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print(f"Response from Additional IO Generation: {idx}")
                print(response, flush=True)

            additional_io_response = parse_response(response)

            # Applying intersection for self-consistancy
            if additional_io is None:
                additional_io = set(self.parse_test_cases(
                    test_cases=additional_io_response
                ))
            else:
                additional_io_ = self.parse_test_cases(
                    test_cases=additional_io_response
                )
                additional_io = additional_io.intersection(set(additional_io_))

        additional_io = list(additional_io)
        if self.verbose >= VERBOSE_FULL:
            print(f"Additional IOs:")
            print(additional_io, flush=True)

        self.run_details["additional_io"] = additional_io

        # Check whether the additional IO is correct or not
        # This block is just for keeping track off the correctness of additional IO
        if type(self.data) == HumanDataset:
            passed, _ = evaluate_io(additional_io, f"{data_row["prompt"]}\n\n{data_row["canonical_solution"]}")
            self.run_details["additional_io_correctness"] = passed
            if not passed and self.verbose >= VERBOSE_FULL:
                print("Problem in additional IO")

        # # Forcing no sample io 
        # self.data_row['sample_io'] = []
        # else:
        #     additional_io = []

        code_generation_input = [
            {
                "role": "user",
                "content": prompt_for_initial_code_generation.format(
                    problem=problem,
                    language=self.language,
                    std_input_prompt=std_input_prompt
                ),
            },
        ]

        if self.verbose >= VERBOSE_FULL:
            print("\n\n" + "_" * 70)
            print(f"Input for Code Generation: ")
            print(code_generation_input[0]['content'], flush=True)

        response = self.gpt_chat(
            processed_input=code_generation_input
        )

        if self.verbose >= VERBOSE_FULL:
            print("\n\n" + "_" * 70)
            print(f"Response from Code Generation:")
            print(response, flush=True)

        code = parse_response(response)

        passed, test_log = self.check(data_row, additional_io, code)

        # Early closing for easily solvable problems so that no extra token consumption
        if passed:
            return code

        # Planning, Coding, Debugging
        for plan_no in range(1, self.max_plan_try + 1):
            # Planning Phase
            input_for_planning = [
                {
                    "role": "user",
                    "content": prompt_for_planning.format(
                        problem=problem,
                        language=self.language,
                    )
                },
            ]

            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print(f"Input for Planning: {plan_no}\n\n")
                print(input_for_planning[0]['content'], flush=True)

            response = self.gpt_chat(
                processed_input=input_for_planning
            )

            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print(f"Response from Planning: {plan_no}\n\n")
                print(response, flush=True)
            
            # if "```" in response:
            #     plan = parse_response(response)
            # else:
            #     plan = response[response.find("### Plan"):]
            
            if "### Plan" not in response:
                plan = f"### Plan\n\n{response}"
            else:
                plan = response[response.find("### Plan"):]

            problem_with_planning = f"## Problem:\n{problem}\n\n{plan}"


            # Code generation
            input_for_final_code_generation = [
                {
                    "role": "user",
                    "content": prompt_for_code_generation.format(
                        problem_with_planning=problem_with_planning,
                        language=self.language,
                        std_input_prompt=std_input_prompt,
                    )
                }
            ]

            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print(f"Input for final code generation:\n\n")
                print(input_for_final_code_generation[0]['content'], flush=True)

            response = self.gpt_chat(
                input_for_final_code_generation
            )

            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print(f"Response from final code generation:\n\n")
                print(response, flush=True)

            code = parse_response(response)

            passed, test_log = self.check(data_row, additional_io, code)

            # Do not need to go for debugging steps
            if passed:
                break

            problem_with_solution = f"{problem_with_planning}\n\n### Code:\n\n```{self.language}\n{code}\n```"

            # Debugging
            for debug_no in range(1, self.max_debug_try + 1):
                
                summary_generation_input = [
                    {
                        "role": "user",
                        "content": prompt_for_debugging.format(
                            language=self.language,
                            problem_with_solution=problem_with_solution,
                            test_log=test_log,
                        ),
                    },
                ]

                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(f"Input for Debugging: {plan_no}, {debug_no}\n\n")
                    print(summary_generation_input[0]['content'], flush=True)

                response = self.gpt_chat(summary_generation_input)

                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(f"Response from Debugging: {plan_no}, {debug_no}\n\n")
                    print(response, flush=True)
                
                input_for_improving_code = [
                    {
                        "role": "user",
                        "content": prompt_for_code_improvement.format(
                            problem_with_planning=problem_with_planning,
                            code=code,
                            language=self.language,
                            test_log=test_log,
                            std_input_prompt=std_input_prompt,
                            debug_info=response,
                        )
                    }
                ]

                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(f"Input for Improving code: {plan_no}, {debug_no}\n\n")
                    print(input_for_improving_code[0]['content'], flush=True)

                response = self.gpt_chat(input_for_improving_code)

                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(f"Response from Improving code: {plan_no}, {debug_no}\n\n")
                    print(response, flush=True)

                code = parse_response(response)

                passed, test_log = self.check(data_row, additional_io, code)

                # Passed so breaking this debugging loop
                if passed:
                    break
            
            if passed:
                break

        if self.verbose >= VERBOSE_FULL:
            print("\n\n" + "_" * 70)

        return code



prompt_for_additional_io = """# Instructions

You are a tester tasked with creating comprehensive unit test cases for a given programming problem.

**Your tasks:**

1. **Understand the Problem**

   - Read and comprehend the programming problem provided.

2. **Generate Test Cases**

   - Create unit test cases that cover both **Normal** and **Edge** case scenarios to ensure the correctness of the code.
   - Do not include the test cases that are mentioned in the problem description.
   - Follow the style of the provided example problem while generation.

---

## Example Problem

```python
def maximum_segments(n, a, b, c):
    '''
    Write a Python function to find the maximum number of segments of lengths a, b, and c
    that can be formed from n.

    For Example:
    assert maximum_segments(7, 5, 2, 5) == 2
    '''
```

```
# Test Cases
## Basic Test Cases:
assert maximum_segments(7, 5, 2, 5) == 2
assert maximum_segments(17, 2, 1, 3) == 17
assert maximum_segments(18, 16, 3, 6) == 6

## Edge Test Cases:
assert maximum_segments(11, 8, 4, 9) == -1
assert maximum_segments(5, 9, 6, 10) == -1
```

---

## Problem

```python
{problem}
```

---

## Test Cases

**TODO:** Generate test cases for the problem above.

---

**Important:**

- **Strictly follow the example problem style.**
- **Do not generate more than 5 test cases.**
- **Write each test case in a single line.**
- **Your response must contain the generated test cases enclosed within triple backticks (```).**
- **Write clean and concise test cases that effectively test the functionality of the code.**
"""


prompt_for_initial_code_generation = """{problem}

---
Important Instructions:
- Generate {language} code to solve the above mentioned problem.
{std_input_prompt}"""


prompt_for_planning = """# Instructions

You are a programmer tasked with generating appropriate plan to solve a given problem using the **{language}** programming language.

**Steps to Solve the Problem:**

1. **Recalling Relevant Problems**

   - Try to find a similar problem you have encountered before.
   - State the algorithm to solve that problem.

2. **Problem Understanding**

   - Think about the original problem.
   - Develop an initial understanding to guide your planning.

3. **Plan Generation**

   - Write down a detailed, step-by-step plan to solve the problem.
   - Ensure each step logically follows from the previous one.

---

## Problem

```{language}
{problem}
```

---

**Expected Output:**

Your response should be structured as follows:

### Example Problem

[Recalling a relevant problem and algorithm to solve it.]

### Problem Understanding

[Your thoughts on the problem, any similar problems, and how you plan to solve it.]

### Plan

[Your detailed, step-by-step plan.]

---

**Important:**

- **Strictly follow the instructions.**
- Do not generate code.
"""


prompt_for_code_generation = """# Instructions

You are a programmer tasked with solving a given problem using the **{language}** programming language. See the plan to solve the plan and implement code to solve it.

---

{problem_with_planning}

---

**Expected Output:**

Your response should be structured as follows:

```{language}
[Your code implementing the plan, with comments explaining each step.]
```

---

**Important:**

- **Strictly follow the instructions.**
- Do not add any explanation.
- The generated **{language}** code must be inside a triple backtick (```) code block.
{std_input_prompt}"""


prompt_for_debugging = """# Instructions

You are a programmer who has received some code written in **{language}** that fails to pass certain test cases. Your task is to:

- Select a failed test sample.
- Take the sample input from that test case and apply the generated code on it line by line to see the output of each step.
- Compare the ouput of each step with the plan to identify the buggy statement from the code.

{problem_with_solution}

### Test Report

{test_log}

---

**Expected Output:**

Your response should be structured as follows:

### Simulation of failed test cases

[Simulate the test case where it fails following the above mentioned steps.]

### Debugging Notes

[Write any discrepancies or deviations from the plan in code generation.]
"""


prompt_for_code_improvement = """# Instructions

You are a programmer tasked with solving a given problem using the **{language}** programming language. I have already tried to solve this problem but it results in buggy code. See the buggy code and debugging notes correct the implementation.

---

{problem_with_planning}

### Buggy Code
{code}

### Test Report

{test_log}


{debug_info}

---

**Expected Output:**

Your response should be structured as follows:

```{language}
[Modified code implementing the plan, with comments explaining each step.]
```

---

**Important:**

- **Strictly follow the instructions.**
- Do not add any explanation.
- The generated **{language}** code must be inside a triple backtick (```) code block.
{std_input_prompt}"""

