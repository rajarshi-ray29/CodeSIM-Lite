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
from models.OpenAI import GPT4

from datasets.Dataset import Dataset
from datasets.APPSDataset import APPSDataset
from datasets.MBPPDataset import MBPPDataset
from datasets.XCodeDataset import XCodeDataset
from datasets.HumanEvalDataset import HumanDataset
from datasets.CodeContestDataset import CodeContestDataset

from evaluations.func_evaluate import evaluate_io

from utils.parse import parse_response
from constants.verboseType import *

class SCoder(DirectStrategy):
    def __init__(
        self,
        additional_info_run=2,
        max_plan_try=5,
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
        
        return f"Test Cases where the generated code failed to generate the expected output:\n{"\n".join(falied_test_cases)}"


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
            data_row[self.data.id_key],
            additional_io,
            code,
            self.language
        )

        return passed_sample & passed_additional, self.process_test_log(test_log_sample + test_log_additional)
    

    def run_single_pass(self, data_row: dict):
        print("", flush=True)

        std_input_prompt = ""

        if type(self.data) == APPSDataset or \
            type(self.data) == CodeContestDataset or \
            type(self.data) == XCodeDataset:
            std_input_prompt = "- Strictly follow the input and output format. The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take input using `input()` function then call the function with specified parameters and finally print the output of the function. Do not add extra print statement otherwise it will failed the test cases."

        problem = self.data.get_prompt(data_row)

        additional_io = None

        if type(self.data) == MBPPDataset:

            # Additional IO collection
            for idx in range(1, self.additional_info_run + 1):
                # Additional IO
                additional_io_generation_input = [
                    {
                        "role": "user",
                        "content": prompt_for_additional_io.format(
                            problem=problem,
                            problem_name=data_row["entry_point"],
                        ),
                    },
                ]

                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(f"Input for Additional IO Generation: {idx}\n\n")
                    print(additional_io_generation_input[0]['content'], flush=True)

                response = self.gpt_chat(
                    processed_input=additional_io_generation_input,
                    frequency_penalty=0.2
                )

                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(f"Response from Additional IO Generation: {idx}\n\n")
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

            # Forcing no sample io as MBPP contains no sample io
            data_row['sample_io'] = []

        else:
            additional_io = []
        
        self.run_details["additional_io"] = additional_io

        
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
                plan = response[response.rfind("### Plan"):]

            problem_with_planning = f"## Problem:\n{problem}\n\n{plan}"

            # Simulation Phase
            input_for_simulation = [
                {
                    "role": "user",
                    "content": prompt_for_simulation.format(
                        problem_with_planning=problem_with_planning,
                        language=self.language,
                    )
                },
            ]

            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print(f"Input for Simulation: {plan_no}\n\n")
                print(input_for_simulation[0]['content'], flush=True)

            response = self.gpt_chat(
                processed_input=input_for_simulation
            )

            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print(f"Response from Simulation: {plan_no}\n\n")
                print(response, flush=True)

            if "Plan Modification Needed" in response and \
                "No Plan Modification Needed" not in response:
                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(f"**Plan Modification Needed.**\n")
                
                # Plan Refinement Phase
                input_for_plan_refinement = [
                    {
                        "role": "user",
                        "content": prompt_for_plan_refinement.format(
                            problem_with_planning=problem_with_planning,
                            language=self.language,
                            critique=response
                        )
                    },
                ]

                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(f"Input for Plan Refinement: {plan_no}\n\n")
                    print(input_for_plan_refinement[0]['content'], flush=True)

                plan = self.gpt_chat(
                    processed_input=input_for_simulation
                )

                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(f"Response from Plan Refinement: {plan_no}\n\n")
                    print(plan, flush=True)
                
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

            # problem_with_solution = f"{problem_with_planning}\n\n### Code:\n\n```{self.language}\n{code}\n```"

            # Debugging
            for debug_no in range(1, self.max_debug_try + 1):
                
                input_for_debugging = [
                    {
                        "role": "user",
                        "content": prompt_for_debugging.format(
                            problem_with_planning=problem_with_planning,
                            code=code,
                            language=self.language,
                            test_log=test_log,
                            std_input_prompt=std_input_prompt,
                        )
                    }
                ]

                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(f"Input for Improving code: {plan_no}, {debug_no}\n\n")
                    print(input_for_debugging[0]['content'], flush=True)

                response = self.gpt_chat(input_for_debugging)

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


prompt_for_additional_io = """You are a tester tasked with creating comprehensive unit test cases for a given programming problem.

## Problem

def maximum_segments(n, a, b, c):
    '''
    Write a Python function to find the maximum number of segments of lengths a, b, and c
    that can be formed from n.
    '''

### Problem Understanding

The task is to maximize the number of segments you can cut from a total length `n`, where the possible segment lengths are `a`, `b`, and `c`. Let say we have a rope of length `n` meter. We need to cut it into segments. Possible segment length is `a`, `b`, and `c`. There may be many possible way of doing these segments. We need to find out the maximum number of segments from that rope.

### Test Cases
assert maximum_segments(7, 5, 2, 5) == 2
assert maximum_segments(17, 2, 1, 3) == 17
assert maximum_segments(18, 16, 3, 6) == 6
assert maximum_segments(11, 8, 4, 9) == -1
assert maximum_segments(5, 9, 6, 10) == -1

---

## Problem

{problem}

--------
**Important Instruction:**
For the problem `{problem_name}`
    - First, understand the problem `{problem_name}` and write down the understanding inside **Problem Understanding** section. 
    - Then Generate five (05) unit test cases that cover both:
        - **Normal** and **Edge** case scenarios
        - **Positive** and **Negative** case scenarios
        - **Valid** and **Invalid** case scenarios
    inside **Test Cases** section.
    - Write down each test case in a single line following the pattern shown in the example problem.
    - The generated test cases must be inside a triple backtick (```) code block.
    - Do not generate any code to solve this problem.
"""


prompt_for_initial_code_generation = """{problem}

--------
Important Instructions:
- Generate {language} code step-by-step to solve the above mentioned problem.
- Do not generate any explanation.
- The generated **{language}** code must be enclosed within triple backticks (```).
{std_input_prompt}"""


prompt_for_code_validation = """You are a tester tasked with checking a code for a given problem. 

---

## Problem

{problem}

## Code

{code}

---

**Your output must follow the steps below:**
- Try to generate a test case other than the sample test cases that are mentioned inside the problem.
- Take a the input and apply the code step by step to get the output.
- Compare the generated output with the expected output to verify if the generated code is ok or not.
- Write **Buggy Code** if you find such a test case otherwise write **Code is ok**.
"""


prompt_for_planning = """You are a programmer tasked with generating appropriate plan to solve a given problem using the **{language}** programming language.

## Problem

{problem}

**Expected Output:**

Your response must be structured as follows:

### Problem Understanding

Think about the original problem. Develop an initial understanding about the problem.

### Recall Example Problem

Recall a relevant and distinct problems (different from problem mentioned above) and
- describe it
- generate {language} code step by step to solve that problem
- finally generate a planning to solve that problem

### Plan

- Write down a detailed, step-by-step plan to solve the **original problem**.

--------
**Important Instruction:**
- Strictly follow the instructions.
- Do not generate code.
"""


prompt_for_simulation = """You are a programmer tasked with verifying a plan to solve a given problem using the **{language}** programming language.

{problem_with_planning}

**Expected Output:**

Your response must be structured as follows:

### Simulation

- Take a sample input and apply plan step by step to get the output.
- Compare the generated output with the sample output to verify if your plan works as expected.

### Plan Evaluation

- If the simulation is successful write **No Need to Modify Plan**.
- Otherwise write **Plan Modification Needed**.

---

**Important Instructions:**

- Strictly follow the instructions.
- Do not generate code.
"""


prompt_for_plan_refinement = """You are a programmer tasked with generating appropriate plan to solve a given problem using the **{language}** programming language. You already have a wrong plan. Correct it so that it can generate correct code.

{problem_with_planning}

## Plan Critique

{critique}

## New Plan

- Write down a detailed, step-by-step modified plan to solve the **original problem**.
- Ensure each step logically follows from the previous one.

--------
**Important Instruction:**
- Your response must contain only the plan.
- Do not add any explanation.
- Do not generate code.
"""



prompt_for_code_generation = """You are a programmer tasked with solving a given problem using the **{language}** programming language. See the plan to solve the plan and implement code to solve it.

{problem_with_planning}

--------
**Important Instructions:**
- Do not add any explanation.
- The generated **{language}** code must be inside a triple backtick (```) code block.
{std_input_prompt}"""


prompt_for_debugging = """You are a programmer who has received a solution of a problem written in **{language}** that fails to pass certain test cases. Your task is to modify the code in such a way so that it can pass all the test cases. Do not generate same code.

{problem_with_planning}

### Buggy Code

{code}

### Test Report

{test_log}

### Simulation with failed test case
To detect where is the bug:
    - Take a sample test case where it fails.
    - Take the input go through each step according to the plan
    - You will get a output that must be different from the expected output. 

### Debugging Notes
Based on this simulation detect any of the following cases:
    - Plan is wrong
    - Plan is correct but plan to code generation is wrong.

- Finally, discuss how to correct this code.

### Modified Code

```{language}
# Your corrected code, with comments explaining each correction.
```

--------
**Important Instructions:**
- Strictly follow the instructions.
- Do not add testing code for example assert statement in your code.
- Do not be overconfident that the generated code is correct. It is wrong.
- The modified **{language}** code must be enclosed within triple backticks (```).
{std_input_prompt}"""

