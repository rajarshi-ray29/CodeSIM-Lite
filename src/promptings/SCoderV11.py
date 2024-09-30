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

        print("\n\n" + "_" * 70)
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

        if type(self.data) == APPSDataset or \
            type(self.data) == CodeContestDataset or \
            type(self.data) == XCodeDataset:
            std_input_prompt = "- Strictly follow the input and output format. The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take input using `input()` function then call the function with specified parameters and finally print the output of the function. Do not add extra print statement otherwise it will failed the test cases."

        problem = self.data.get_prompt(item)

        item['api_calls'] = 0
        pr_tok = 0
        com_tok = 0

        # Additional IO
        additional_io_generation_input = [
            {
                "role": "user",
                "content": prompt_for_additional_io.format(
                    problem=problem
                ),
            },
        ]

        print("\n\n" + "_" * 70)
        print(f"Input for Additional IO Generation: ")
        print(additional_io_generation_input[0]['content'], flush=True)

        response, pr_tok_1, com_tok_1 = self.gpt_chat(
            processed_input=additional_io_generation_input,
            frequency_penalty=0.2
        )
        item['api_calls'] += 1
        pr_tok += pr_tok_1
        com_tok += com_tok_1

        print("\n\n" + "_" * 70)
        print(f"Response from Additional IO Generation:")
        print(response, flush=True)

        additional_io_response = parse_response(response)

        additional_io = self.parse_test_cases(
            test_cases=additional_io_response
        )

        print(f"Additional IOs:")
        print(additional_io, flush=True)

        # Check whether the additional IO is correct or not
        if type(self.data) == HumanDataset:
            passed, _ = evaluate_io(additional_io, f"{item["prompt"]}\n\n{item["canonical_solution"]}")
            item["additional_io_correctness"] = passed
            if not passed:
                print("Problem in additional IO")

        # # Forcing no sample io 
        # self.item['sample_io'] = []
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

        print("\n\n" + "_" * 70)
        print(f"Input for Code Generation: ")
        print(code_generation_input[0]['content'], flush=True)

        response, pr_tok_1, com_tok_1 = self.gpt_chat(
            processed_input=code_generation_input
        )

        item['api_calls'] += 1
        pr_tok += pr_tok_1
        com_tok += com_tok_1

        print("\n\n" + "_" * 70)
        print(f"Response from Code Generation:")
        print(response, flush=True)

        code = parse_response(response)

        passed, test_log = self.check(item, additional_io, code)

        # Early closing for easily solvable problems so that no extra token consumption
        if passed:
            return code, pr_tok, com_tok

        # For first pass no summary
        summary = ''

        # Planning, Coding, Debugging
        for plan_no in range(1, self.k+1):
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

            print("\n\n" + "_" * 70)
            print(f"Input for Planning: {plan_no}\n\n")
            print(input_for_planning[0]['content'], flush=True)

            response, pr_tok_1, com_tok_1 = self.gpt_chat(
                processed_input=input_for_planning
            )

            item['api_calls'] += 1
            pr_tok += pr_tok_1
            com_tok += com_tok_1

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

            problem_with_planning = f"# Problem:\n{problem}\n\n{plan}"
            

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

            print("\n\n" + "_" * 70)
            print(f"Input for Simulation: {plan_no}\n\n")
            print(input_for_simulation[0]['content'], flush=True)

            response, pr_tok_1, com_tok_1 = self.gpt_chat(
                processed_input=input_for_simulation
            )

            item['api_calls'] += 1
            pr_tok += pr_tok_1
            com_tok += com_tok_1

            print("\n\n" + "_" * 70)
            print(f"Response from Simulation: {plan_no}\n\n")
            print(response, flush=True)

            if "Plan Modification Needed" in response and "No Plan Modification Needed" not in response:
                print("\n\n" + "_" * 70)
                print(f"Plan Modification Needed. Skipping Debugging.")
                continue


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

            print("\n\n" + "_" * 70)
            print(f"Input for final code generation:\n\n")
            print(input_for_final_code_generation[0]['content'], flush=True)

            response, pr_tok_1, com_tok_1 = self.gpt_chat(
                input_for_final_code_generation
            )

            item['api_calls'] += 1
            pr_tok += pr_tok_1
            com_tok += com_tok_1

            print("\n\n" + "_" * 70)
            print(f"Response from final code generation:\n\n")
            print(response, flush=True)

            code = parse_response(response)

            passed, test_log = self.check(item, additional_io, code)

            # Do not need to go for debugging steps
            if passed:
                break

            problem_with_solution = f"{problem_with_planning}\n\n## Code:\n\n```{self.language}\n{code}\n```"

            # Debugging
            for debug_no in range(1, self.d + 1):
                

                input_for_improving_code = [
                    {
                        "role": "user",
                        "content": prompt_for_debugging.format(
                            problem_with_solution=problem_with_solution,
                            language=self.language,
                            test_log=test_log,
                            std_input_prompt=std_input_prompt,
                            summary=summary
                        )
                    }
                ]

                print("\n\n" + "_" * 70)
                print(f"Input for Debugging: {plan_no}, {debug_no}\n\n")
                print(input_for_improving_code[0]['content'], flush=True)

                response, pr_tok_1, com_tok_1 = self.gpt_chat(
                    input_for_improving_code
                )
                item['api_calls'] += 1
                pr_tok += pr_tok_1
                com_tok += com_tok_1

                print("\n\n" + "_" * 70)
                print(f"Response from Debugging: {plan_no}, {debug_no}\n\n")
                print(response, flush=True)

                code = parse_response(response)

                # problem_with_solution = f"{problem_with_solution}\n\n## Debug Attempt {debug_no}:\n\n{response}\n"

                passed, test_log = self.check(item, additional_io, code)

                # Passed so breaking this debugging loop
                if passed:
                    break
            
                summary_generation_input = [
                    {
                        "role": "user",
                        "content": prompt_for_summary.format(
                            problem_with_solution=problem_with_solution,
                            test_log=test_log,
                            debug_response=response
                        ),
                    },
                ]

                print("\n\n" + "_" * 70)
                print(f"Input for Summary: {plan_no}, {debug_no}\n\n")
                print(summary_generation_input[0]['content'], flush=True)

                response, pr_tok_1, com_tok_1 = self.gpt_chat(
                    processed_input=summary_generation_input,
                )
                item['api_calls'] += 1
                pr_tok += pr_tok_1
                com_tok += com_tok_1

                print("\n\n" + "_" * 70)
                print(f"Response from Summary: {plan_no}, {debug_no}\n\n")
                print(response, flush=True)

                summary = f"### Experience from previous run\n\n{response}\n\n[Use this experience to debug the code.]"


            if passed:
                break

        print("________________________\n\n", flush=True)
        return code, pr_tok, com_tok



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


prompt_for_simulation = """# Instructions

You are a programmer tasked with verifying a plan to solve a given problem using the **{language}** programming language.

**Steps to Evaluate the Plan:**

1. **Simulation**

   - Take the sample input and apply plan step by step to get the output.
   - Compare the generated output with the sample output to verify if your plan works as expected.

2. **Plan Evaluation**

   - **If the simulation is successful**: Write "**No Need to Modify Plan**".
   - **If the simulation fails**: Write "**Plan Modification Needed**".

---

{problem_with_planning}

---

**Expected Output:**

Your response should be structured as follows:

### Simulation

[The simulation of the sample input according to the plan.]

### Plan Evaluation

[**No Need to Modify Plan** or **Plan Modification Needed**]

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

1. **Simulation**

    - Select a failed test sample.
    - Take the sample input from that test case and apply the generated code on it line by line to see the output of each step.
    - Compare the ouput of each step with the plan to identify the buggy statement from the code.

2. **Correct the Code**

    - Modify the code to resolve the issues.
    - Ensure that the corrected code aligns with the original plan.
    - Include comments explaining the corrections made.

---

{problem_with_solution}

### Test Report

{test_log}

{summary}

---

**Expected Output:**

Your response should be structured as follows:

### Simulation of failed test cases

[Simulate the test case where it fails.]

### Debugging Notes

[Write any discrepancies or deviations from the plan in code generation.]

### Modified Code

```{language}
[Your corrected code, with comments explaining each correction.]
```

---

**Important:**

- **Strictly follow the instructions.**
- The generated **{language}** code must be enclosed within triple backticks (```).
{std_input_prompt}"""


prompt_for_summary = """# Instructions

You are a code reviewer tasked with providing the summary based on the previous attempts so that on the next try programmer can avoid that path.

---

{problem_with_solution}

{debug_response}

### Test Report

{test_log}

### Summary
[Write summary here]

---

**Important Guidelines:**
- Your summary should be concise and should inital code generation attempt and also the debugging attempt.
"""


