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
        k=1,
        d=1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        self.d = d

    @staticmethod
    def get_sample_io_str(sample_io: any) -> str:
        if len(sample_io) > 0:
            if type(sample_io[0]) == str:
                return "\n".join(sample_io)
            if type(sample_io[0]) == dict:
                return "\n".join([f"Input:\n{io['input']}\nExpected output:\n{io['output'][0]}" for io in sample_io])
        return sample_io

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

        return passed_sample & passed_additional, test_log_sample + test_log_additional
    
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

        if type(self.data) == APPSDataset or type(self.data) == CodeContestDataset or type(self.data) == XCodeDataset:
            std_input_prompt = "- Strictly follow the input and output format. The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take input using `input()` function then call the function with specified parameters and finally print the output of the function. Do not add extra print statement otherwise it will failed the test cases."

        problem = self.data.get_prompt(item)

        item['api_calls'] = 0
        pr_tok = 0
        com_tok = 0

        # Planning Phase

        input_for_planning = [
            {
                "role": "user",
                "content": prompt_for_planning.format(
                    problem=problem
                )
            },
        ]

        print("\n\n________________________")
        print(f"Input for Planning: ")
        print(input_for_planning[0]['content'], flush=True)

        response, pr_tok_1, com_tok_1 = self.gpt_chat(
            processed_input=input_for_planning
        )

        item['api_calls'] += 1
        pr_tok += pr_tok_1
        com_tok += com_tok_1

        print("\n\n________________________")
        print(f"Response from Planning: ")
        print(response, flush=True)

        if problem not in response:
            problem_with_planning = f"# Problem:\n{problem}\n\n{response}"
        else:
            problem_with_planning = f"{response}"

        # Simulation Phase
        input_for_simulation = [
            {
                "role": "user",
                "content": prompt_for_simulation.format(
                    problem_with_planning=problem_with_planning
                )
            },
        ]

        print("\n\n________________________")
        print(f"Input for Simulation: ")
        print(input_for_simulation[0]['content'], flush=True)

        response, pr_tok_1, com_tok_1 = self.gpt_chat(
            processed_input=input_for_simulation
        )

        item['api_calls'] += 1
        pr_tok += pr_tok_1
        com_tok += com_tok_1

        print("\n\n________________________")
        print(f"Response from Simulation: ")
        print(response, flush=True)

        problem_with_planning_simulation = f"{problem_with_planning}\n\n{response}"

        # Critique for the confirmation of plan
        input_for_planning_critique = [
            {
                "role": "user",
                "content": prompt_for_critique.format(
                    problem_with_planning_simulation=problem_with_planning_simulation
                )
            },
        ]

        print("\n\n________________________")
        print(f"Input for Critique: ")
        print(input_for_planning_critique[0]['content'], flush=True)

        response, pr_tok_1, com_tok_1 = self.gpt_chat(
            processed_input=input_for_planning_critique
        )

        item['api_calls'] += 1
        pr_tok += pr_tok_1
        com_tok += com_tok_1

        print("\n\n________________________")
        print(f"Response from Critique: ")
        print(response, flush=True)

        critique = response

        if "Decision: Correct" not in critique:
            # Refinement of plan and simulation
            input_for_planning_refinement = [
                {
                    "role": "user",
                    "content": prompt_for_planning_refinement.format(
                        problem_with_planning_simulation=problem_with_planning_simulation,
                        critique=critique
                    )
                },
            ]

            print("\n\n________________________")
            print(f"Input for Planning Simulation Refinement: ")
            print(input_for_planning_refinement[0]['content'], flush=True)

            response, pr_tok_1, com_tok_1 = self.gpt_chat(
                processed_input=input_for_planning_refinement
            )

            item['api_calls'] += 1
            pr_tok += pr_tok_1
            com_tok += com_tok_1

            print("\n\n________________________")
            print(f"Response from Planning Simulation Refinement: ")
            print(response, flush=True)

            if problem not in response:
                problem_with_planning_simulation = f"# Problem:\n{problem}\n\n{response}"
            else:
                problem_with_planning_simulation = f"{response}"
        

        # Additional IO generation
        additional_io_generation_input = [
            {
                "role": "user",
                "content": prompt_for_additional_io.format(
                    problem_with_planning_simulation=problem_with_planning_simulation
                ),
            },
        ]

        print("\n\n________________________")
        print(f"Input for Additional IO Generation: ")
        print(additional_io_generation_input[0]['content'], flush=True)

        response, pr_tok_1, com_tok_1 = self.gpt_chat(
            processed_input=additional_io_generation_input,
            frequency_penalty=0.5
        )
        item['api_calls'] += 1
        pr_tok += pr_tok_1
        com_tok += com_tok_1

        print("\n\n________________________")
        print(f"Response from Additional IO Generation:")
        print(response, flush=True)

        additional_io = self.parse_test_cases(
            test_cases=parse_response(response)
        )

        input_for_final_code_generation = [
            {
                "role": "user",
                "content": prompt_for_code_generation.format(
                    problem_with_planning_simulation=problem_with_planning_simulation,
                    language=self.language,
                    std_input_prompt=std_input_prompt,
                )
            }
        ]

        print("\n\n________________________")
        print(f"Input for final code generation:")
        print(input_for_final_code_generation[0]['content'], flush=True)

        code, pr_tok_1, com_tok_1 = self.gpt_chat(
            input_for_final_code_generation
        )
        item['api_calls'] += 1
        pr_tok += pr_tok_1
        com_tok += com_tok_1

        code = parse_response(code)

        print("\n\n________________________")
        print(f"Response from final code generation:")
        print(code, flush=True)


        passed = False

        problem_with_solution = f"{problem_with_planning}\n\n## Code:\n\n```{self.language}\n{code}\n```"

        # Debugging
        for i in range(1, self.d + 1):
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
            print("Input for Debugging: ")
            print(input_for_improving_code[0]['content'], flush=True)

            response, pr_tok_1, com_tok_1 = self.gpt_chat(
                input_for_improving_code
            )
            item['api_calls'] += 1
            pr_tok += pr_tok_1
            com_tok += com_tok_1
            # time.sleep(1)

            code = parse_response(response)

            print("\n\n________________________")
            print("Response from Debugging: ")
            print(response, flush=True)
            

        print("________________________\n\n", flush=True)
        return code, pr_tok, com_tok



prompt_for_planning = '''# Example Problem:

## Problem Description:

```python
def left_rotate(s, d):
    """
    Write a Python function to left rotate the string by d positions.
    
    For Example:
    assert left_rotate("python", 2) == "thonpy"
    assert left_rotate("bigdata", 3) == "databig"
    assert left_rotate("hadoop", 1) == "adooph"
    """
```

## Problem Understanding:

Given a string `s` and an integer `d`, the task is to **left rotate** the string by `d` positions. In simpler terms, the first `d` characters of the string will be moved to the end. This is a typical string manipulation problem that involves substring operations.

## Plan:

1. **Extract the first `d` characters** from the string. Let's call this substring `s1`.
2. **Extract the remaining characters** starting from index `d` to the end. This will be `s2`.
3. **Concatenate `s2` and `s1`** to get the final rotated string.


# Example Problem:

## Problem Description:

```python
def maximum_segments(n, a, b, c):
    """
    Write a Python function to find the maximum number of segments of lengths a, b, and c
    that can be formed from n.
    
    For Example:
    assert maximum_segments(7, 5, 2, 5) == 2
    """
```

## Problem Understanding:

The task is to **maximize the number of segments** you can cut from a total length `n`, where the possible segment lengths are `a`, `b`, and `c`. The problem is similar to the classic **Coin Changing Problem** where the task is to determine the highest number of coins that add up to a given amount. This problem can be solved using **dynamic programming (DP)** to find the optimal number of segments.

## Plan:

1. Initialize a DP array `dp[]` where `dp[i]` stores the **maximum number of segments** that can be obtained for length `i`. 
   - Set `dp[0] = 0` (no segments for length 0), and all other values to `-1` (impossible lengths).
   
2. For each length `i` from 0 to `n`, if a solution exists (`dp[i] != -1`), update the values for `dp[i + a]`, `dp[i + b]`, and `dp[i + c]` by adding 1 to `dp[i]`.

3. The result is found in `dp[n]`, which gives the maximum number of segments.


# Example Problem:

## Problem Description:

```python
def camel_to_snake(text):
    """
    Write a Python function to convert a CamelCase string to a snake_case string.
    
    For Example:
    assert camel_to_snake('PythonProgram') == 'python_program'
    assert camel_to_snake('pythonLanguage') == 'python_language'
    """
```

## Problem Understanding:

The task is to convert a string from **CamelCase** (e.g., `PythonProgram`) to **snake_case** (e.g., `python_program`). In snake_case, words are separated by underscores (`_`), and all letters are lowercase.

## Plan:

1. **Iterate** through each character of the string.
   - If the character is lowercase, append it to the result.
   - If the character is uppercase:
     - If it's the first character, just append the lowercase version.
     - Otherwise, append an underscore (`_`) followed by the lowercase version.


# Problem:

## Problem Description:

```python
{problem}
```

## TODO: Generate Problem Understanding and a step by step Plan to solve the given problem.

----------------
Instructions: 
- Suppose you are an expert programmer. By seeing a problem you first understand the problem then generate a plan to solve the problem. 
- Give a paragraph containing problem description followed by a details step by step plan to solve the programming problem.
- Follow the style shown in the given example problem.
- Your response must contain only two sections. They are **Problem Understanding** and **Plan**.'''


prompt_for_simulation = '''# Example Problem:

## Problem Description:

```python
def left_rotate(s, d):
    """
    Write a Python function to left rotate the string by d positions.
    
    For Example:
    assert left_rotate("python", 2) == "thonpy"
    assert left_rotate("bigdata", 3) == "databig"
    assert left_rotate("hadoop", 1) == "adooph"
    """
```

## Problem Understanding:

Given a string `s` and an integer `d`, the task is to **left rotate** the string by `d` positions. In simpler terms, the first `d` characters of the string will be moved to the end. This is a typical string manipulation problem that involves substring operations.

## Plan:

1. **Extract the first `d` characters** from the string. Let's call this substring `s1`.
2. **Extract the remaining characters** starting from index `d` to the end. This will be `s2`.
3. **Concatenate `s2` and `s1`** to get the final rotated string.

## Simulation:

### Example 1: `left_rotate("python", 2)`

1. First 2 characters: `"py"`
2. Remaining part from index 2: `"thon"`
3. Concatenate: `"thon" + "py" = "thonpy"`

### Example 2: `left_rotate("bigdata", 3)`

1. First 3 characters: `"big"`
2. Remaining part from index 3: `"data"`
3. Concatenate: `"data" + "big" = "databig"`

### Example 3: `left_rotate("hadoop", 1)`

1. First 1 character: `"h"`
2. Remaining part from index 1: `"adoop"`
3. Concatenate: `"adoop" + "h" = "adooph"`

# Example Problem:

## Problem Description:

```python
def maximum_segments(n, a, b, c):
    """
    Write a Python function to find the maximum number of segments of lengths a, b, and c
    that can be formed from n.
    
    For Example:
    assert maximum_segments(7, 5, 2, 5) == 2
    """
```

## Problem Understanding:

The task is to **maximize the number of segments** you can cut from a total length `n`, where the possible segment lengths are `a`, `b`, and `c`. The problem is similar to the classic **Coin Changing Problem** where the task is to determine the highest number of coins that add up to a given amount. This problem can be solved using **dynamic programming (DP)** to find the optimal number of segments.

## Plan:

1. Initialize a DP array `dp[]` where `dp[i]` stores the **maximum number of segments** that can be obtained for length `i`. 
   - Set `dp[0] = 0` (no segments for length 0), and all other values to `-1` (impossible lengths).
   
2. For each length `i` from 0 to `n`, if a solution exists (`dp[i] != -1`), update the values for `dp[i + a]`, `dp[i + b]`, and `dp[i + c]` by adding 1 to `dp[i]`.

3. The result is found in `dp[n]`, which gives the maximum number of segments.

## Simulation:

### Example 1: `maximum_segments(7, 5, 2, 5)`

1. **Initialize** the DP array: `dp = [0, -1, -1, -1, -1, -1, -1, -1]`.
   
2. Process each length `i` from 0 to 7:
   - For `i = 0`, update:
     - `dp[5] = max(dp[5], dp[0] + 1) = 1`
     - `dp[2] = max(dp[2], dp[0] + 1) = 1`
   - For `i = 2`, update:
     - `dp[7] = max(dp[7], dp[2] + 1) = 2`
     - `dp[4] = max(dp[4], dp[2] + 1) = 2`
   - For `i = 5`, update:
     - `dp[7]` remains 2.
     
3. **Final result**: `dp[7] = 2`, meaning two segments can be cut from length 7.

# Example Problem:

## Problem Description:

```python
def camel_to_snake(text):
    """
    Write a Python function to convert a CamelCase string to a snake_case string.
    
    For Example:
    assert camel_to_snake('PythonProgram') == 'python_program'
    assert camel_to_snake('pythonLanguage') == 'python_language'
    """
```

## Problem Understanding:

The task is to convert a string from **CamelCase** (e.g., `PythonProgram`) to **snake_case** (e.g., `python_program`). In snake_case, words are separated by underscores (`_`), and all letters are lowercase.

## Plan:

1. **Iterate** through each character of the string.
   - If the character is lowercase, append it to the result.
   - If the character is uppercase:
     - If it's the first character, just append the lowercase version.
     - Otherwise, append an underscore (`_`) followed by the lowercase version.

## Simulation:

### Example 1: `camel_to_snake('PythonProgram')`

1. `'P'` is uppercase, and it's the first character → Append `'p'`.
2. `'y'`, `'t'`, `'h'`, `'o'`, `'n'` are lowercase → Append as is.
3. `'P'` is uppercase, and not the first character → Append `'_p'`.
4. `'r'`, `'o'`, `'g'`, `'r'`, `'a'`, `'m'` are lowercase → Append as is.

Result: `"python_program"`

### Example 2: `camel_to_snake('pythonLanguage')`

1. `'p'`, `'y'`, `'t'`, `'h'`, `'o'`, `'n'` are lowercase → Append as is.
2. `'L'` is uppercase, not the first character → Append `'_l'`.
3. `'a'`, `'n'`, `'g'`, `'u'`, `'a'`, `'g'`, `'e'` are lowercase → Append as is.

Result: `"python_language"`

{problem_with_planning}

## TODO: Generate Simulation.

----------------
Instructions: 
- Suppose you are an expert programmer. After getting a plan you always simulatie the steps and try to map from input to output to see the correctness of the plan.
- Follow the style shown in the given example problem and generate simulation for the given problem. 
- Your response must contain only the **Simulation* section.'''


prompt_for_critique = '''{problem_with_planning_simulation}

----------------
Instructions: 
- Suppose you are an expert programmer.
- You are given a problem along with a plan and simulation that maps input to output according to the plan.
- Comment on the correctness of the plan and simulation given here. If there is any mistake give suggestion to make it correct.
- At last line specify in one word whether correct or wrong in the given format: `# Decision: Correct` or `# Decision: Wrong`.'''


prompt_for_planning_refinement = '''{problem_with_planning_simulation}

# Critique
{critique}

----------------
Instructions: 
- Suppose you are an expert programmer.
- You are given a problem along with a plan and simulation that maps input to output according to the plan.
- But the plan and simulation has some problem that was pointed out in Critique section.
- Follow the suggestion and update the plan and simulation to make it correct.
- Your output must contain only three sections **Problem Understanding**, **Plan** and **Simulation**.'''


prompt_for_additional_io = """# Example Problem:
```python
def maximum_segments(n, a, b, c):
    '''
    Write a Python function to find the maximum number of segments of lengths a, b, and c
    that can be formed from n.
    
    For Example:
    assert maximum_segments(7, 5, 2, 5) == 2
    '''
```

## Problem Understanding:

The task is to **maximize the number of segments** you can cut from a total length `n`, where the possible segment lengths are `a`, `b`, and `c`. The problem is similar to the classic **Coin Changing Problem** where the task is to determine the highest number of coins that add up to a given amount. This problem can be solved using **dynamic programming (DP)** to find the optimal number of segments.

## Plan:

1. Initialize a DP array `dp[]` where `dp[i]` stores the **maximum number of segments** that can be obtained for length `i`. 
   - Set `dp[0] = 0` (no segments for length 0), and all other values to `-1` (impossible lengths).
   
2. For each length `i` from 0 to `n`, if a solution exists (`dp[i] != -1`), update the values for `dp[i + a]`, `dp[i + b]`, and `dp[i + c]` by adding 1 to `dp[i]`.

3. The result is found in `dp[n]`, which gives the maximum number of segments.

## Simulation:

### Example 1: `maximum_segments(7, 5, 2, 5)`

1. **Initialize** the DP array: `dp = [0, -1, -1, -1, -1, -1, -1, -1]`.
   
2. Process each length `i` from 0 to 7:
   - For `i = 0`, update:
     - `dp[5] = max(dp[5], dp[0] + 1) = 1`
     - `dp[2] = max(dp[2], dp[0] + 1) = 1`
   - For `i = 2`, update:
     - `dp[7] = max(dp[7], dp[2] + 1) = 2`
     - `dp[4] = max(dp[4], dp[2] + 1) = 2`
   - For `i = 5`, update:
     - `dp[7]` remains 2.
     
3. **Final result**: `dp[7] = 2`, meaning two segments can be cut from length 7.

```
# Test Cases
## Basic Test Cases:
assert maximum_segments(7, 5, 2, 5) == 2
assert maximum_segments(17, 2, 1, 3) == 17
assert maximum_segments(18, 16, 3, 6) == 6
assert maximum_segments(6, 5, 6, 3) == 2
assert maximum_segments(4, 7, 4, 2) == 2

## Edge Test Cases:
assert maximum_segments(11, 8, 4, 9) == -1
assert maximum_segments(5, 9, 6, 10) == -1
assert maximum_segments(2, 10, 6, 10) == -1
assert maximum_segments(18, 13, 7, 11) == 2
assert maximum_segments(17, 20, 1, 5) == 17
```

{problem_with_planning_simulation}

## TODO: Generate test cases for the above problem.

----------------
Instructions: 
- Suppose you are a program tester, your task is to create comprehensive test cases for a programming problem. You can see the plan to solve the problem and simulation form input to output.
- The test cases must contain both the **Normal** and **Edge** case scenarios to ensure the code's correctness.
- Very careful while generating additional test cases because it can result in wrong code generation in subsequent step. 
- Do not generate more than ten (10) test cases.
- Do not generate code.
- Your response must contain the generated test cases inside ``` block. No other ``` block is allowed."""


prompt_for_code_generation = """{problem_with_planning_simulation}

## TODO: Let's think step by step and generate {language} code to solve the problem according to the the given plan and simulation.

----------------
Instructions: 
- Suppose you are expert in {language} programming, your task is to solve the given problem in {language} programming language.
- Follow the plan and take help from the simulation to generate the code.
- Your response must contain the {language} code to solve this problem inside ``` block. No other ``` block is allowed.
{std_input_prompt}"""


prompt_for_debugging = """{problem_with_solution}

## Test Report:
{test_log}

----------------
Instructions: 
- Suppose you are expert in {language} programming.
- Your {language} code fails to pass all the test cases. 
- Verify the plan and if necessary change the plan and generate {language} code accordingly.
- Your response must contain the modified {language} code inside ``` block. No other ``` block is allowed.
{std_input_prompt}"""


prompt_for_initial_code_generation = """# Problem
{problem}

## TODO: Let's think step by step and generate {language} code to solve the problem.

----------------
Instructions: 
- Suppose you are a {language} programming expert, your task is to solve the given problem in {language} programming language.
- Your response must contain the {language} code to solve this problem inside ``` block. Do not include any other ``` block in your response.
{std_input_prompt}"""

