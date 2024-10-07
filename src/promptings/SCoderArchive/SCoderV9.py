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

        print("\n\n" + "_" * 50)
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

            print("\n\n" + "_" * 50)
            print(f"Input for Planning and Coding: {plan_no}\n\n")
            print(input_for_planning_coding[0]['content'], flush=True)

            response, pr_tok_1, com_tok_1 = self.gpt_chat(
                processed_input=input_for_planning_coding
            )

            item['api_calls'] += 1
            pr_tok += pr_tok_1
            com_tok += com_tok_1

            print("\n\n" + "_" * 50)
            print(f"Response from Planning and Coding: {plan_no}\n\n")
            print(response, flush=True)

            if problem not in response:
                problem_with_solution = f"# Problem:\n{problem}\n\n{response}"
            else:
                problem_with_solution = f"{response}"

            if "Plan Modification Needed" in response and "No Plan Modification Needed" not in response:
                print("\n\n" + "_" * 50)
                print(f"Plan Modification Needed. Skipping Debugging.")
                continue

            code = parse_response(response)

            passed, test_log = self.check(item, additional_io, code)
            
            # Do not need to go for debugging steps
            if passed:
                break
            
            # At first feedback is empty
            feedback_response = ''

            # Debugging Phase
            for debug_no in range(1, self.d + 1):
                # Improving code 
                input_for_improving_code = [
                    {
                        "role": "user",
                        "content": prompt_for_debugging.format(
                            problem_with_solution=problem_with_solution,
                            language=self.language,
                            test_log=test_log,
                            std_input_prompt=std_input_prompt,
                            feedback=feedback_response
                        )
                    }
                ]

                print("\n\n" + "_" * 50)
                print(f"Input for Debugging: {plan_no}, {debug_no}\n\n")
                print(input_for_improving_code[0]['content'], flush=True)

                debug_response, pr_tok_1, com_tok_1 = self.gpt_chat(
                    input_for_improving_code
                )
                item['api_calls'] += 1
                pr_tok += pr_tok_1
                com_tok += com_tok_1

                print("\n\n" + "_" * 50)
                print(f"Response from Debugging: {plan_no}, {debug_no}\n\n")
                print(debug_response, flush=True)

                code = parse_response(debug_response)

                # problem_with_solution = f"{problem_with_solution}\n\n## Debug Attempt {debug_no}:\n\n{response}\n"

                passed, test_log = self.check(item, additional_io, code)

                # Passed so breaking this debugging loop
                if passed:
                    break

                # Feedback Generation
                input_for_feedback = [
                    {
                        "role": "user",
                        "content": prompt_for_feedback.format(
                            problem_with_solution=problem_with_solution,
                            language=self.language,
                            test_log=test_log,
                            debug_response=debug_response,
                        )
                    }
                ]

                print("\n\n" + "_" * 50)
                print(f"Input for Feedback: {plan_no}, {debug_no}\n\n")
                print(input_for_feedback[0]['content'], flush=True)

                feedback_response, pr_tok_1, com_tok_1 = self.gpt_chat(
                    input_for_feedback
                )
                item['api_calls'] += 1
                pr_tok += pr_tok_1
                com_tok += com_tok_1

                print("\n\n" + "_" * 50)
                print(f"Response from Feedback: {plan_no}, {debug_no}\n\n")
                print(feedback_response, flush=True)

            
            # Do not need to investigate further 
            if passed:
                break

        print("________________________\n\n", flush=True)
        return code, pr_tok, com_tok



prompt_for_Planning_coding = """# Instructions

You are a programmer tasked with solving a given problem using the **{language}** programming language.

**Steps to Solve the Problem:**

1. **Problem Understanding**

   - Think about the problem.
   - Try to find a similar problem you have encountered before.
   - Develop an initial understanding to guide your planning.

2. **Plan Generation**

   - Write down a detailed, step-by-step plan to solve the problem.
   - Ensure each step logically follows from the previous one.

3. **Simulation**

   - Simulate the sample input according to your plan.
   - Generate the output to verify if your plan works as expected.

4. **Plan Evaluation**

   - **If the simulation is successful**:
     - Write "**No Need to Modify Plan**" and proceed to code generation.
   - **If the simulation fails**:
     - Write "**Plan Modification Needed**" and **stop the generation**; do not proceed to code generation.

5. **Code Generation** *(only if no plan modification is needed)*

   - Write the code according to your plan.
   - Include comments explaining how each part of the code implements the corresponding step of your plan.
   - Ensure the code is clean, readable, and follows best practices.

**General Guidelines:**

- Use the `Decimal` library for precise mathematical calculations.
- Write clean, readable code with appropriate comments.
- Focus on accurately implementing your plan.

---

## Problem

```{language}
{problem}
```

---

**Expected Output:**

Your response should be structured as follows:

### Problem Understanding

[Your thoughts on the problem, any similar problems, and how you plan to solve it.]

### Plan

[Your detailed, step-by-step plan.]

### Simulation

[The simulation of the sample input and the output generated according to your plan.]

### Plan Evaluation

[State whether the plan needs modification or not.]

**If the simulation is successful and no plan modification is needed, proceed to:**

### Code Generation

```{language}
[Your code implementing the plan, with comments explaining each step.]
```

---

**Important:**

- **Strictly follow the instructions.**
- The generated **{language}** code must be inside a triple backtick (```) code block.
{std_input_prompt}
"""


prompt_for_debugging = """# Instructions

You are a programmer who has received some code written in **{language}** that fails to pass certain test cases. Your task is to:

1. **Simulate the Sample Input**

   - Use the provided plan to simulate the sample input.
   - Generate the output to identify where the problem occurs in the code.

2. **Identify Discrepancies**

   - Compare the implemented code with the original plan.
   - Identify any discrepancies or deviations from the plan.

3. **Correct the Code**

   - Modify the code to resolve the issues.
   - Ensure that the corrected code aligns with the original plan.
   - Include comments explaining the corrections made.

**General Guidelines:**

- Use the `Decimal` library for precise mathematical calculations.
- Write clean, readable code with appropriate comments.
- Focus on accurately implementing the corrected plan.

---

{problem_with_solution}

### Test Report

{test_log}

{feedback}

---

**Expected Output:**

Your response should be structured as follows:

### Debugging Notes

[Simulate the test case where it fails.]

### Debugging Notes

[Your explanation of why the code fails.]

### Modified Code

```{language}
[Your corrected code, with comments explaining each correction.]
```

---

**Important:**

- **Strictly follow the instructions.**
- The generated **{language}** code must be enclosed within triple backticks (```).
{std_input_prompt}
"""


prompt_for_feedback = """# Instructions

You are a code reviewer tasked with providing constructive feedback to help improve a piece of code that has failed to solve a given problem, even after initial debugging attempts. Your goal is to help the programmer identify issues in their plan, code, and debugging process, and to offer guidance for correcting these issues.

**You will be provided with:**

- **Problem Description**: The programming problem to be solved.
- **Plan**: The step-by-step plan devised to solve the problem.
- **Original Code**: The initial code written according to the plan.
- **Debugging Notes**: Notes detailing the debugging attempts, including errors encountered and fixes tried.
- **Modified Code**: The code after the debugging attempts.

**Your tasks:**

1. **Analyze the Plan and Code:**

   - Identify any discrepancies between the plan and the original code.
   - Highlight any logical errors or misunderstandings in the plan that may have led to issues in the code.

2. **Review the Debugging Attempts:**

   - Evaluate the debugging notes and the modified code to understand what was attempted.
   - Point out any incorrect assumptions or overlooked issues during the debugging process.

3. **Provide Feedback and Suggestions:**

   - Offer clear, actionable feedback on how to correct the code and improve the plan if necessary.
   - Suggest effective debugging strategies or techniques that could be applied in the next attempt.


---

{problem_with_solution}

### Test Report

{test_log}

{debug_response}

---

**Expected Output:**

Your feedback should be structured into the following sections:

### Feedback and Suggestions
[Your feedback and actionable suggestions here.]


**Important Guidelines:**

- Focus on being helpful and encouraging.
- Do not provide the corrected code; instead, guide the programmer toward finding the solution themselves.
- Keep your feedback organized and concise.
"""


prompt_for_debugging_after_feedback = """# Instructions

You are a programmer who has attempted to solve a programming problem but have not succeeded, even after initial debugging attempts. You have received feedback on your previous attempt.

**Your task is to:**

1. **Review the Feedback:**

   - Carefully read the feedback provided on your plan, code, and debugging efforts.

2. **Analyze the Feedback:**

   - Identify the key issues and misunderstandings highlighted.
   - Understand how these issues affect your plan and code.

3. **Revise Your Plan:**

   - Modify your original plan to address the issues pointed out in the feedback.
   - Ensure that your new plan is detailed and logically sound.

4. **Update Your Code:**

   - Rewrite your code according to the revised plan.
   - Include comments explaining how each part of the code implements the corresponding step of your plan.

5. **Test Your Solution:**

   - Simulate sample inputs and verify that your code produces the correct outputs.
   - Ensure that all test cases are passing.

6. **Reflect on the Process:**

   - Briefly explain how the feedback helped improve your solution.
   - Note any additional insights gained during this iteration.

**General Guidelines:**

- Use the `Decimal` library for precise mathematical calculations (if applicable).
- Write clean, readable code with appropriate comments.
- Keep your focus on implementing the corrected plan accurately.

---

{problem_with_solution}

{feedback}

---

**Expected Output:**

Your revised solution should be structured as follows:

```plaintext
### Revised Plan
[Provide your updated, detailed plan that addresses the feedback.]

### Updated Code
```{language}
[Provide your updated code here, with comments explaining each step.]
```

---

**Important:**

- **Strictly follow the instructions.**
- The generated **{language}** code must be enclosed within triple backticks (```).
{std_input_prompt}
"""

