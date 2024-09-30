from typing import List
import tiktoken
import os
import re
from copy import deepcopy

from .Base import BaseStrategy
from models.Base import BaseModel
from datasets.Dataset import Dataset
from results.Results import Results

# self-generate exemplars and knowledge
class AnalogicalStrategy(BaseStrategy):
    
    def run_single_pass(self, data_row: dict):
        input = [
            {
                "role": "user",
                "content": 
f"""Your goal is to write {self.language} code to solve competitive programming problems. Given a problem , explain the core concepts in it and provide other relevant problems. Then solve the original problem.

# Problem:
{self.data.get_prompt(data_row)}

# Instruction: (Your response must include the following points sequentially)

## Algorithms:
Identify the core concepts or algorithms used to solve the problem.

## Tutorial:
Write a useful tutorial about these algorithms.

## Example Problems: 
Provide three examples of relevant competitive programming problems that involve these algorithms. For each problem , describe the problem , explain the solution in detail , and then write the correct Python3 code.

## {self.language} code to solve the original problem: 
Include the following points in your response: 
- Explanation of the solution: 
- {self.language} code to solve the problem (inside ```  ``` block):""",
            },
        ]

        return self.gpt_chat(
            processed_input=input
        )


