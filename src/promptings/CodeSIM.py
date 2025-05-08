from typing import List
import tiktoken
import os
import re
import time

from copy import deepcopy

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
from constants.verboseType import *

# ---------------------------------------------------------------------------
# Lightweight prompts – drastically shorter to fit small‑model context
# ---------------------------------------------------------------------------

PROMPT_PLAN = (
    "You are given a programming problem. Write a concise, step‑by‑step plan (bullet list) to solve it using {language}. Do *not* write any code.\n\n"
    "## Problem\n{problem}\n"
)

PROMPT_SIMULATE = (
    "Below is a problem followed by a proposed plan. Manually walk through one simple input (you invent it if none provided). "
    "If the plan would produce the right output, reply with *OK. Otherwise reply with **REVISE* and give a one‑line reason.\n\n"
    "{problem_with_planning}"
)

PROMPT_REFINE = (
    "The previous simulation said the plan needs changes. Produce a revised concise plan (bullet list). Do *not* add explanations or code.\n\n"
    "{problem_with_planning}\n\n# Reason\n{critique}"
)

PROMPT_CODE = (
    "Write full {language} code that follows the plan below and solves the problem. Return *only* the code inside triple back‑ticks.\n\n"
    "{problem_with_planning}"
)

PROMPT_DEBUG = (
    "You are given a failing solution. Explain briefly why it fails in one sentence, then provide corrected {language} code *only* (inside triple back‑ticks).\n\n"
    "Problem & Plan\n{problem_with_planning}\n\n"  # minimal context
    "### Failing code\n{language}\n{code}\n\n\n"
    "### Test feedback\n{test_log}"
)

# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

class CodeSIM(DirectStrategy):
    """A lighter‑weight variant of the original CodeSIM with shorter prompts and an
    optional early‑exit when the sample test passes (helpful for small models)."""

    def __init__(
        self,
        additional_info_run: int = 0,
        max_plan_try: int = 3,
        max_debug_try: int = 3,
        early_exit_on_sample_pass: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.additional_info_run = additional_info_run
        self.max_plan_try = max_plan_try
        self.max_debug_try = max_debug_try
        self.early_exit_on_sample_pass = early_exit_on_sample_pass

        self.is_competitive = isinstance(self.data, (APPSDataset, CodeContestDataset, XCodeDataset))

        if self.verbose >= VERBOSE_FULL:
            print("\n" + "_" * 60)
            print(
                f"CodeSIM (light) – plan_try={max_plan_try}, debug_try={max_debug_try}, early_exit={early_exit_on_sample_pass}",
                flush=True,
            )

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _count_tokens(text: str, enc_name: str = "cl100k_base") -> int:
        """Rough token count for budgeting (optional)."""
        enc = tiktoken.get_encoding(enc_name)
        return len(enc.encode(text))

    @staticmethod
    def _sample_io_to_str(sample_io: any) -> str:
        if sample_io and isinstance(sample_io[0], str):
            return "\n".join(sample_io)
        if sample_io and isinstance(sample_io[0], dict):
            return "\n".join(
                f"Input:\n{io['input']}\nExpected:\n{io['output'][0]}" for io in sample_io
            )
        return ""

    @staticmethod
    def _process_test_log(test_logs: str):
        failed = [line for line in test_logs.splitlines() if line.startswith("Failed")]
        return "\n".join(failed[:5])  # trim

    # ------------------------------------------------------------------
    # Evaluation wrapper that also returns whether sample IO passed
    # ------------------------------------------------------------------

    # def check(self, data_row: dict, additional_io: List[str], code: str):
    #     passed_sample, log_sample = self.data.evaluate_sample_io(data_row, code, self.language)
    #     passed_extra, log_extra = self.data.evaluate_additional_io(
    #         data_row[self.data.id_key], additional_io, code, self.language
    #     )

    #     overall_pass = passed_sample and passed_extra
    #     combined_log = self._process_test_log(log_sample + log_extra)
    #     return overall_pass, combined_log, passed_sample
    def check(self, data_row: dict, additional_io: List[str], code: str):
        passed_sample, log_sample = self.data.evaluate_sample_io(data_row, code, self.language)
        passed_extra, log_extra = self.data.evaluate_additional_io(
            data_row[self.data.id_key], additional_io, code, self.language
        )

        overall_pass = passed_sample and passed_extra
        combined_log = self._process_test_log(log_sample + log_extra)
        
        # Calculate pass percentage
        test_lines = combined_log.splitlines()
        total_tests = len(test_lines)
        failed_tests = sum(1 for line in test_lines if line.startswith("Failed"))
        pass_percentage = (total_tests - failed_tests) / total_tests if total_tests > 0 else 0
        
        return overall_pass, combined_log, passed_sample, pass_percentage


    # ------------------------------------------------------------------
    # Main driver
    # ------------------------------------------------------------------

    # def run_single_pass(self, data_row: dict):
    #     problem = self.data.get_prompt(data_row)

    #     # Remove long competition boilerplate if necessary
    #     if self.is_competitive and "-------\nImportant" in problem:
    #         problem = problem.split("-------\nImportant")[0]

    #     additional_io: List[str] = []  # Disabled for light version
    #     self.run_details["additional_io"] = additional_io

    #     # --------------------------------------------------------------
    #     # PLAN ↦ (optional simulation & refinement) ↦ CODE ↦ DEBUG
    #     # --------------------------------------------------------------
    #     for plan_no in range(1, self.max_plan_try + 1):
    #         # ---- Planning
    #         plan_prompt = PROMPT_PLAN.format(problem=problem, language=self.language)
    #         plan_resp = self.gpt_chat([{"role": "user", "content": plan_prompt}])
    #         plan = plan_resp if "-" in plan_resp else f"- {plan_resp.strip()}"
    #         problem_with_plan = f"## Problem\n{problem}\n\n### Plan\n{plan}"

    #         # ---- Simulation check (cheap)
    #         sim_prompt = PROMPT_SIMULATE.format(problem_with_planning=problem_with_plan)
    #         sim_resp = self.gpt_chat([{"role": "user", "content": sim_prompt}])
    #         if "REVISE" in sim_resp.upper():
    #             refine_prompt = PROMPT_REFINE.format(
    #                 problem_with_planning=problem_with_plan,
    #                 critique=sim_resp,
    #             )
    #             plan = self.gpt_chat([{"role": "user", "content": refine_prompt}])
    #             problem_with_plan = f"## Problem\n{problem}\n\n### Plan\n{plan}"

    #         # ---- Code generation
    #         code_prompt = PROMPT_CODE.format(problem_with_planning=problem_with_plan, language=self.language)
    #         code_resp = self.gpt_chat([{"role": "user", "content": code_prompt}])
    #         code = parse_response(code_resp)

    #         # ---- Quick evaluation
    #         passed, log, passed_sample = self.check(data_row, additional_io, code)
    #         elapsed_time = time.time() - start_time
    #         if passed or (self.early_exit_on_sample_pass and passed_sample) or (elapsed_time > 120 and pass_percentage > 0.7):
    #             return code  # good enough

    #         # ---- Debug loop
    #         for dbg_no in range(1, self.max_debug_try + 1):
    #             dbg_prompt = PROMPT_DEBUG.format(
    #                 problem_with_planning=problem_with_plan,
    #                 code=code,
    #                 language=self.language,
    #                 test_log=log,
    #             )
    #             dbg_resp = self.gpt_chat([{"role": "user", "content": dbg_prompt}])
    #             code = parse_response(dbg_resp)

    #             passed, log, passed_sample = self.check(data_row, additional_io, code)
    #             if passed or (self.early_exit_on_sample_pass and passed_sample):
    #                 return code
    def run_single_pass(self, data_row: dict):
        start_time = time.time()  # Add this line to track execution time
        problem = self.data.get_prompt(data_row)

        # Remove long competition boilerplate if necessary
        if self.is_competitive and "-------\nImportant" in problem:
            problem = problem.split("-------\nImportant")[0]

        additional_io: List[str] = []  # Disabled for light version
        self.run_details["additional_io"] = additional_io

        # --------------------------------------------------------------
        # PLAN ↦ (optional simulation & refinement) ↦ CODE ↦ DEBUG
        # --------------------------------------------------------------
        for plan_no in range(1, self.max_plan_try + 1):
            # Check timeout before starting a new planning cycle
            if time.time() - start_time > 300:  # 5 minutes timeout
                print(f"Timeout reached for {data_row[self.data.id_key]}")
                return code if 'code' in locals() else ""  # Return best code so far or empty string
                
            # ---- Planning
            plan_prompt = PROMPT_PLAN.format(problem=problem, language=self.language)
            plan_resp = self.gpt_chat([{"role": "user", "content": plan_prompt}])
            plan = plan_resp if "-" in plan_resp else f"- {plan_resp.strip()}"
            problem_with_plan = f"## Problem\n{problem}\n\n### Plan\n{plan}"

            # ---- Simulation check (cheap)
            sim_prompt = PROMPT_SIMULATE.format(problem_with_planning=problem_with_plan)
            sim_resp = self.gpt_chat([{"role": "user", "content": sim_prompt}])
            if "REVISE" in sim_resp.upper():
                refine_prompt = PROMPT_REFINE.format(
                    problem_with_planning=problem_with_plan,
                    critique=sim_resp,
                )
                plan = self.gpt_chat([{"role": "user", "content": refine_prompt}])
                problem_with_plan = f"## Problem\n{problem}\n\n### Plan\n{plan}"

            # ---- Code generation
            code_prompt = PROMPT_CODE.format(problem_with_planning=problem_with_plan, language=self.language)
            code_resp = self.gpt_chat([{"role": "user", "content": code_prompt}])
            code = parse_response(code_resp)

            # ---- Quick evaluation
            passed, log, passed_sample, pass_percentage = self.check(data_row, additional_io, code)
            
            # Calculate pass percentage for partial success
            test_lines = log.splitlines()
            total_tests = len(test_lines)
            failed_tests = sum(1 for line in test_lines if line.startswith("Failed"))
            pass_percentage = (total_tests - failed_tests) / total_tests if total_tests > 0 else 0
            
            elapsed_time = time.time() - start_time
            if passed or (self.early_exit_on_sample_pass and passed_sample) or (elapsed_time > 120 and pass_percentage > 0.7):
                return code  # good enough

            # ---- Debug loop
            for dbg_no in range(1, self.max_debug_try + 1):
                # Check timeout before starting a new debug cycle
                if time.time() - start_time > 300:  # 5 minutes timeout
                    return code  # Return best code so far
                    
                # Trim test logs for long-running problems
                if elapsed_time > 120:
                    log = "\n".join(log.splitlines()[:3])  # Only show first 3 failures
                    
                dbg_prompt = PROMPT_DEBUG.format(
                    problem_with_planning=problem_with_plan,
                    code=code,
                    language=self.language,
                    test_log=log,
                )
                dbg_resp = self.gpt_chat([{"role": "user", "content": dbg_prompt}])
                code = parse_response(dbg_resp)

                passed, log, passed_sample, pass_percentage = self.check(data_row, additional_io, code)
                
                # Recalculate pass percentage
                test_lines = log.splitlines()
                total_tests = len(test_lines)
                failed_tests = sum(1 for line in test_lines if line.startswith("Failed"))
                pass_percentage = (total_tests - failed_tests) / total_tests if total_tests > 0 else 0
                
                elapsed_time = time.time() - start_time
                if passed or (self.early_exit_on_sample_pass and passed_sample) or (elapsed_time > 180 and pass_percentage > 0.8):
                    return code

        # Fallback – return last attempt even if failing (caller may handle)
        return code