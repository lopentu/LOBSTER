import argparse
import os
import glob
import json
import asyncio
import pathlib
from rich.live import Live
import time
from dotenv import load_dotenv
from google.api_core import retry_async

import google.genai
import google.genai.types
import google.genai.errors

# initialize gemini client
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")

gemini_client = google.genai.Client(api_key=api_key)

def is_retryable(e) -> bool:
    if hasattr(e, 'code') and e.code == 500:
        return False
    return True

@retry_async.AsyncRetry(predicate=is_retryable)
async def call_gemini(**kwargs):
    return await gemini_client.aio.models.generate_content(**kwargs)

# Problem <-> Answer + Explanation
# Question <-> Entry

# fuzzy metrics
async def fuzzy_llm(solver_entry, reference_entry, problem_content, solver_explanation):

    retry = True
    while retry:
        system_instruction = '''You are a grader for Linguistic Olympiad problems. Your job is to grade a student's submission for a problem. Do not solve the problem yourself.'''

        if reference_entry["type"] == "select":
            prompt = f'''<<TASK>>
Grade the student's answer for a specific question ({reference_entry["question_id"]}) in the given Linguistic Olympiad problem.

<<Grading instructions>>
Check how many distinct items in the student's answer matches (is semantically equivalent to) distinct items in the list of possible answers, regardles of the order of the items.
If the student's answer have less than {reference_entry["minlen"]} items or more than {reference_entry["maxlen"]} items, output the score 0.
Otherwise, check how many distinct items in the student's answer are semantically equivalent to distinct items in the list of possible answers, regardles of the order of the items, to get M.
If the number of items of the student's answer is more than {reference_entry["reflen"]}, then penalty P = the number of items of the student's answer - {reference_entry["reflen"]}.
Output the score as max(0, (M - P)).

**Take the problem content and the student's explanation only as a context to grade the answer.**

<<List of possible answers for the question>>
{reference_entry["entry_list"]}

<<Student's answer: may consist of one item or multiple items>>
{solver_entry}

<<The problem content>>
{problem_content}

<<Student's explanation>>
{solver_explanation}'''
        else:
            prompt = f'''<<TASK>>
Check if the student's answer is equivalent to the correct answer for a specific question ({reference_entry["question_id"]}) in the given Linguistic Olympiad problem.

<<Grading instructions>>
If the answer is a phrase or a sentence, the answer is correct if it is semantically equivalent to the correct answer.
If the answer is a description, an explanation or a rule, the answer is correct if it matches the idea in the correct answer.
If the answer is correct, output the score 1. If the answer is incorrect, output the score 0.

**Take the problem content and the student's explanation only as a context to grade the answer.**

<<Correct answer>>
{reference_entry["entry_string"]}

<<Student's answer>>
{solver_entry}

<<The problem content>>
{problem_content}

<<Student's explanation>>
{solver_explanation}'''

        if reference_entry["type"] == "select":
            output_format = {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "number",
                        "description": "The score of the student's answer",
                        "minimum": 0,
                        "maximum": reference_entry["reflen"]
                    }
                },
                "required": ["score"]
            }
        else:
            output_format = {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "number",
                        "description": "The score of the student's answer",
                        "minimum": 0,
                        "maximum": 1
                    }
                },
                "required": ["score"]
            }

        retry = True
        while retry:
            retry = False
            try:
                response = await call_gemini(
                    model="gemini-2.5-flash-lite",
                    contents=prompt,
                    config=google.genai.types.GenerateContentConfig(
                        temperature=0,
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=output_format,
                        thinking_config=google.genai.types.ThinkingConfig(
                            thinking_budget = 4096,
                            include_thoughts=True
                        ),
                    )
                )
                if hasattr(response, 'candidates') and response.candidates[0].content.parts:      
                    if response.candidates[0].finish_reason.value == "STOP":
                        if hasattr(response, 'parsed') and response.parsed:
                            if (score := response.parsed.get('score')) is not None:
                                if reference_entry["type"] == "select":
                                    return score / reference_entry["reflen"]
                                else:
                                    return score
                            else:
                                raise ValueError("Invalid response format")
                        else:
                            raise ValueError("Invalid response format")
                    else:
                        raise ValueError("Token limit exceeded")
                else:
                    raise Exception("No response from the model")
            except Exception as e:
                retry = True
        return score
    return 0
async def fuzzy_embeddings(solver_entry, reference_entry):
    raise NotImplementedError("Fuzzy embeddings not implemented")
async def fuzzy_exact(solver_entry, reference_entry):
    raise NotImplementedError("Fuzzy exact not implemented")
# implement custom metrics here
# ...

async def normalize_entry(entry: str):

    entry_normalized = entry
    for replace_str in ['sg', 'du', 'pl']:
        entry_normalized = entry_normalized.replace(f'_{{{replace_str}}}', f'({replace_str})' )
        entry_normalized = entry_normalized.replace(f' ({replace_str})', f'({replace_str})')

    for remove_str in ['[', ']', '**', '_']: ## Removing '[]' is for phonology problems, e.g., 2018-1
        entry_normalized = entry_normalized.replace(remove_str, '')
    
    if entry == '':
        return ''

    if entry_normalized[-1] == '.':
        entry_normalized = entry_normalized[:-1]
    if len(entry_normalized) == 1 and entry_normalized.isupper():
        entry_normalized = entry_normalized.lower()
        
    return entry_normalized

async def match_exact(solver_entry, reference_entry):
    if isinstance(solver_entry, str):
        # the entry should be a string for exact match
        solver_entry_normalized = await normalize_entry(solver_entry)
        return 1 if solver_entry_normalized == reference_entry["entry_string"] else 0
    else:
        return 0

async def match_select(solver_entry, reference_entry):
    solver_entry_list = []
    if isinstance(solver_entry, str):
        solver_entry_list = [solver_entry]
    elif isinstance(solver_entry, list):
        solver_entry_list = solver_entry
    else:
        return 0

    solver_set = set([await normalize_entry(entry) for entry in solver_entry_list])
    reference_set = set([await normalize_entry(entry) for entry in reference_entry["entry_list"]])

    if len(solver_set) > reference_entry["maxlen"] or len(solver_set) < reference_entry["minlen"]:
        score = 0
    else:
        correct_items = min(len(solver_set.intersection(reference_set)), reference_entry["reflen"])
        penalty = len(solver_set) - reference_entry["reflen"]
        correct_items = max(0, correct_items - penalty)
        score = correct_items / reference_entry["reflen"]
    return score

async def match_fuzzy(solver_entry, reference_entry, problem_content, solver_explanation, fuzzy_metric: str):
    if isinstance(solver_entry, str):
        # the entry should be a string for fuzzy match
        # no need to normalize for fuzzy match
        if fuzzy_metric == "llm":
            return await fuzzy_llm(solver_entry, reference_entry, problem_content, solver_explanation)
        elif fuzzy_metric == "embeddings":
            return await fuzzy_embeddings(solver_entry, reference_entry)
        elif fuzzy_metric == "exact":
            return await fuzzy_exact(solver_entry, reference_entry)
        else:
            raise ValueError(f"Fuzzy metric `{fuzzy_metric}` not implemented!")
    else:
        return 0

async def judge_entry(solver_entry, reference_entry, problem_content, solver_explanation, fuzzy_metric: str):
    if reference_entry["fuzzy"]: # fuzzy match
        if reference_entry["type"] == "select": # fuzzy select: convert list to string, and match as fuzzy
            if isinstance(solver_entry, list):
                solver_entry = ", ".join([str(i) for i in solver_entry])
            reference_entry_string = {"entry_string": ", ".join(reference_entry["entry_list"])}
            reference_entry["entry_string"] = reference_entry_string
            return await match_fuzzy(solver_entry, reference_entry, problem_content, solver_explanation, fuzzy_metric)
        else:
            return await match_fuzzy(solver_entry, reference_entry, problem_content, solver_explanation, fuzzy_metric)
    elif reference_entry["type"] == "select": # select match
        return await match_select(solver_entry, reference_entry)
    else: # exact match
        return await match_exact(solver_entry, reference_entry)

def parse_reference_answer(reference_answer: dict):
    parsed = {}
    for question, entry in reference_answer.items():
        fuzzy = False
        if entry.startswith('<fuzzy>'):
            fuzzy = True
            entry = entry[7:] # strip away the `<fuzzy>`

        if entry.startswith('<select'):
            tag, entry_list = entry.split('>', 1)
            reflen, minlen, maxlen = ((int(k) if 'inf' != (k:=x.strip()) else 20) for x in tag[7:].split(','))
            entry_list = json.loads(entry_list.replace("'", '"'))
            parsed[question] = {
                "type": "select",
                "entry_list": entry_list,
                "reflen": reflen,
                "minlen": minlen,
                "maxlen": maxlen,
                "fuzzy": fuzzy
            }
        else:
            parsed[question] = {
                "type": "string",
                "entry_string": entry,
                "fuzzy": fuzzy
            }
    return parsed

async def judge_answer(solver_solution: dict, reference_solution: dict, fuzzy_metric: str):
    scores = {}
    # judge all entries for one problem
    solver_answer = solver_solution["answer"]
    reference_answer = parse_reference_answer(reference_solution["answer"])
    problem_content = reference_solution["problem_content"]
    solver_explanation = solver_solution.get("explanation", "")
    for question, reference_entry in reference_answer.items():
        if (solver_entry := solver_answer.get(question)) is None:
            scores[question] = 0
        else:
            reference_entry["question_id"] = question
            scores[question] = await judge_entry(solver_entry, reference_entry, problem_content, solver_explanation, fuzzy_metric)
    return scores

async def judge_rule_point(solver_answer: dict, solver_explanation: dict, rules: dict, rule_point: str, problem_content: str):

    retry = True
    while retry:
        system_instruction = '''You are a grader for Linguistic Olympiad problems. Your job is to grade a student's submission for a problem. Do not solve the problem yourself.'''
        
        prompt = f'''<<TASK>>
Grade the "explanation" part of a student's solution to a Linguistic Olympiad problem, for the given criterion.

<<Grading instructions>>
A solution consists of an "answer" and an "explanation". The "explanation" is the linguistic rules or patterns deduced from the problem.
Output a score between 0 and 1, according to how much the student's explanation matches the reference explanation, just for the given criterion.

**Take the problem content and the student's "answer" only as a context to grade the "explanation".**

<<Student's explanation>>
{solver_explanation}

<<Reference explanation>>
{rules}

<<Criterion>>
{rule_point}

<<The problem content>>
{problem_content}

<<Student's answer>>
{solver_answer}'''
        output_format = {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "description": "The score of the student's explanation",
                    "minimum": 0,
                    "maximum": 1
                }
            },
            "required": ["score"]
        }
        retry = True
        model_upgrade_for_retry = False
        while retry:
            retry = False
            try:
                response = await call_gemini(
                    model="gemini-2.5-flash-lite" if not model_upgrade_for_retry else "gemini-2.5-pro",
                    contents=prompt,
                    config=google.genai.types.GenerateContentConfig(
                        temperature=0,
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=output_format,
                        thinking_config=google.genai.types.ThinkingConfig(
                            thinking_budget = 4096,
                            include_thoughts=True
                        ),
                    )
                )

                if hasattr(response, 'candidates') and response.candidates[0].content.parts:      
                    if response.candidates[0].finish_reason.value == "STOP":
                        if hasattr(response, 'parsed') and response.parsed:
                            if (score := response.parsed.get('score')) is not None:
                                return score
                            else:
                                raise ValueError("Invalid response format")
                        else:
                            raise ValueError("Invalid response format")
                    else:
                        raise ValueError("Token limit exceeded")
                else:
                    raise Exception("No response from the model")
            except Exception as e:
                if (e.code == 500):
                    if model_upgrade_for_retry:
                        raise Exception("Gemini internal server error")
                    print("flash-lite responds with 500 error, grading with gemini-2.5-pro")
                    model_upgrade_for_retry = True
                retry = True
        return response["score"]

async def judge_explanation(solver_solution: dict, reference_solution: dict):
    scores = []
    problem_content = reference_solution["problem_content"]
    solver_answer = solver_solution.get("answer", {})
    solver_explanation = solver_solution.get("explanation", "")
    for rule_point in reference_solution["explanation"]["rule_checklist"]:
        scores.append({
            "rule_point": rule_point,
            "score": await judge_rule_point(solver_answer, solver_explanation, reference_solution["explanation"]["rules"], rule_point, problem_content)
        })
    return scores

def check_solution_format(solver_solution: dict):
    if "answer" not in solver_solution:
        return False
    if "explanation" not in solver_solution:
        return False
    if not isinstance(solver_solution["explanation"], str):
        return False
    return True

async def evaluate_solution(problem_id, solver_solution, reference_solution, args):
    # check if the format of the solver solution is valid
    if not check_solution_format(solver_solution):
        if args.invalid_format_fallback == "zero":
            pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(args.output, f"{problem_id}_scores.json"), "w", encoding="utf-8") as f:
                json.dump({"answer_avg_score": 0, "explanation_avg_score": 0, "solution_avg_score": 0, "answer_scores": {}, "explanation_scores": {}}, f, indent=2, ensure_ascii=False)
            return 0
        else:
            return "invalid solution format"
    else:

        answer_avg_score = 0
        explanation_avg_score = 0
        if len(reference_solution["explanation"]["rule_checklist"]) > 0 and args.explanation_weight != 0: # judge explanation
            explanation_scores = await judge_explanation(solver_solution, reference_solution)
            explanation_avg_score = sum([score["score"] for score in explanation_scores])/len(explanation_scores)
        if args.explanation_weight != 1: # judge answer
            answer_scores = await judge_answer(solver_solution, reference_solution, args.fuzzy_metric)
            answer_avg_score = sum(answer_scores.values())/len(answer_scores.values())
        
        if len(reference_solution["explanation"]["rule_checklist"]) > 0:
            solution_avg_score = answer_avg_score * (1-args.explanation_weight) + explanation_avg_score * args.explanation_weight
        else:
            solution_avg_score = answer_avg_score

        solution_score_details = {}
        if len(reference_solution["explanation"]["rule_checklist"]) > 0 and args.explanation_weight != 0:
            solution_score_details["explanation_scores"] = explanation_scores
            solution_score_details["explanation_avg_score"] = explanation_avg_score
        if args.explanation_weight != 1:
            solution_score_details["answer_scores"] = answer_scores
            solution_score_details["answer_avg_score"] = answer_avg_score
        solution_score_details["solution_avg_score"] = solution_avg_score

        # save the solution score details to a json file
        pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(args.output, f"{problem_id}_scores.json"), "w", encoding="utf-8") as f:
            json.dump(solution_score_details, f, indent=2, ensure_ascii=False)
        
        return solution_avg_score
    

async def evaluate_solutions(solver_solutions: dict, reference_solutions: dict, args):
    def render_status(evaluation_status):
        return f"""[yellow]Remaining Problems:[/yellow] {" ".join(evaluation_status["remaining_problems"])}\n""" + \
            f"""[green]Working Problems:[/green] {" ".join(evaluation_status["working_problems"])}\n""" + \
            f"""[blue]Completed Problems:[/blue] {" ".join(evaluation_status["completed_problems"])}\n""" + \
            f"""[red]Error Problems:[/red] {" ".join(evaluation_status["error_problems"])}\n""" + \
            f"""Time Elapsed: {time.strftime('%M:%S', time.gmtime(time.time() - evaluation_status["start_time"]))}"""

    async def work_on_problem(problem_id, evaluation_status, live):
        evaluation_status["remaining_problems"] = [p for p in evaluation_status["remaining_problems"] if p != problem_id]
        evaluation_status["working_problems"].append(problem_id)
        live.update(render_status(evaluation_status))
    async def complete_problem(problem_id, evaluation_status, live):
        evaluation_status["working_problems"] = [p for p in evaluation_status["working_problems"] if p != problem_id]
        evaluation_status["completed_problems"].append(problem_id)
        live.update(render_status(evaluation_status))
    async def error_problem(problem_id, evaluation_status, error_message, live):
        evaluation_status["working_problems"] = [p for p in evaluation_status["working_problems"] if p != problem_id]
        evaluation_status["error_problems"].append(f"{problem_id}({error_message})")
        live.update(render_status(evaluation_status))

    async def evaluation_task(problem_id, solver_solution, reference_solution, evaluation_status, live, args):
        async with semaphore:
            await work_on_problem(problem_id, evaluation_status, live)
            try:
                solution_score_detail = await evaluate_solution(problem_id, solver_solution, reference_solution, args)
            except Exception as e:
                await error_problem(problem_id, evaluation_status, str(e), live)
                return str(e)
            await complete_problem(problem_id, evaluation_status, live)
            return solution_score_detail

    async def timer_task(evaluation_status, live):
        while True:
            await asyncio.sleep(1)
            live.update(render_status(evaluation_status))
            if len(evaluation_status["remaining_problems"]) == 0 and len(evaluation_status["working_problems"]) == 0:
                live.stop()
                break
        return None
    tasks = []
    semaphore = asyncio.Semaphore(args.max_concurrency)
    evaluation_status = {
        "remaining_problems": list(solver_solutions.keys()),
        "working_problems": [],
        "completed_problems": [],
        "error_problems": [],
        "start_time": time.time()
    }
    with Live(render_status(evaluation_status), refresh_per_second=4) as live:
        for problem_id in solver_solutions.keys():
            solver_solution = solver_solutions[problem_id]
            reference_solution = reference_solutions[problem_id]
            tasks.append(evaluation_task(problem_id, solver_solution, reference_solution, evaluation_status, live, args))
        tasks.append(timer_task(evaluation_status, live))
        solution_scores = await asyncio.gather(*tasks)
    return solution_scores[:-1]

def main(args):
    #load all reference solutions
    reference_solutions = {}
    for file in glob.glob(os.path.join(args.reference_solution, "*.json")):
        with open(file, "r", encoding="utf-8") as f:
            reference_solution = json.load(f)
            reference_solutions[reference_solution["id"]] = reference_solution
    
    #Load all problem contents into reference solutions
    for file in glob.glob(os.path.join(args.problem, "*.json")):
        with open(file, "r", encoding="utf-8") as f:
            problem = json.load(f)
            reference_solutions[problem["id"]]["problem_content"] = problem["content"]

    #Load all solver solutions
    solver_solutions = {}
    for file in glob.glob(os.path.join(args.solver_solution, "*.json")):
        with open(file, "r", encoding="utf-8") as f:
            solver_solution = json.load(f)
            if not args.overwrite and os.path.exists(os.path.join(args.output, f"{solver_solution['id']}_scores.json")):
                continue
            if args.select_years and solver_solution["id"].split("-")[0] not in args.select_years:
                continue
            if args.select_problem_numbers and solver_solution["id"].split("-")[1] not in args.select_problem_numbers:
                continue
            if args.select_problem_ids and solver_solution["id"] not in args.select_problem_ids:
                continue
            if solver_solution["id"] in reference_solutions.keys():
                solver_solutions[solver_solution["id"]] = solver_solution
    if solver_solutions == {}:
        print("No solver solutions to evaluate")
        return
    if not args.yes:
        if "y" != input(f"Evaluating on {len(solver_solutions)} solver solutions: {" ".join(solver_solutions.keys())}\nContinue? (y/n): "):
            return

    # run evaluations in parallel
    solution_scores = asyncio.run(evaluate_solutions(solver_solutions, reference_solutions, args))
    real_solution_scores = [score for score in solution_scores if isinstance(score, float)]
    print(f"Evaluations completed with {len(real_solution_scores)} successful evaluations and {len(solution_scores) - len(real_solution_scores)} error evaluations")
    print(f"Average score: {sum(real_solution_scores)/len(real_solution_scores)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation script for the LOBSTER benchmark')
    parser.add_argument('--problem', default="data/problem", help='Path to folder containing problem specifications')
    parser.add_argument('--solver_solution', default="output/solver", help='Path to folder containing solver solutions')
    parser.add_argument('--reference_solution', default="data/solution", help='Path to folder containing reference solutions')
    parser.add_argument('--output', default="output/eval", help='Path to save evaluation results')
    # Customizations
    parser.add_argument('--fuzzy_metric', default="llm", help='Metric to use for non-exact matches', choices=["llm", "embeddings", "exact"])
    parser.add_argument('--explanation_weight', type=float, default=0.5, help='Weight for the explanation score')
    parser.add_argument('--invalid_format_fallback', default="skip", help='What to do if the format of the solver solution is invalid', choices=["skip", "zero"])
    parser.add_argument('--max_concurrency', default=96, type=int, help='Maximum number of concurrent evaluations')
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite the evaluation results')
    parser.add_argument('--select_years', nargs='+', default=None, help='Select years to evaluate')
    parser.add_argument('--select_problem_numbers', nargs='+', default=None, help='Select problem numbers to evaluate')
    parser.add_argument('--select_problem_ids', nargs='+', default=None, help='Select problem IDs to evaluate')
    parser.add_argument('--yes', action='store_true', help='Whether to skip the confirmation prompt')
    # For evaluating multiple results with one command
    parser.add_argument('--settings', nargs='+', default=None, help='Settings to evaluate')
    parser.add_argument('--run_numbers', nargs='+', default=None, type=int, help='Run numbers to evaluate')
    args = parser.parse_args()
    if args.run_numbers:
        solver_solution_base = args.solver_solution
        output_base = args.output
        for setting in args.settings:
            for run_number in args.run_numbers:
                args.solver_solution = os.path.join(solver_solution_base, setting, f"run_{run_number}")
                args.output = os.path.join(output_base, setting, f"run_{run_number}")
                print(f"Evaluating setting {setting} with run number {run_number}")
                main(args)
    else:
        main(args)