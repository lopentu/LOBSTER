import argparse
import os
import glob
import json
import asyncio
import pathlib
from tqdm.asyncio import tqdm
from rich.live import Live
import time
from dotenv import load_dotenv
import backoff
from google.api_core import retry_async
from rich.logging import RichHandler

import google.genai
import google.genai.types
import google.genai.errors
import openai
import fastmcp

# initialize gemini client
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")

gemini_client = google.genai.Client(api_key=api_key)

# initialize openai client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")
openai_client = openai.AsyncOpenAI(api_key=api_key)

grammar_agent_implemented = False

if grammar_agent_implemented:
    mcp_client = fastmcp.Client("")

def is_retryable(e) -> bool:
    return True

@retry_async.AsyncRetry(predicate=is_retryable)
async def call_gemini(**kwargs):
    return await gemini_client.aio.models.generate_content(**kwargs)

@backoff.on_exception(backoff.expo, openai.RateLimitError)
async def call_openai(**kwargs):
    return await openai_client.responses.create(**kwargs)

async def get_guide_book():
    with open("src/solver/guide_book/01-IntroToLO.tex", "r", encoding="utf-8") as f:
        guide_book = f.read()
        return guide_book

async def run_solver(problem_id: str, problem: dict, model: str, previous_solutions: list[dict] = None, guided = False, grammar_info: str = None):

    prompt = f"""Please solve the following problem and provide your final answer and your explanation of linguistic rules or patterns deduced from the problem.

In each field of the final answer, you only need to write down the unknown part without copying the question. If the problem is a matching problem, please write down the labels only (for example, write down a, b, c, 1, 2, 3... instead of the full text).

{problem["content"]}"""

    system_instruction = f"""You are an agent that solves Linguistic Olympiad problems."""
    
    if guided:
        system_instruction += f" For your reference, here's an introduction to linguistics puzzles in TeX format. Please read it as a guide in solving Linguistic Olympiad problems. \n{await get_guide_book()}"

    if grammar_info:
        prompt += f"""

Here's some relevant information queried and summarized from reference grammar books. Take it as a reference for solving the problem.
            
{grammar_info}"""

    if previous_solutions:
        prev_solutions_text = ""
        for i, previous_solution in enumerate(previous_solutions):
            prev_solutions_text = f"""{prev_solutions_text}
            
Solution {i+1}'s answer:
{previous_solution["answer"]}
Solution {i+1}'s explanation:
{previous_solution["explanation"]}"""

        prompt += f"""

Below are {len(previous_solutions)} solutions written by different agents. Please read the solutions, gather insights from them, and provide a new solution:{prev_solutions_text}"""

    output_schema = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "object",
                "description": "Your final answer to the problem.",
                "properties": problem["answer_format"],
            },
            "explanation": {
                "type": "string",
                "description": "Your explanation of the linguistic rules or patterns deduced from the problem."
            },
        },
        "required": ["answer", "explanation"],
    }

    start_time = time.time()

    if model == "gemini":
        retry = True
        while retry: # retry on invalid response format
            retry = False
            try:
                response = await call_gemini(
                    model="gemini-2.5-pro",
                    contents=prompt,
                    config=google.genai.types.GenerateContentConfig(
                        temperature = 0.75,
                        system_instruction = system_instruction,
                        response_mime_type = 'application/json',
                        response_schema = output_schema, 
                        thinking_config=google.genai.types.ThinkingConfig(
                            thinking_budget = -1,
                            include_thoughts=True
                        ),
                    )
                )

                response_time = time.time() - start_time
                if hasattr(response, 'candidates') and response.candidates[0].content.parts:      
                    thought_summary = '\n\n'.join( part.text for part in response.candidates[0].content.parts if part.text and part.thought )
                    raw_text = '\n\n'.join( part.text for part in response.candidates[0].content.parts if part.text and not part.thought )    
                    if response.candidates[0].finish_reason.value == "STOP":
                        if hasattr(response, 'parsed') and response.parsed:
                            parsed = response.parsed
                            if (answer := parsed.get('answer')) and (explanation := parsed.get('explanation')):
                                result = {
                                    "id": problem_id,
                                    "answer": answer,
                                    "explanation": explanation,
                                    "thought_summary": thought_summary,
                                    "raw_text": raw_text,
                                    "input_tokens": response.usage_metadata.prompt_token_count or 0,
                                    "thoughts_tokens": response.usage_metadata.thoughts_token_count or 0,
                                    "output_tokens": response.usage_metadata.candidates_token_count or 0,
                                    "model": "gemini-2.5-pro",
                                    "response_time": response_time
                                }
                                if grammar_info:
                                    result["grammar_info"] = grammar_info
                                return result
                            else:
                                raise ValueError("Invalid response format")
                        else:
                            raise ValueError("Invalid response format")
                    else:
                        raise ValueError("Token limit exceeded")
                else:
                    raise Exception("No response from the model")
            except ValueError:
                retry = True
    else:
        retry = True
        while retry: # retry on invalid response format
            retry = False
            try:
                response = await call_openai(
                    instructions = system_instruction,
                    input = prompt,
                    model = "o4-mini" if model == "openai" else "gpt-5",
                    truncation = 'disabled',
                    text = {
                        'format': {
                            'type': 'json_schema',
                            'name': 'solution',
                            'strict': False,
                            'schema': output_schema,
                        }
                    },
                )
                response_time = time.time() - start_time
                if response.error or response.status in ('failed', 'cancelled'):
                    raise Exception(f"Error calling OpenAI API: {response.error}")
                else:
                    try:
                        parsed = json.loads(response.output_text)
                        if (answer := parsed.get('answer')) and (explanation := parsed.get('explanation')):
                            return {
                                "id": problem_id,
                                "answer": answer,
                                "explanation": explanation,
                                "thought_summary": "N/A",
                                "raw_text": response.output_text,
                                "input_tokens": response.usage.input_tokens,
                                "thoughts_tokens": response.usage.output_tokens_details.reasoning_tokens,
                                "output_tokens": response.usage.output_tokens - response.usage.output_tokens_details.reasoning_tokens,
                                "model": "o4-mini" if model == "openai" else "gpt-5",
                                "response_time": response_time
                            }
                        else:
                            raise ValueError("Invalid response format")
                    except json.JSONDecodeError:
                        raise ValueError("Invalid response format")
            except ValueError as e:
                print(f"Problem {problem_id} error: {e}")
                retry = True

async def run_grammar_agent(problem):

    async def call_gemini_with_mcp(prompt, system_instruction):
        async with mcp_client:
            response = await call_gemini(
                model="gemini-2.5-pro",
                contents=prompt,
                config=google.genai.types.GenerateContentConfig(
                    temperature = 0.75,
                    system_instruction = system_instruction,
                    tools=[mcp_client.session],
                    thinking_config=google.genai.types.ThinkingConfig(
                        thinking_budget = -1,
                        include_thoughts=True
                    ),
                )
            )
        return response
    prompt = f"""
You are given a Linguistic Olympiad problem. Please call the tools to query the reference grammar books for information that is potentially useful in solving the problem, and summarize the relevant information for the solver agent. Search for related languages or similar grammatical features if you cannot find information about the exact language.

If you cannot find any relevant information, please return 'No relevant information found in the reference grammar books'.

{problem}"""

    system_instruction = f"""
You are an agent that calls tools to query for relevant information and summarize it to help the solver agent solve Linguistic Olympiad problems."""

    retry = True
    while retry:
        retry = False
        try:
            response = await call_gemini_with_mcp(prompt, system_instruction)
            return '\n\n'.join( part.text for part in response.candidates[0].content.parts if part.text and not part.thought )
        except Exception as e:
            retry = True

async def run_baseline(problems, model, args):
    def render_status(solver_status):
        return f"""[yellow]Remaining Problems:[/yellow] {" ".join(solver_status["remaining_problems"])}\n""" + \
            f"""[green]Working Problems:[/green] {" ".join(solver_status["working_problems"])}\n""" + \
            f"""[blue]Completed Problems:[/blue] {" ".join(solver_status["completed_problems"])}\n""" + \
            f"""[red]Error Problems:[/red] {" ".join(solver_status["error_problems"])}\n""" + \
            f"""Time Elapsed: {time.strftime('%M:%S', time.gmtime(time.time() - solver_status["start_time"]))}"""

    def work_on_problem(problem_id, solver_status, live):
        solver_status["remaining_problems"] = [p for p in solver_status["remaining_problems"] if p != problem_id]
        solver_status["working_problems"].append(problem_id)
        live.update(render_status(solver_status))
        return
    def complete_problem(problem_id, solver_status, live):
        solver_status["working_problems"] = [p for p in solver_status["working_problems"] if p != problem_id]
        solver_status["completed_problems"].append(problem_id)
        live.update(render_status(solver_status))
        return
    def error_problem(problem_id, solver_status, error_message, live):
        solver_status["working_problems"] = [p for p in solver_status["working_problems"] if p != problem_id]
        solver_status["error_problems"].append(f"{problem_id} ({error_message})")
        live.update(render_status(solver_status))
        return

    async def solver_task(problem_id, problem, model, solver_status, live):
        async with semaphore:
            try:
                work_on_problem(problem_id, solver_status, live)
                if args.setting.startswith("grammar_"):
                    grammar_info = await run_grammar_agent(problem)
                else:
                    grammar_info = None
                solver_solution = await run_solver(problem_id, problem, model, guided = args.setting.startswith("guided_"), grammar_info = grammar_info)
                save_path = os.path.join(args.output, args.setting, f"run_{args.run_number}" if args.run_number else "")
                pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
                with open(os.path.join(save_path, f"{problem_id}_solution.json"), "w", encoding="utf-8") as f:
                    json.dump(solver_solution, f, indent=2, ensure_ascii=False)
            except Exception as e:
                error_problem(problem_id, solver_status, str(e), live)
                return str(e)
            complete_problem(problem_id, solver_status, live)
            return "success"

    async def timer_task(solver_status, live):
        while True:
            await asyncio.sleep(1)
            live.update(render_status(solver_status))
            if len(solver_status["remaining_problems"]) == 0 and len(solver_status["working_problems"]) == 0:
                live.stop()
                break
        return None

    tasks = []
    semaphore = asyncio.Semaphore(args.max_concurrency)
    solver_status = {
        "remaining_problems": list(problems.keys()),
        "working_problems": [],
        "completed_problems": [],
        "error_problems": [],
        "start_time": time.time()
    }
    with Live(render_status(solver_status), refresh_per_second=4) as live:
        for problem_id in problems.keys():
            problem = problems[problem_id]
            tasks.append(solver_task(problem_id, problem, model, solver_status, live))
        tasks.append(timer_task(solver_status, live))
        task_statuses = await asyncio.gather(*tasks)
    return task_statuses[:-1]

async def run_mixture_of_agents(problems, model_list, rounds, args):
    def render_status(solver_status):
        return f"""[yellow]Remaining Problems:[/yellow] {" ".join(solver_status["remaining_problems"])}\n""" + \
            "".join([f"""[green]Round {i+1}:[/green] {" ".join(solver_status[f"working_problems_{i+1}"])}\n""" for i in range(rounds)]) + \
            f"""[blue]Completed Problems:[/blue] {" ".join(solver_status["completed_problems"])}\n""" + \
            f"""[red]Error Problems:[/red] {" ".join(solver_status["error_problems"])}\n""" + \
            f"""Time Elapsed: {time.strftime('%M:%S', time.gmtime(time.time() - solver_status["start_time"]))}"""

    async def work_on_problem(problem_id, solver_status, round, live):
        if round == 1:
            solver_status["remaining_problems"] = [p for p in solver_status["remaining_problems"] if p != problem_id]
            solver_status["working_problems_1"].append(problem_id)
            live.update(render_status(solver_status))
        else:
            solver_status[f"working_problems_{round-1}"] = [p for p in solver_status[f"working_problems_{round-1}"] if p != problem_id]
            solver_status[f"working_problems_{round}"].append(problem_id)
            live.update(render_status(solver_status))
        return
    async def complete_problem(problem_id, solver_status, rounds, live):
        solver_status[f"working_problems_{rounds}"] = [p for p in solver_status[f"working_problems_{rounds}"] if p != problem_id]
        solver_status["completed_problems"].append(f"{problem_id}")
        live.update(render_status(solver_status))
        return
    async def error_problem(problem_id, solver_status, rounds, error_message, live):
        for round in range(1, rounds+1):
            solver_status[f"working_problems_{round}"] = [p for p in solver_status[f"working_problems_{round}"] if p != problem_id]
        solver_status["error_problems"].append(f"{problem_id} ({error_message})")
        live.update(render_status(solver_status))
        return

    async def get_existing_solution(problem_id, model, round):
        save_path = os.path.join(args.output, f"{'single' if len(model_list) == 1 else 'moa'}_{model}_{round}", f"run_{args.run_number}" if args.run_number else "")
        if not os.path.exists(os.path.join(save_path, f"{problem_id}_solution.json")):
            return None
        with open(os.path.join(save_path, f"{problem_id}_solution.json"), "r", encoding="utf-8") as f:
            return json.load(f)

    async def solver_task(problem_id, problem, model_list, rounds, solver_status, live):
        async with semaphore:
            try:
                previous_solutions = []
                for round in range(1, rounds+1):
                    await work_on_problem(problem_id, solver_status, round, live)
                    if args.overwrite:
                        solver_solutions = [None for _ in model_list]
                    else:
                        solver_solutions = [await get_existing_solution(problem_id, model, round) for model in model_list]
                    if all(solver_solutions):
                        previous_solutions.append(solver_solutions)
                        continue
                    if round == 1:
                        solver_solutions = await asyncio.gather(
                            *[run_solver(problem_id, problem, model) for model in model_list]
                        )
                        previous_solutions.append(solver_solutions)
                    else:
                        solver_solutions = await asyncio.gather(
                            *[run_solver(problem_id, problem, model, previous_solutions[-1]) for model in model_list]
                        )
                        previous_solutions.append(solver_solutions)
                    for i in range(len(model_list)):
                        save_path = os.path.join(args.output, f"{'single' if len(model_list) == 1 else 'moa'}_{model_list[i]}_{round}", f"run_{args.run_number}" if args.run_number else "")
                        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
                        with open(os.path.join(save_path, f"{problem_id}_solution.json"), "w", encoding="utf-8") as f:
                            json.dump(solver_solutions[i], f, indent=2, ensure_ascii=False)
            except Exception as e:
                await error_problem(problem_id, solver_status, rounds, str(e), live)
                return str(e)
            await complete_problem(problem_id, solver_status, rounds, live)
            return "success"

    async def timer_task(solver_status, live):
        while True:
            await asyncio.sleep(1)
            live.update(render_status(solver_status))
            if len(solver_status["remaining_problems"]) == 0 and all([len(solver_status[f"working_problems_{round}"]) == 0 for round in range(1, rounds+1)]):
                live.stop()
                break
        return None

    tasks = []
    semaphore = asyncio.Semaphore(args.max_concurrency)
    solver_status = {
        "remaining_problems": list(problems.keys()),
        "completed_problems": [],
        "error_problems": [],
        "start_time": time.time()
    }
    for round in range(1, rounds+1):
        solver_status[f"working_problems_{round}"] = []
    with Live(render_status(solver_status), refresh_per_second=4) as live:
        for problem_id in problems.keys():
            problem = problems[problem_id]
            tasks.append(solver_task(problem_id, problem, model_list, 6, solver_status, live))
        tasks.append(timer_task(solver_status, live))
        task_statuses = await asyncio.gather(*tasks)
    return task_statuses[:-1]

def main(args):
    available_settings = ["baseline_gemini", "baseline_openai", "gpt-5"
        "guided_gemini", "guided_openai",
        "grammar_gemini",
        "single_gemini", "single_openai",
        "moa"]
    if args.setting not in available_settings:
        print(f"Invalid setting: {args.setting}")
        return
    problems = {}
    for file in glob.glob(os.path.join(args.problem, "*.json")):
        with open(file, "r", encoding="utf-8") as f:
            problem = json.load(f)
            if not args.overwrite:
                if args.setting.startswith("baseline_") or args.setting.startswith("guided_") or args.setting.startswith("grammar_"):
                    save_path = os.path.join(args.output, args.setting, f"run_{args.run_number}" if args.run_number else "")
                    if os.path.exists(os.path.join(save_path, f"{problem['id']}_solution.json")):
                        continue
                # moa runs check overwrite for each round, implemented in run_mixture_of_agents
            if args.select_years and problem["id"].split("-")[0] not in args.select_years:
                continue
            if args.select_problem_numbers and problem["id"].split("-")[1] not in args.select_problem_numbers:
                continue
            if args.select_problem_ids and problem["id"] not in args.select_problem_ids:
                continue
            problems[problem["id"]] = problem
    
    if problems == {}:
        print("No problems to run on")
        return
    if not args.yes:
        if not "y" == input(f"Run setting {args.setting} on {len(problems)} problems: {', '.join(problems.keys())}\nContinue? (y/n): "):
            return
    
    if args.setting.startswith("baseline_") or args.setting.startswith("guided_") or args.setting.startswith("grammar_"):
        run_statuses = asyncio.run(run_baseline(problems, args.setting.split("_")[1], args))
    elif args.setting.startswith("single_"):
        run_statuses = asyncio.run(run_mixture_of_agents(problems, [args.setting.split("_")[1]], 6, args))
    elif args.setting == "moa":
        run_statuses = asyncio.run(run_mixture_of_agents(problems, ["gemini", "openai"], 6, args))
    print(f"Solver runs completed with {sum([1 for status in run_statuses if status == 'success'])} successful runs and {sum([1 for status in run_statuses if status != 'success'])} error runs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for running solvers on the LOBSTER benchmark')
    parser.add_argument('--problem', default="data/problem", help='Path to folder containing problem specifications')
    parser.add_argument('--select_years', nargs='+', default=None, help='Run on selected years')
    parser.add_argument('--select_problem_numbers', nargs='+', default=None, help='Run on selected problem numbers')
    parser.add_argument('--select_problem_ids', nargs='+', default=None, help='Run on selected problem IDs')
    parser.add_argument('--setting', required=True, help='solver setting to use')
    parser.add_argument('--run_number', default=None, type=int, help='Run number')
    parser.add_argument('--run_numbers', nargs='+', default=None, type=int, help='Run numbers')
    parser.add_argument('--output', default="output/solver", help='Path to save solver solutions')
    parser.add_argument('--max_concurrency', default=96, type=int, help='Maximum number of concurrent solver runs')
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite the solver solutions')
    parser.add_argument('--yes', action='store_true', help='Whether to skip the confirmation prompt')
    args = parser.parse_args()
    if args.run_numbers:
        for run_number in args.run_numbers:
            args.run_number = run_number
            print(f"Running with run number {args.run_number}")
            main(args)
    else:
        print(f"Running with run number {args.run_number}")
        main(args)