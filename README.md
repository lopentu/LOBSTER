# LOBSTER🦞 : Linguistics Olympiad Benchmark for Structured Evaluation on Reasoning

## Dataset

### 📝 Problems

Problem texts and answer schemas, stored in 96 `.json` files: `data/problem/{year)-{problem_number}_problem.json`

See [**here**](documents/problem_details.md) for details about the format of problems.

### 💯 Gold-standard Reference Solutions

Gold-standard reference solutions for evaluation, stored in 96 `.json` files: `data/solution/{year)-{problem_number}_reference_solution.json`

See [**here**](documents/reference_solution_details.md) for details about the format of reference solutions.

### 📖 Problem Metadata

Typological information of all problems in the benchmark: `data/problem_metadata.json`

### 🤔 Reasoning Traces

Gold-standard reasoning traces for each problem: `data/reasoning_traces.json`

## ⚙️ Installation

Install packages: `pip install -r requirements.txt`
Create `.env` and add the variables `GEMINI_API_KEY` and `OPEN_API_KEY`

## Run Evaluation

### 🚀 To run the evaluation script
```bash
python -m src.eval.evaluate
```

The script will look for the solver's solution files in the directory specified by `--solver_solution`. (`output/solver` by default)

For ease of evaluating multiple setting and run numbers (see [*Run Solvers*](#Run-Solvers)), arguments `--settings` and `--run numbers` can be added, and the script will look for the solver's solution in `output/solver/{setting}/run_{run_number}`.

### Output

The average scores and detailed scores will be output in `output/evaluation` as `{id}_scores.json`.

If `--settings` and `--run numbers` are specified, then the scores will be output in `output/evaluation/{setting}/run_{run_number}`

If the output directory already has the score file, the script will skip the problem. Unless the `--overwrite` flag is added, then the existing score file will be overwritten.

### Concurrency

In default, all 96 problems are evaluated concurrently. The number of maximum allowed concurrent evaluations can be set if needed.

### Evaluate on a subset of problems

In default all 96 problems will be evaluated. Use the arguments `--select_years`, `--select_problem_numbers`, and `--select_problem_ids` to select spicific problems to evaluate.

## Solution Format

For a solution to be evaluated successfully, it must be a `.json` file with the following items:

### ID

`"id"`: The problem ID, in the format of `"{year}-{problem_number}"`

### Answer


`"answer"`: An object containing the "answer" part of the the model's solution. The format is determined by the schema `answer_format` in `data/problem/{id}_problem.json`

### Explanation

`"explanation"`: a string containing the "explanation" part of the model's solution.

### Example
```json
{
    "id": "2025-6",
    "answer": {
        {
            "a_translation": "Hello Word!",
            "b_words": ["foo", "bar"],
        },
    "explanation": "This is an explanation."
    }
}
```

It is crucial for the model to output the solution in the specified format. In particular, the `answer` is judged by getting the value of each key and matching it with the value in the reference answer, therefore, correct key-value pairs are essential for a high-scoring `answer`.

## Run Solvers

### 🧪 To reproduce the experiments in the paper
```bash
python -m src.solver.run_solvers --setting ${SETTING}
```
### Available settings
```python
["baseline_gemini", "baseline_openai", "gpt-5", "guided_gemini", "guided_openai",
"grammar_gemini", "single_gemini", "single_openai", "moa"]
```

### Output

The solutions output by solvers will be found in `output/solver/{setting}`.

To run multiple times with the same setting, add the `--run_numbers` argument, for example:
```bash
python -m src.solver.run_solvers --setting gpt-5 --run_numbers 1 2 3 4 5
```

And the solution will be output in `output/solver/{setting}/run_{run_number}`

If the output directory already has the solution file, the script will skip the problem. Unless the `--overwrite` flag is added, then the existing solution file will be overwritten.

### Concurrency

In default, all 96 problems are solved concurrently. The number of maximum allowed concurrent solvers can be set if needed.

### Run on a subset of problems

In default all 96 problems will be solved. Use the arguments `--select_years`, `--select_problem_numbers`, and `--select_problem_ids` to select spicific problems to solve.