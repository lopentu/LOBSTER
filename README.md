# LOBSTER🦞 : Linguistics Olympiad Benchmark for Structured Evaluation on Reasoning

## Installation

Install packages: `pip install -r requirements.txt`
Create `.env` and add the variables `GEMINI_API_KEY` and `OPEN_API_KEY`

## Dataset

`data/problem/{year)-{problem_number}_problem.json`: problem texts and answer schemas
`data/solution/{year)-{problem_number}_reference_solution.json`: gold-standard reference solutions for evaluation
`data/problem_metadata.json`: typological information of problems in the benchmark
`data/reasoning_traces.json`: gold-standard reasoning traces for each problem

## Run Evaluation

**To run the evaluation script:**
```bash
python -m src.eval.evaluate
```

## Solution Format

For a solution to be evaluated successfully, the solution must be a `.json` file with names and values:

`id`: the problem ID in the format of `"{year}-{problem_number}"`
`answer`: an object determined by the schema `answer_format` in `data/problem/{id}_problem.json
`explanation`: a string

**Example:**
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

## Run Solvers

**To reproduce the experiments in the paper:**
```bash
python -m src.solver.run_solvers --setting ${SETTING}
```
**Available settings:**
```python
["baseline_gemini", "baseline_openai", "gpt-5", "guided_gemini", "guided_openai",
"grammar_gemini", "single_gemini", "single_openai", "moa"]
```