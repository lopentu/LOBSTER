# Reference Solutions

Here's a detailed description of the format of reference solution files, and how the evaluation scripts matches the solver's solution to the reference solution.

Each reference solution is a `.json` file with the following items:

## ID

`"id"`: The problem ID, in the format of `"{year}-{problem_number}"`

## Answer

`"answer"`: A dictionary/object.

Each value is string representation of the correct answer or the set of possible correct answers for the question, with the **matching type** which determines how the evaluation script should judge for that question.

Here, an **entry** refers to an answer to a single question in the problem.

### Matching Types

#### Exact Match

The solver's entry (which should be a string for questions of this type) is normalized and compared to the reference entry by string equality ("=="). The score is `1` when the strings are equally, otherwise `0`.

#### Fuzzy Match

Identified by the prefix `<fuzzy>` in the string representation, this matching type allows non-exact matching for questions that requires it.

In the default setting, an LLM is used to determine if the solver's entry is equivalent to the reference entry and output either `1` for equivalence or `0` for otherwise.

Other metrics for this matching type can be implemented by users for customization.


#### Select Match

Identified by the prefix `<select{reflen},{minlen},{maxlen}>`, used in various scenarios where the question requires a list of items without specific order, and/or the question allows multiple possible answers.

#### Fuzzy Select

Identified by both the fuzzy prefix and the select prefix. An LLM is prompted to do the same task as Select Match, but allowing non-exact equivalents.

## Explanation

`"explanation"`: Consists of two parts:
- `"rules"`, a string of gold-standard explanation to the entire problem,
- `"rule_checklist"`, a list of separate criteria for scoring the solver's explanation.

In cases where explanation is not judged, both values are empty.