# Problems

Here's a detailed description of the format of problem files

## ID

`"id"`: The problem ID, in the format of `"{year}-{problem_number}"`

## Content

`"content"`: The problem content text.

## Answer Format

`"answer_format"`: A **JSON schema** specifying the format of the solver's output, for the "answer" part of the solution

Each key of `"answer_format"` refers to a specific question found in the problem content. For instance, `"b_10_translation"` may refer to translating the sentence (10) in task (b).

Each value of `"answer_format"` specifies the data type (only `string` or `list`) required for the question. And the minimum and maximum length if it requires a list.

The following schema can be used to guide the model to output both the "answer" and "explanation" in the correct format, if the model supports structured output.

```python
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
```