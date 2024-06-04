NEWS_PROMPT = '''You are an examiner tasked with evaluating a model’s abilities. Create questions based on the provided materials that meet the following criteria:
- Challenging, containing various styles and dimensions.
- Can combine or focus solely on content or format requirements.
- Only need 5 precise questions.

Possible inquiry angles/requirements:
- Summarize, analyze, infer, predict, translate, rewrite content.
- Extract and summarize information (time, place, event, data, terms, opinions, assumptions).
- Calculate missing but computable data.
- Infer past situations, future changes, event causes.
- Constrain output format (paragraph structure, specific formats, word limits, prohibited or mandatory words).

Output format should be a dictionary: 
[
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}}
]

Materials:
{}

Create your question now:
'''

QUESTION_PROMPT = '''You are a professional assistant, please answer the questions according to the given materials.
Materials:
{}

Question:
{}

Please give your answer:
'''

EVALUATION_PROMPT = '''You are a professional evaluator. You need to judge whether [model output] meets the requirements according to [question requirements] and [reference answer]. If there is any point that does not meet the requirements, please give 0 points and give reasons for rejection, otherwise give 1 point.
The following is [question request]:
{}
Here are the [answers]:
{}
Here is [Model Output]:
{}
please return the evaluation results in the following json format:
{{
    "score": 0
    "reason": "Word count requirement not met"
}}
Please directly return the evaluation result in json format:
'''
