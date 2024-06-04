import os
import json
import time
from tqdm import tqdm
from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours

from mix_model import mix_models
from models import Mock_OpenAI_Models, Llama3_Models
from prompt import EVALUATION_PROMPT

EVAL_DATA_PATH = "./data/eval_data.json"
MODEL_PATH_LLAMA3_8B = "meta-llama/Meta-Llama-3-8B-Instruct"
TRAIN_DELTA_PATHS = [
    "saves/llama3-8b/full/sft_delta_1",
    "saves/llama3-8b/full/sft_delta_2",
    "saves/llama3-8b/full/sft_delta_3",
    "saves/llama3-8b/full/sft_delta_4"
]

openai_model = Mock_OpenAI_Models()

def eval_score(questions, answers, model_answers):
    """Evaluates the model answers based on the provided questions and answers.
    
    Args:
        questions: A list of questions.
        answers: A list of correct answers corresponding to the questions.
        model_answers: A list of model-generated answers.
        
    Returns:
        The pass ratio of the model answers.
    """
    pass_cnt = 0
    for question, answer, model_answer in zip(questions, answers, model_answers):
        gpt_answer = openai_model.answer(EVALUATION_PROMPT.format(question, answer, model_answer))
        try:
            if json.loads(gpt_answer)['score'] == 1:
                pass_cnt += 1
        except Exception as e:
            print(f"GPT evaluation error! {gpt_answer}")
    
    pass_ratio = pass_cnt / len(questions)
    
    print(f"Pass count: {pass_cnt}")
    print(f"Pass ratio: {pass_ratio}")
    
    return pass_ratio

def merge(a, b, c, d):
    """Merges multiple models with given weights and returns the model path.
    
    Args:
        a, b, c, d: Weights for each model component.
        
    Returns:
        The path to the merged model.
    """
    start = time.time()
    model_path = f'saves/llama3-8b/autollm/{a}_{b}_{c}_{d}/'
    if os.path.isdir(model_path):
        print(f"🥛 Model [{model_path}] has been merged before!")
    else:
        print(f"☕️ Starting to merge models {model_path} ~ Take a coffee break ~")
        avg_model = mix_models(
            model_names_or_paths=[MODEL_PATH_LLAMA3_8B] + TRAIN_DELTA_PATHS,
            model_type='decoder',
            weights=[1 - a - b - c - d, a, b, c, d],
            output_path=model_path
        )
        print("🥛 Model merged!")
    
    print(f"[MERGE TIMECOST] {time.time() - start}")
    return model_path

def eval(model_path):
    """Evaluates a model at the given path.
    
    Args:
        model_path: The path to the model to be evaluated.
        
    Returns:
        The performance score of the model.
    """
    start = time.time()
    predictor = Llama3_Models(model_path)
    questions = []
    answers = []
    model_answers = []
    
    with open(EVAL_DATA_PATH, "r", encoding='utf-8') as f:
        sft_datas = json.load(f)
    
    for sft_data in tqdm(sft_datas, colour="green"):
        questions.append(sft_data['question'])
        answers.append(sft_data['answer'])
        model_answers.append(predictor.answer(sft_data['question']))

    performance = eval_score(questions, answers, model_answers)
    
    print(f"[EVAL TIMECOST] {time.time() - start}")
    
    return performance

def black_box_function(a, b, c, d):
    """Performs a full iteration of merging and evaluation.
    
    Args:
        a, b, c, d: Weights for each model component.
        
    Returns:
        The performance score of the merged model.
    """
    start = time.time()
    
    model_path = merge(a, b, c, d)
    performance = eval(model_path)
    
    print(f"Score for parameters {a}, {b}, {c}, {d} is {performance}")
    print(f"[ITER TIMECOST] {time.time() - start}")
    
    return performance

if __name__ == "__main__":
    pbounds = {'a': (0, 1), 'b': (0, 1), 'c': (0, 1), 'd': (0, 1)}
    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=1,
        verbose=2
    )

    init_point = {'a': 1, 'b': 1, 'c': 1, 'd': 1}
    init_performance = black_box_function(**init_point)
    optimizer.register(params=init_point, target=init_performance)
    
    optimizer.maximize(init_points=0, n_iter=2)
    
    print(Colours.green(f"Best parameter: {optimizer.max['params']}"))
