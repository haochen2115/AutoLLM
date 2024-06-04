import json
import pandas as pd
from tqdm import tqdm

from models import Mock_OpenAI_Models
from prompt import NEWS_PROMPT, QUESTION_PROMPT

def generate_instructions_from_news(model, text):
    """Generates instruction data from given news text using the model.
    
    Args:
        model: The model to generate question-answer pairs.
        text: The news text to generate instructions from.
    
    Returns:
        A list of instructions containing questions and answers.
    """
    try:
        instructions = []
        question_answer_pairs = json.loads(model.answer(NEWS_PROMPT.format(text)))
        for qa in question_answer_pairs:
            instructions.append({
                "question": QUESTION_PROMPT.format(text, qa['question']),
                "answer": qa['answer']
            })
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        instructions = []
    except Exception as e:
        print(f"An error occurred: {e}")
        instructions = []
    
    return instructions

if __name__ == "__main__":
    dataset = []
    model = Mock_OpenAI_Models()
    news = pd.read_csv("data/news.csv")
    
    for _, row in tqdm(news.iterrows(), desc="Processing news articles"):
        for instruction in generate_instructions_from_news(model, row['content']):
            dataset.append(instruction)
    
    # Split dataset into chunks and save to JSON files
    chunk_size = len(dataset) // 5
    for i in range(4):
        file_path = f"data/delta_data_{i + 1}.json"
        with open(file_path, "w") as f:
            json.dump(dataset[chunk_size * i:chunk_size * (i + 1)], f, indent=2)
    
    eval_file_path = "data/eval_data.json"
    with open(eval_file_path, "w") as f:
        json.dump(dataset[chunk_size * 4:], f, indent=2)
