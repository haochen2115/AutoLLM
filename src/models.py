import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

class Mock_OpenAI_Models:
    """A mock class representing OpenAI models for testing purposes."""
    
    def __init__(self):
        pass
    
    def answer(self, query: str) -> str:
        """Generates an answer based on the input query.
        
        Args:
            query: The input query string.
        
        Returns:
            default answer.
        """
        if "Please directly return the evaluation result in json format" in query:
            return '''{"score": 1, "reason": ""}'''
        elif("Create your question now" in query):
            return '''[
    {
        "question": "Summarize the main factors that contribute to the international sugar price forecast adjusting down to 19.5-23 cents per pound.",
        "answer": "The main factors contributing to the adjustment of the international sugar price forecast down to 19.5-23 cents per pound include a significant increase in sugarcane yield in the south, recovery of national sugar production growth, and improved international sugar supply expectations."
    },
    {
        "question": "Based on the provided materials, infer the potential reasons why farmers' enthusiasm for planting has increased recently.",
        "answer": "Farmers' enthusiasm for planting has increased recently due to improvements in sugar planting benefits, prominent comparative advantages with competitive crops, and generally normal weather conditions."
    },
    {
        "question": "Predict the trends in sugarcane and sugar beet sown areas for 2024/25 and suggest the implications for sugar production.",
        "answer": "For 2024/25, the sown area of sugarcane is expected to be flat or slightly increased, while the sown area of sugar beet is expected to increase significantly. This suggests that sugar output will maintain a restorative growth, possibly increasing to 11 million tons."
    },
    {
        "question": "Translate the expected sugar output growth into a brief statement limited to 20 words.",
        "answer": "Sugar output is expected to grow in 2024/25, increasing to 11 million tons with stable consumption, narrowing the production-demand gap."
    },
    {
        "question": "Rewrite the following sentence without using the word 'basically': 'The consumption was basically flat, and the domestic sugar production and demand gap narrowed.'",
        "answer": "The consumption remained steady, and the domestic sugar production and demand gap narrowed."
    }
]'''
        else:
            return "default answer"
        
class Llama3_Models:
    """A class representing the LLaMA3 model for question answering."""
    
    def __init__(self, model_path: str):
        """Initializes the LLaMA3 model with the specified model path.
        
        Args:
            model_path: The path to the pretrained model.
        """
        print(f"Starting to load model at {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation='flash_attention_2'
        ).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    def answer(self, query: str, max_length: int = 2048, temperature: float = 1.0, repetition_penalty: float = 1.0) -> str:
        """Generates an answer based on the input query using the LLaMA3 model.
        
        Args:
            query: The input query string.
            max_length: The maximum length of the generated response.
            temperature: The temperature for sampling.
            repetition_penalty: The penalty for repeated tokens.
        
        Returns:
            The generated response as a string.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
        
        input_ids = self.tokenizer.apply_chat_template(
            conversation=messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors='pt'
        )
        
        output_ids = self.model.generate(
            input_ids.to("cuda"),
            max_new_tokens=max_length,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            eos_token_id=[
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                self.tokenizer.convert_tokens_to_ids("<|end_of_text|>")
            ]
        )
        
        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response
