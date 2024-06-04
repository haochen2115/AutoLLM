# Step1
python src/generate_data.py 

# Step2
echo "Ensure that you have downloaded llama factory to your autollm folder."

CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train config/llama3_sft_delta_1.yaml
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train config/llama3_sft_delta_2.yaml
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train config/llama3_sft_delta_3.yaml
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train config/llama3_sft_delta_4.yaml

# Step3
python src/auto_merge.py 

echo "End 🎉"