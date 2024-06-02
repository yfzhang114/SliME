


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

name="$1"
for checkpoint in '' #checkpoint-3000 checkpoint-6000 
do
    cd playground/data/eval/MMMU

    model=/mnt/workspace/xue.w/yf/checkpoint/$name/$checkpoint
    model_name=$name-$checkpoint

    cd eval

    python run_llava.py \
    --output_path outputs/$model_name.json \
    --model_path $model \
    --model-base meta-llama/Meta-Llama-3-8B-Instruct \
    --config_path configs/llava1.5.yaml \
    --conv-mode llama3

    python main_eval_only.py --output_path outputs/$model_name.json


done

# cd playground/data/eval/MMMU

# model=liuhaotian/llava-v1.5-7b
# name=llava-v1.5-7b

# cd eval

# python run_llava.py \
# --output_path outputs/$name.json \
# --model_path $model \
# --config_path configs/llava1.5.yaml

# python main_eval_only.py --output_path outputs/$name.json
