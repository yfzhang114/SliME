cd playground/data/eval/MathVision

model=liuhaotian/llava-v1.5-7b
name=llava-v1.5-7b

name=${1:-"liuhaotian/llava-v1.5-7b"}
cd ./playground/data/eval/MathVision

for checkpoint in '' #checkpoint-3000 checkpoint-6000 
do

model_name=$name-$checkpoint
model_dir=/mnt/workspace/xue.w/yf/checkpoint/$name/$checkpoint

python models/LLaVa.py \
--model $model_dir \
--output_file ./outputs/$model_name.jsonl \
--conv-mode vicuna_v1

done
# This script will examine all outputs located in the outputs/ directory, computing overall accuracy as well as accuracy for each subject and level.
python evaluation/evaluate.py
rm -rf ./outputs