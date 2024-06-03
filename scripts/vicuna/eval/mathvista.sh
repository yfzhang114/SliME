
cd playground/data/eval/MathVista

model=${1:-"liuhaotian/llava-v1.5-7b"}
name=$model

cd evaluation
python generate_response_slime.py \
--model  /mnt/data/xue.w/yf/checkpoint/$model \
--output_dir ../results/$name \
--output_file output_$name.json \
--conv-mode vicuna_v1

python extract_answer.py \
--output_dir ../results/$name \
--output_file output_$name.json 

python calculate_score.py \
--output_dir ../results/$name \
--output_file output_$name.json \
--score_file scores_$name.json