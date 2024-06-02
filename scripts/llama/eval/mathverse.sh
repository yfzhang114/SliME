
cd playground/data/eval/MathVerse

model=${1:-"liuhaotian/llava-v1.5-7b"}
name=$model

cd evaluation
# python generate_response_slime.py \
# --model  /mnt/data/xue.w/yf/checkpoint/$model \
# --output_dir ../results/$name \
# --output_file output_$name.json \
# --conv-mode llama3

python extract_answer_s1.py \
--model_output_file ../results/$name/output_$name.json \
--save_file ../results/$name/extract_$name.json \

python score_answer_s2.py \
--answer_extraction_file ../results/$name/extract_$name.json \
--save_file ../results/$name/scores_$name.json