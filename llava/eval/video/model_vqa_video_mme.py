import argparse
import torch
import os
import json
from tqdm import tqdm
import pdb

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def extract_frames(video_id, image_folder):
    # 构造与 video_id 同名的文件夹路径
    video_folder_path = os.path.join(image_folder, video_id)
    frames_folder_path = os.path.join(video_folder_path, 'frames')
    subs_path = os.path.join(video_folder_path, 'subtitles.txt')
    
    
    with open(subs_path, 'r', encoding='utf-8') as file:
        texts = file.read()
    # 检查 frames 文件夹是否存在
    if not os.path.exists(frames_folder_path):
        print(f"Frames folder does not exist: {frames_folder_path}")
        return []
    # 遍历 frames 文件夹中的所有图像
    image_files = [f for f in os.listdir(frames_folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    images = []
    for image_file in image_files:
        image_path = os.path.join(frames_folder_path, image_file)
        try:
            image = Image.open(image_path).convert('RGB')
            images.append(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
    
    return images, texts

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, prompt, subs):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.prompt = prompt
        self.subs = subs

    def __getitem__(self, index):
        line = self.questions[index]
        video_id = line['videoID']
        
        image_folder = self.image_folder
        images, texts = extract_frames(video_id, image_folder)
        image_tensor = process_images(images, self.image_processor, self.model_config, video=True)
        
        if self.subs:
            qs = f"This video's subtitles are listed below:\n {texts}\n" + f'Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.\n' + line["question"]
        else:
            qs = 'Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.\n' + line["question"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        qs += self.prompt
        
        choice_prompt = ' The choices are listed below: \n'
        for choice in line['options']:
            choice_prompt += choice + "\n"
        qs += choice_prompt  + 'The best answer is:'
        
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        print(prompt)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, images[0].size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4, prompt='', subs=False):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, prompt, subs)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    from datasets import load_dataset

    questions = load_dataset("lmms-lab/Video-MME", cache_dir=os.path.abspath('/mnt/data/xue.w/yf/Video-MME/.huggingface'))['test']
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, prompt=args.test_prompt, subs=args.subs)

    index, cnt_images, pre_video_id, answer_item = 0, [], '', {}
    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        video_id = line["video_id"]
        cur_prompt = line["question"]
        
        if pre_video_id != video_id and pre_video_id != '':
            ans_file.write(json.dumps(answer_item) + "\n")
            ans_file.flush()
            
        if pre_video_id == '' or video_id != pre_video_id:
            pre_video_id = video_id
            answer_item = {
                "video_id": line['video_id'],
                "duration": line['duration'],
                "domain": line['domain'],
                "sub_category": line['sub_category'],
                "questions": []
            }
        
        
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        cnt_images.append(image_tensor.shape[0])
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
            
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        index += 1
        if index % 100 == 0:
            print(f'Prompt: {cur_prompt}\n\n Output: {outputs}')
        answer_item['questions'].append({
                "question_id": line['question_id'],
                "task_type": line['task_type'],
                "question": line['question'],
                "options": line['options'],
                "answer": line['answer'],
                "response": outputs,
            })
        if index == len(questions):
            ans_file.write(json.dumps(answer_item) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/mnt/data/xue.w/yf/Video-MME/extract_frames_8")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="./slime_vicuna_13B_8frame.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=64)

    parser.add_argument("--subs", type=bool, default=False)
    parser.add_argument("--qlora-path", type=str, default="")

    parser.add_argument(
        "--test-prompt",
        type=str,
        default="",
    )
    args = parser.parse_args()
    print(args)

    eval_model(args)
