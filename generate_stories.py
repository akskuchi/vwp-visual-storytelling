import os
import csv
import argparse
from tqdm import tqdm
import pandas as pd
import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
from PIL import Image


os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

datainfo = pd.read_csv('data/vwp-v2.1.csv')

num_to_word = {
    1: 'one',
    2: 'two',
    3: 'three',
    4: 'four',
    5: 'five',
    6: 'six',
    7: 'seven',
    8: 'eight',
    9: 'nine',
    10: 'ten'
}

prompts_VC = {
    'P1': 'Write a story using exactly [num_sents] sentences for this image sequence. Do not use more than [num_sents] sentences.',
    'P2': 'Generate a story consisting of [num_sents] sentences for this image sequence. Use only [num_sents] sentences and not more.',
    'P3': 'Output a story about this sequence of images using only [num_sents] sentences. Make sure the story does not include more than [num_sents] sentences.'
}


def load_model(model_name):
    if model_name == 'qwen-vl':
        version = 'Qwen/Qwen2.5-VL-7B-Instruct'
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            version, torch_dtype='auto', device_map='auto'
        )
        processor = AutoProcessor.from_pretrained(version)
    elif model_name == 'llava':
        version = 'llava-hf/llava-v1.6-mistral-7b-hf'
        processor = LlavaNextProcessor.from_pretrained(version)
        model = LlavaNextForConditionalGeneration.from_pretrained(version, torch_dtype=torch.float16)
        model.to(device)
    elif model_name == 'deepseek-vl' and device == 'cuda':
        model_path = "deepseek-ai/deepseek-vl-7b-chat"
        processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        model = model.to(torch.bfloat16).cuda().eval()
    else:
        raise NotImplementedError
    return model, processor


def load_vwp_metadata(args):
    data = {}
    raw_data = datainfo[datainfo.split==args.split]
    for idx, ind in enumerate(raw_data.index):
        data[str(idx)] = (eval(raw_data.loc[ind, 'img_id_list']), f'{args.vwp_data_path}/vwp-strips/{args.split}/{str(idx)}.png')

    return {'0': data['0']} if args.sample_run else data


def post_process(s):
    return s.replace(u'\u00e9', 'e').replace(u'\u00ea', 'e').replace('\n\n', ' ')
 

def generate_story_llava(seq, model, processor, prompt_id, num_sents):
    seq = Image.open(seq).convert('RGB')
    prompt = prompts_VC[prompt_id].replace('[num_sents]', num_to_word[num_sents])
    inputs = processor(prompt, seq, return_tensors='pt').to(device)
    output = model.generate(**inputs, max_new_tokens=500, num_beams=1, do_sample=False)
    output = processor.decode(output[0], skip_special_tokens=True)
    
    return post_process(output.split('[/INST]')[1].strip())


def generate_story_qwen(seq, model, processor, prompt_id, num_sents):
    prompt = prompts_VC[prompt_id].replace('[num_sents]', num_to_word[num_sents])
    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'image',
                    'image': seq,
                },
                {
                    'type': 'text', 
                    'text': prompt
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt',
    )
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=128, num_beams=1, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output[0]


def generate_story_deepseek(seq, model, processor, prompt_id, num_sents):
    tokenizer = processor.tokenizer
    prompt = prompts_VC[prompt_id].replace('[num_sents]', num_to_word[num_sents])
    conversation = [
        {
            'role': 'User',
            'content': f'<image_placeholder>{prompt}',
            'images': [seq]
        },
        {
            'role': 'Assistant',
            'content': ''
        }
    ]

    pil_images = load_pil_images(conversation)
    prepare_inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True
    ).to(model.device)

    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

    outputs = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=500,
        do_sample=False,
        num_beams=1,
        use_cache=True
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer


def generate_stories(args):
    model, processor = load_model(args.model)
    data = load_vwp_metadata(args)
    generate_story = generate_story_qwen if args.model == 'qwen-vl' else generate_story_llava if args.model == 'llava' else generate_story_deepseek

    stories = {}
    for sid, (img_ids, seq) in tqdm(data.items()):
        try:
            stories[sid] = (str(img_ids), generate_story(seq, model, processor, args.prompt_id, num_sents=len(img_ids)))
        except Exception as e:
            print(f'could not generate story for {sid}: {e}')
            stories[sid] = (str(img_ids), 'N/A')

    return stories


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='zero-shot stories for VWP using VLMs', 
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', default='qwen-vl', choices=['qwen-vl', 'llava', 'deepseek-vl'],
                        help='VLM to use')
    parser.add_argument('--vwp_data_path', default='data',
                        help='path containing VWP visual sequences')
    parser.add_argument('--split', default='val', choices=['val', 'test'],
                        help='split of VWP data')
    parser.add_argument('--sample_run', action='store_true',
                        help='pass --sample_run to test on 1 sample')
    parser.add_argument('--save_to', default='data/stories',
                        help='path to save generated stories')
    parser.add_argument('--prompt_id', default='P1', choices=['P1', 'P2', 'P3'],
                        help='prompt ID')
    args = parser.parse_args()
    print(f'ARGS: model: {args.model}, prompt: {args.prompt_id}, sample_run: {args.sample_run}')
    
    stories = generate_stories(args)
    
    save_to_file = f'{args.save_to}/{args.model}_{args.prompt_id.lower()}_{args.split}.csv'
    print(f'saving stories to {save_to_file}', end=' ')
    with open(save_to_file, 'w', newline='') as fh:
        fieldnames = ['story_id', 'img_id_list', 'generated_story']
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for key, (img_ids, story) in stories.items():
            writer.writerow({'story_id': key, 'img_id_list': img_ids, 'generated_story': story})
    fh.close()
    print('complete.\n')
