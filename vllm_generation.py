# SPDX-License-Identifier: Apache-2.0
"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for text generation.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""
import os
from tqdm import tqdm
import json
import random
from dataclasses import asdict
from PIL import Image
from typing import NamedTuple, Optional, List
import pandas as pd
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.lora.request import LoRARequest
import argparse

class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompts: List[str]
    stop_token_ids: Optional[List[int]] = None
    lora_requests: Optional[List[LoRARequest]] = None


# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.

def save_json(data, save_path):
    # with open(save_path, 'w') as f:
    #     json.dump(data, f, indent=4)

    # line by line
    with open(save_path, 'w') as f:
        for d in data:
            json.dump(d, f)
            f.write('\n')

def load_json(save_path):
    # with open(save_path, 'r') as f:
    #     data = json.load(f)

    # line by line
    data = []
    with open(save_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# BLIP-2
def run_blip2(questions: List[str], modality: str, model_path: str) -> ModelRequestData:
    assert modality == "image"

    # BLIP-2 prompt format is inaccurate on HuggingFace model repository.
    # See https://huggingface.co/Salesforce/blip2-opt-2.7b/discussions/15#64ff02f3f8cf9e4f5b038262 #noqa
    prompts = [f"Question: {question} Answer:" for question in questions]
    engine_args = EngineArgs(
        model=model_path,
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )

# Deepseek-VL2
def run_deepseek_vl2(questions: List[str], modality: str, model_path: str) -> ModelRequestData:
    assert modality == "image"

    model_name = model_path

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=8,
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
        hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
    )

    prompts = [
        f"<|User|>: <image>\n{question}\n\n<|Assistant|>:"
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Gemma 3
def run_gemma3(questions: List[str], modality: str, model_path: str) -> ModelRequestData:
    assert modality == "image"
    model_name = model_path

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=22000,
        max_num_seqs=2,
        mm_processor_kwargs={"do_pan_and_scan": True},
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    prompts = [("<bos><start_of_turn>user\n"
                f"<start_of_image>{question}<end_of_turn>\n"
                "<start_of_turn>model\n") for question in questions]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# GLM-4v
def run_glm4v(questions: List[str], modality: str, model_path: str) -> ModelRequestData:
    assert modality == "image"
    model_name = "THUDM/glm-4v-9b"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=2048,
        max_num_seqs=2,
        trust_remote_code=True,
        enforce_eager=True,
        hf_overrides={"architectures": ["GLM4VForCausalLM"]},
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    prompts = [
        f"<|user|>\n<|begin_of_image|><|endoftext|><|end_of_image|>\
        {question}<|assistant|>" for question in questions
    ]

    stop_token_ids = [151329, 151336, 151338]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )

# InternVL
def run_internvl(questions: List[str], modality: str, model_path: str) -> ModelRequestData:
    assert modality == "image"

    model_name = model_path

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=32000,
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    messages = [[{
        'role': 'user',
        'content': f"<image>\n{question}"
    }] for question in questions]
    prompts = tokenizer.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True)

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B/blob/main/conversation.py
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )


# LLaVA-1.5
def run_llava(questions: List[str], modality: str, model_path: str) -> ModelRequestData:
    assert modality == "image"

    prompts = [
        f"USER: <image>\n{question}\nASSISTANT:" for question in questions
    ]

    engine_args = EngineArgs(
        model=model_path,
        max_model_len=4096,
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# LLaVA-1.6/LLaVA-NeXT
def run_llava_next(questions: List[str], modality: str, model_path: str) -> ModelRequestData:
    assert modality == "image"

    prompts = [f"[INST] <image>\n{question} [/INST]" for question in questions]
    engine_args = EngineArgs(
        model=model_path,
        max_model_len=8192,
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# LLaVA-OneVision
def run_llava_onevision(questions: List[str],
                        modality: str, model_path: str) -> ModelRequestData:

    if modality == "video":
        prompts = [
            f"<|im_start|>user <video>\n{question}<|im_end|> \
        <|im_start|>assistant\n" for question in questions
        ]

    elif modality == "image":
        prompts = [
            f"<|im_start|>user <image>\n{question}<|im_end|> \
        <|im_start|>assistant\n" for question in questions
        ]

    engine_args = EngineArgs(
        model=model_path,
        max_model_len=16384,
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )



# MiniCPM-V
def run_minicpmv_base(questions: List[str], modality: str, model_name):
    assert modality in ["image", "video"]
    # If you want to use `MiniCPM-o-2_6` with audio inputs, check `audio_language.py` # noqa

    # 2.0
    # The official repo doesn't work yet, so we need to use a fork for now
    # For more details, please see: See: https://github.com/vllm-project/vllm/pull/4087#issuecomment-2250397630 # noqa
    # model_name = "HwwwH/MiniCPM-V-2"

    # 2.5
    # model_name = "openbmb/MiniCPM-Llama3-V-2_5"

    # 2.6
    # model_name = "openbmb/MiniCPM-V-2_6"
    # o2.6

    # modality supports
    # 2.0: image
    # 2.5: image
    # 2.6: image, video
    # o2.6: image, video, audio
    # model_name = "openbmb/MiniCPM-o-2_6"
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        trust_remote_code=True,
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )
    # NOTE The stop_token_ids are different for various versions of MiniCPM-V
    # 2.0
    # stop_token_ids = [tokenizer.eos_id]

    # 2.5
    # stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]

    # 2.6 / o2.6
    stop_tokens = ['<|im_end|>', '<|endoftext|>']
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    modality_placeholder = {
        "image": "(<image>./</image>)",
        "video": "(<video>./</video>)",
    }

    prompts = [
        tokenizer.apply_chat_template(
            [{
                'role': 'user',
                'content': f"{modality_placeholder[modality]}\n{question}"
            }],
            tokenize=False,
            add_generation_prompt=True) for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )


def run_minicpmo(questions: List[str], modality: str, model_path: str) -> ModelRequestData:
    return run_minicpmv_base(questions, modality, "openbmb/MiniCPM-o-2_6")


def run_minicpmv(questions: List[str], modality: str, model_path: str) -> ModelRequestData:
    return run_minicpmv_base(questions, modality, "openbmb/MiniCPM-V-2_6")


# Phi-3-Vision
def run_phi3v(questions: List[str], modality: str, model_path: str) -> ModelRequestData:
    assert modality == "image"

    prompts = [
        f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"
        for question in questions
    ]

    # num_crops is an override kwarg to the multimodal image processor;
    # For some models, e.g., Phi-3.5-vision-instruct, it is recommended
    # to use 16 for single frame scenarios, and 4 for multi-frame.
    #
    # Generally speaking, a larger value for num_crops results in more
    # tokens per image instance, because it may scale the image more in
    # the image preprocessing. Some references in the model docs and the
    # formula for image tokens after the preprocessing
    # transform can be found below.
    #
    # https://huggingface.co/microsoft/Phi-3.5-vision-instruct#loading-the-model-locally
    # https://huggingface.co/microsoft/Phi-3.5-vision-instruct/blob/main/processing_phi3_v.py#L194
    engine_args = EngineArgs(
        model=model_path,
        trust_remote_code=True,
        max_model_len=30000,
        max_num_seqs=4,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        mm_processor_kwargs={"num_crops": 16},
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Phi-4-multimodal-instruct
def run_phi4mm(questions: List[str], modality: str, model_path: str) -> ModelRequestData:
    """
    Phi-4-multimodal-instruct supports both image and audio inputs. Here, we
    show how to process image inputs.
    """
    assert modality == "image"
    # model_path = snapshot_download("microsoft/Phi-4-multimodal-instruct")
    # Since the vision-lora and speech-lora co-exist with the base model,
    # we have to manually specify the path of the lora weights.
    # vision_lora_path = os.path.join(model_path, "vision-lora")
    prompts = [
        f"<|user|><|image_1|>{question}<|end|><|assistant|>"
        for question in questions
    ]
    engine_args = EngineArgs(
        model=model_path,
        trust_remote_code=True,
        max_model_len=30000,
        max_num_seqs=4,
    )
    # lora_requests=[LoRARequest("vision", 1, vision_lora_path)],
    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )

# Qwen2.5-VL
def run_qwen2_5_vl(questions: List[str], modality: str, model_path: str) -> ModelRequestData:

    model_name = model_path

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        limit_mm_per_prompt={"image": 1},
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompts = [
        ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
         f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
         f"{question}<|im_end|>\n"
         "<|im_start|>assistant\n") for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )

# Pixtral HF-format
def run_pixtral_hf(questions: List[str], modality: str, model_name: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "mistral-community/pixtral-12b"

    # NOTE: Need L40 (or equivalent) to avoid OOM
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    prompts = [f"<s>[INST]{question}\n[IMG][/INST]" for question in questions]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )



model_example_map = {
    "blip-2": run_blip2,
    "deepseek_vl_v2": run_deepseek_vl2,
    "gemma3": run_gemma3,
    "glm4v": run_glm4v,
    "internvl_chat": run_internvl,
    "llava": run_llava,
    "llava-next": run_llava_next,
    "llava-onevision": run_llava_onevision,
    "minicpmo": run_minicpmo,
    "minicpmv": run_minicpmv,
    "phi3_v": run_phi3v,
    "phi4_mm": run_phi4mm,
    "pixtral_hf": run_pixtral_hf,

}

def match_name_to_func(model_name: str) -> ModelRequestData:
    """Match model name to the corresponding function."""
    # phi
    if 'phi-4' in model_name.lower():
        return run_phi4mm
    elif 'phi-3' in model_name.lower():
        return run_phi3v
    # miniCPM
    elif 'MiniCPM-o' in model_name.lower():
        return run_minicpmo
    elif 'MiniCPM-V' in model_name.lower():
        return run_minicpmv
    # pixtral
    elif 'pixtral' in model_name.lower():
        return run_pixtral_hf
    # llama
    elif 'deepseek-vl' in model_name.lower():
        return run_deepseek_vl2
    elif 'llava-next' in model_name.lower():
        return run_llava_next
    elif 'llava' in model_name.lower():
        return run_llava
    elif 'llava-onevision' in model_name.lower():
        return run_llava_onevision
    # glm
    elif 'glm-4v' in model_name.lower():
        return run_glm4v
    # gemma
    elif 'gemma-3' in model_name.lower():
        return run_gemma3
    # internvl
    elif 'internvl' in model_name.lower():
        return run_internvl
    # blip
    elif 'blip-2' in model_name.lower():
        return run_blip2
    elif 'pixtral' in model_name.lower():
        return run_pixtral_hf
    elif 'qwen' in model_name.lower():
        return run_qwen2_5_vl


def prepare_input_data(input_list, image_folder=None):
    new_input_data_list = []
    assert 'prompt' in input_list[0], "input_list should contain 'prompt' key"
    for d in input_list:
        prompt = d['prompt']
        if 'image_path' in d:
            image_path = d['image_path']
            if '/' not in d['image_path']:
                assert image_folder is not None, "image_folder is None"
                image_path = os.path.join(image_folder, d['image_path'])
        elif 'image' in d:
            image_name = d['image']
            assert image_folder is not None, "image_folder is None"
            image_path = os.path.join(image_folder, image_name)
        else:
            raise ValueError("No image or video inputs found in the input data.")
        image = Image.open(image_path).convert("RGB")
        new_input_data_list.append({
            'prompt': prompt,
            'multi_modal_data': {
                'image': image
            },
        })
    return new_input_data_list

def batch_generation(llm, sampling_params, batch_data, args):
    """
    generate text for a batch of input data
    """
    processed_input_list = prepare_input_data(batch_data, args.image_folder)
    outputs = llm.generate(processed_input_list, sampling_params=sampling_params, use_tqdm=True)
    output_texts = [output.outputs[0].text for output in outputs]
    return output_texts

def check_images(image_path, image_folder):
    if not os.path.exists(image_path):
        image_path = os.path.join(image_folder, image_path)
    if not os.path.exists(image_path):
        return False
    # try to load the image, and make sure it is not truncated
    try:
        image = Image.open(image_path)
        image.verify()
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return False
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./all_data.json")
    parser.add_argument("--model_path", type=str, default="Qwen/QVQ-72B-Preview") # Qwen/QVQ-72B-Preview
    parser.add_argument("--image_folder", type=str, default="image_data/images")
    parser.add_argument("--output_folder", type=str, default="/home/stduan/codes/vqa/visual/output")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument('--disable-mm-preprocessor-cache',
        action='store_true',
        help='If True, disables caching of multi-modal preprocessor/mapper.')
    parser.add_argument('--model-type',
                        '-m',
                        type=str,
                        default="llava",
                        choices=model_example_map.keys(),
                        help='Huggingface "model_type".')
    parser.add_argument('--modality',
                        type=str,
                        default="image",
                        choices=['image', 'video'],
                        help='Modality of the input.')
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="Set the seed when initializing `vllm.LLM`.")
    args = parser.parse_args()
    model_path = args.model_path
    model_name = model_path.split('/')[-1]

    input_data_path = args.data_path
    input_data_df = pd.read_json(input_data_path, lines=True)
    input_data_df['is_valid'] = input_data_df.apply(
        lambda x: check_images(x['image_path'], args.image_folder), axis=1)
    input_data_df = input_data_df[input_data_df['is_valid'] == True]
    input_data_list = input_data_df.to_dict('records')
    print("Total data: ", len(input_data_list))

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    save_path = os.path.join(args.output_folder, f'answer_{model_name}_{args.suffix}.json')
    save_data = []

    if os.path.exists(save_path):
        save_data = load_json(save_path)
        save_questions = [d['question'] for d in save_data]
        input_data_list = [d for d in input_data_list if d['question'] not in save_questions]
        print("Loaded existing data: ", len(save_data))
    
    questions = [d['question'] for d in input_data_list]
    modality = args.modality
    model_func = match_name_to_func(model_path)
    req_data = model_func(questions, modality, model_path)
    prompts = req_data.prompts
    # input_data_list = prepare_input_data(prompts, input_data_list, args.image_folder)

    for i, d in enumerate(input_data_list):
        input_data_list[i]['prompt'] = prompts[i]

    sample_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_model_len)
    engine_args = asdict(req_data.engine_args) | {"seed": args.seed} | {'gpu_memory_utilization': args.gpu_memory_utilization, 'tensor_parallel_size': args.tensor_parallel_size}
    try:
        llm = LLM(**engine_args)
    except Exception as e:
        engine_args['tensor_parallel_size'] = engine_args['tensor_parallel_size'] // 2
        llm = LLM(**engine_args)
    
    for i in tqdm(range(0, len(input_data_list), args.batch_size)):
        batch_data = input_data_list[i:i+args.batch_size] if i+args.batch_size < len(input_data_list) else input_data_list[i:]
        try:
            gen_texts = batch_generation(llm, sample_params, batch_data, args)
            for d, gen_text in zip(batch_data, gen_texts):
                d['model_answer'] = gen_text
                d['model_name'] = model_name
                save_data.append(d)
        except Exception as e:
            print("Error: ", e)
        save_json(save_data, save_path)

