import sys
sys.path.append("/home/stduan/codes/vqa/honestvqa/")
from request_utils.visual_llm_utils import VisualLLMFactory
from request_utils.request_wrapper import get_model_visual_responses
import os
import json
import pandas as pd
from tqdm import tqdm
import argparse

def get_eval_message(question, model_response, groundtruth_response):
    prompt = f"""Your task is to determine if the model response is correct given the question and groundtruth response.
Ensure to interpret the model response in accordance to the the question.

If the question asks about a detail of an element that is not present in the image, A prediction of "yes", "no" or "nothing" should be considered incorrect because it inaccurately suggests that the element is presented in the image.
The correct prediction in such cases should acknowledge the absence of the element in question by stating the element is not present.
If prediction says that it can not assist or cannot provide an answer, then the prediction is incorrect.
If the question is about counting, then the prediction is correct only it matches the groundtruth counts exactly.

question={question},
model_response={model_response}
groundtruth_response={groundtruth_response},

Your answer is: """
    return [
        {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
        ],
        }
    ]

def get_eval_response(request_dict_list, eval_model_name):
    request_messages = [get_eval_message(request_dict['question'], request_dict['model_answer'], request_dict["groundtruth responses"]) for request_dict in request_dict_list]
    responses = VisualLLMFactory.gather_multiple_messages(request_messages, eval_model_name, max_tokens=2048)
    for i in range(len(request_dict_list)):
        request_dict_list[i]['eval_answer'] = responses[i]
        request_dict_list[i]['eval_model_name'] = eval_model_name
    return request_dict_list

def eval_all_data(data_list, batch_size, save_path):
    # start index
    start_index = 0
    for i in range(0, len(data_list), batch_size):
        if 'eval_answer' in data_list[i] and data_list[i]['eval_answer'] is not None:
            continue
        else:
            start_index = i
            break
    for i in tqdm(range(start_index, len(data_list), batch_size)):
        request_dict_list = data_list[i:i+batch_size] if i+batch_size < len(data_list) else data_list[i:]
        request_dict_list = get_eval_response(request_dict_list, eval_model_name)
        if i + batch_size < len(data_list):
            data_list[i:i+batch_size] = request_dict_list
        else:
            data_list[i:] = request_dict_list
        # save and replace the local data
        pd.DataFrame(data_list).to_json(save_path, lines=True, orient='records')

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--data_path", type=str, default="./all_data.json")
    arg.add_argument("--batch_size", type=int, default=64)
    arg.add_argument("--image_folder", type=str, default="/home/v-shiduan/blob/data/val2014")
    arg.add_argument("--eval_model_name", type=str, default="GPT-4o")
    arg.add_argument("--request_model", type=str, default="GPT-4o-Mini") # GPT-4o
    arg.add_argument("--save_path", type=str, default="/home/stduan/codes/vqa/honestvqa/haloquest/model_outputs")
    arg.add_argument("--eval_path", type=str, default=None)
    arg.add_argument("--max_tokens", type=int, default=12000)
    args = arg.parse_args()

    batch_size = args.batch_size
    eval_model_name = args.eval_model_name
    request_model = args.request_model
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    input_data_path = args.data_path
    if args.eval_path is not None:
        # eval mode "/home/stduan/codes/vqa/honestvqa/haloquest/eval_outputs/GPT-4o-Mini.json"
        eval_data = pd.read_json(args.eval_path, lines=True).to_dict('records')
        eval_all_data(eval_data, batch_size, args.eval_path)
    else:
        model_save_path = os.path.join(save_path, f"{request_model}.json")
        input_data = pd.read_json(input_data_path, lines=True)
        def map_image_path(image_name):
            return os.path.join(args.image_folder, image_name)
        if 'image_path' not in input_data.columns or '/' not in input_data['image_path'][0]:
            # assert 'image' in input_data.columns, "The input data should contain 'image' column."
            if 'image' in input_data.columns:
                input_data['image_path'] = input_data['image'].map(map_image_path)
            else:
                input_data['image_path'] = input_data['image_path'].map(map_image_path)
        input_data = input_data.to_dict('records')
        saved_data = []
        
        if os.path.exists(model_save_path):
            saved_data = pd.read_json(model_save_path, lines=True)
            all_questions = saved_data['question'].unique()
            input_data = [d for d in input_data if d['question'] not in all_questions and os.path.exists(d['image_path'])]
            saved_data = saved_data.to_dict('records')
            print("Loaded existing data: ", len(saved_data))

        for i in tqdm(range(0, len(input_data), batch_size)):
            request_dict_list = input_data[i:i+batch_size] if i+batch_size < len(input_data) else input_data[i:]
            request_dict_list = get_model_visual_responses(request_dict_list, request_model, request_type='local', max_tokens=args.max_tokens)
            saved_data.extend(request_dict_list)
            pd.DataFrame(saved_data).to_json(model_save_path, lines=True, orient='records')
        # eval_data = pd.read_json(model_save_path, lines=True).to_dict('records')
        # eval_all_data(eval_data, batch_size, model_save_path.replace(".json", "_eval.json"))
