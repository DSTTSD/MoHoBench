
batch_size=16
model_name=O1
max_tokens=12000

# output file path
output_dir=""
# input data file name
data_dir=""
# image folder path for input data
image_folder=""

python openai_request.py \
    --request_model $model_name \
    --batch_size $batch_size \
    --save_path $output_dir/$input_data_name \
    --data_path $data_dir/$input_data_name.json \
    --image_folder $image_folder \
    --max_tokens $max_tokens