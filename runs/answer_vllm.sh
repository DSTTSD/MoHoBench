
# list of models to run
models=(
    ""
)
model_base_dir="models/"
num_gpus=1
input_data_name="mohobench"
suffix="ind"
for model_name in "${models[@]}"
do
    model=${model_base_dir}${model_name}
    python3 avllm_generation.py \
            --model_path $model \
            --data_path "question_data/${input_data_name}.json" \
            --image_folder "data/${input_data_name}" \
            --output_folder "output/answer_${suffix}_${model_name}/" \
            --max_model_len 32000 \
            --gpu_memory_utilization 0.5 \
            --tensor_parallel_size $num_gpus \
            --disable-mm-preprocessor-cache \
            --batch_size 512 \
            --suffix $suffix
done
