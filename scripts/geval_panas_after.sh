api_key_yaml="<< Dir to open yaml file >>"

data_path="<< Dir to evaluate the result >>"
save_folder="<< DIr to save the result >>"

python ./src/panas_scoring_after.py \
        --input_path $data_path \
        --prompt './prompts/panas_after.txt' \
        --save_dir $save_folder  \
        --api_key_path $api_key_yaml