#!/bin/bash
export OPENAI_API_KEY="<< Your OpenAI API key >>"

result_folder="<< Dir to save the result >>"
criteria=("general_1_understanding" "general_2_interpersonal_effectiveness" "general_3_collaboration" "CBT_1_guided_discovery" "CBT_2_focus" "CBT_3_strategy")

for crt in "${criteria[@]}"; do
    python src/evaluation.py \
        --model_name "gpt-4o" \
        --input_path ${result_folder}/results.json \
        --prompt_name ./prompts/${crt}.txt \
        --max_tokens 256 \
        --save_dir ${result_folder}/score_${crt}.json
done

criteria_str="${criteria[@]}"
python src/get_score.py \
    --result_foler ${result_folder} \
    --criteria_list "${criteria_str}"