#!/bin/bash
# store repos ids in a list
repos=("google/gemma-1.1-2b-it" "google/recurrentgemma-2b-it" "microsoft/Phi-3-mini-4k-instruct" "meta-llama/Meta-Llama-3-8B-Instruct" "google/gemma-1.1-7b-it" "mistralai/Mistral-7B-Instruct-v0.2" "meta-llama/Llama-2-7b-chat-hf" "Equall/Saul-Instruct-v1")
prompt_strategies=("fs" "zs")
idk=(True False)
export CUDA_VISIBLE_DEVICES=0,1

# nested loop to iterate over the repos and, prompt strategies, and idk
for repo in "${repos[@]}"
do
    for prompt_strategy in "${prompt_strategies[@]}"
    do
        for idk in "${idk[@]}"
        do
            # run the python script
            export REPO_ID=$repo
            export PROMPT_STRATEGY=$prompt_strategy
            export IDK=$idk
            export INPUT_PATH='./data/prompts.json'
            python run_prompts.py
        done
    done
done