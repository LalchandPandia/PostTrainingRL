#!/bin/bash
#SBATCH --job-name=new_job_context_sbatch
#SBATCH --output=example_sbatch_subset_siqa_10.out
#SBATCH --error=example_sbatch_subset_siqa_10.err
#SBATCH --partition=general
#SBATCH --gpus-per-node=1
#SBATCH --exclude=i001-ds,j004-ds,p001,p002,p003
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --constraint='a100|h100|h200'
cuda_version=11.8
export CUDA_HOME=/usr/local/cuda-${cuda_version}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
echo $PATH
#pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
OUTPUT_DIR=/net/projects/clab/lcpandia/rlvr_8b
python open_instruct/ppo_vllm_thread_ray_gtrl.py \
    --dataset_mixer_list allenai/IF_multi_constraints_upto5 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/IF_multi_constraints_upto5 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-DPO \
    --reward_model_path allenai/Llama-3.1-Tulu-3-8B-RM \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --chat_template_name tulu \
    --learning_rate 3e-7 \
    --total_episodes 100 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --local_mini_batch_size 32 \
    --local_rollout_batch_size 32 \
    --actor_num_gpus_per_node 7 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.05 \
    --apply_verifiable_reward true \
    --output_dir ${OUTPUT_DIR} \
    --seed 3 \
    --num_evals 3 \
    --save_freq 100 \
    --reward_model_multiplier 0.0 \
    --gradient_checkpointing \
    --with_tracking