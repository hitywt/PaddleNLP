#bash ./kill_processes.sh

set -x
unset CUDA_VISIBLE_DEVICES

#export GLOG_v=6
export FLAGS_call_stack_level=2

#export FLAGS_benchmark=1

#export FLAGS_log_memory_stats=True

task_name="llama1_13B_pp2_sd4_st"
rm -rf output/$task_name/
rm -rf "output/$task_name""_log"

export SOT_LOG_LEVEL=4
export PYTHONPATH=../../:$PYTHONPATH

export PATH=/opt/nvidia/nsight-systems/2023.2.1/bin:$PATH

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1

export CUDA_DEVICE_MAX_CONNECTIONS=1

export PARALLEL_CROSS_ENTROPY=true

# hack configs back
#----------------------------------
#--gpus "0,1,2,3,4,5,6,7" \
#--sharding_parallel_degree 4 \
#----------------------------------

#nsys profile --stats=true -t cuda,nvtx,cublas,cudnn -o $task_name --capture-range=cudaProfilerApi --force-overwrite true \
python -u  -m paddle.distributed.launch \
    --gpus "4,5,6,7" \
    --log_dir "output/$task_name""_log" \
    auto_parallel/run_pretrain_auto.py \
    --model_type "llama" \
    --model_name_or_path "facebook/llama-13b" \
    --tokenizer_name_or_path "facebook/llama-13b" \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --split 949,50,1 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --bf16 1 \
    --fp16_opt_level "O2" \
    --amp_custom_white_list "fused_rms_norm" \
    --amp_custom_black_list "softmax_with_cross_entropy" \
    --scale_loss 1024 \
    --pipeline_parallel_degree 2 \
    --tensor_parallel_degree 1 \
    --sharding_parallel_degree 2 \
    --sharding "stage1" \
    --learning_rate 0.0001 \
    --min_learning_rate 0.00001 \
    --max_steps 50 \
    --save_steps 5000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --dataloader_num_workers 1 \
    --sharding "" \
    --eval_steps 1000 \
    --report_to "visualdl" \
    --disable_tqdm true \
    --continue_training 0 \
    --do_train \
    --do_eval 0 \
    --device "gpu" \
    --data_impl "mmap" \
    --parallel_mode "auto" \
    --use_flash_attention 1 \
    --fuse_attention_qkv 1 \
    --fuse_attention_ffn 0 \
    --use_fused_rope 1 \
    --use_fused_rms_norm 1 \
    --pipeline_parallel_config "enable_send_recv_overlap" \
    --tensor_parallel_config "enable_mp_async_allreduce" \
    --virtual_pp_degree 1 \
    --pipeline_schedule_mode "1F1B" \

