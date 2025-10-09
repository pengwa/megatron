#!/bin/bash


SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Common arguments and base model specific arguments
source "${SCRIPT_DIR}/conf/arguments.sh"


# Set up cache dir for HF to avoid out of space error
export HF_DATASETS_CACHE="/tmp/hf_datasets_cache"

# Extra arguments of this script
MLM_DEFAULT_ARGS=" \
    --distributed-timeout-minutes 30 \
    --auto-detect-ckpt-format \
    --export-te-mcore-model \
    --finetune \
"


if [ -z ${MLM_MODEL_SAVE} ]; then
    MLM_MODEL_SAVE=${MLM_MODEL_CKPT}
    printf "${MLM_WARNING} Variable ${PURPLE}MLM_MODEL_SAVE${WHITE} is not set (default: ${MLM_MODEL_CKPT})!\n"
fi

if [ -z ${MLM_DATA_ARGS} ]; then
    MLM_DATA_ARGS=" \
        --train-samples 64 \
        --lr-decay-samples 32 \
        --lr-warmup-samples 0 \
        --split 100,0,0 \
        --finetune-hf-dataset Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered \
    "
fi

if [ -z ${MLM_TRAIN_ARGS} ]; then
    MLM_TRAIN_ARGS=" \
        --no-gradient-accumulation-fusion \
        --reset-position-ids \
        --reset-attention-mask \
        --eod-mask-loss \
        --micro-batch-size 1 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --no-check-for-nan-in-loss-and-grad \
    "
fi


if [ -z ${MLM_OPTIM_ARGS} ]; then
    MLM_OPTIM_ARGS=" \
        --lr 5.0e-5 \
        --min-lr 1.0e-7 \
        --lr-decay-style cosine \
        --clip-grad 1.0 \
        --weight-decay 0.0 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.010 \
	--use-distributed-optimizer \
    "
fi

if [ -z ${MLM_EVAL_ARGS} ]; then
    MLM_EVAL_ARGS=" \
        --eval-iters 1 \
        --eval-interval 1000 \
        --save-interval 1000 \
        --log-interval 1 \
    "
fi
unset CUDA_DEVICE_MAX_CONNECTIONS


 export NCCL_IB_DISABLE=1 
  export NCCL_SOCKET_IFNAME=eth0 
  export NCCL_NET_PLUGIN=none 
  export NCCL_IBEXT_DISABLE=1 
  export NCCL_SHARP_DISABLE=1 
  export NCCL_P2P_DIRECT_DISABLE=1 
  export NCCL_P2P_DISABLE=1 
  export NCCL_SHM_DISABLE=1 
  export NCCL_NET_GDR_LEVEL=0 
  export NCCL_DEBUG=WARN 
  export NCCL_DEBUG_SUBSYS=ALL



GPUS_PER_NODE=2
NUM_NODES=${NUM_NODES:-1}



MASTER_ADDR=${MASTER_ADDR:-10.0.0.4}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

#     --nproc_per_node $GPUS_PER_NODE
DISTRIBUTED_ARGS=(
    --nproc_per_node=$GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)



# ${LAUNCH_SCRIPT} 
#    --use-megatron-fsdp \
#  --data-parallel-sharding-strategy "optim_grads_params" \
#  --ckpt-format "fsdp_dtensor" \

torchrun ${DISTRIBUTED_ARGS[@]} \
 ${SCRIPT_DIR}/finetune.py \
    ${MODEL_ARGS} \
    --tensor-model-parallel-size ${TP} \
    --expert-tensor-parallel-size ${ETP} \
    --expert-model-parallel-size ${EP} \
    --pipeline-model-parallel-size ${PP} \
    --tokenizer-model ${TOKENIZER_MODEL} \
 --no-gradient-accumulation-fusion \
    ${MLM_DATA_ARGS} \
    ${MLM_OPTIM_ARGS} \
    ${MLM_TRAIN_ARGS} \
    ${MLM_EVAL_ARGS} \
    ${MLM_RESUME_ARGS} \
    ${MLM_DEFAULT_ARGS} ${MLM_EXTRA_ARGS} 



