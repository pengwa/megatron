#!/bin/bash

if [ -z ${HF_MODEL_CKPT} ]; then
    HF_MODEL_CKPT=Qwen/Qwen3-30B-A3B-Instruct-2507
    TOKENIZER_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
else
    TOKENIZER_MODEL=${HF_MODEL_CKPT}
fi

MAX_LENGTH=${MAX_LENGTH:-20000}

MODEL_ARGS=" \
    --save-interval 100000 \
    --micro-batch-size 1 \
    --bf16 \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --num-layers 48 \
    --hidden-size 2048 \
    --ffn-hidden-size 6144 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 4 \
    --kv-channels 128 \
    --qk-layernorm \
    --num-experts 128 \
    --moe-ffn-hidden-size 768 \
    --moe-router-topk 8 \
    --moe-router-dtype fp32 \
    --moe-aux-loss-coeff 1e-3 \
    --moe-token-dispatcher-type alltoall \
    --moe-router-load-balancing-type aux_loss \
    --seq-length ${MAX_LENGTH} \
    --max-position-embeddings 262144 \
    --tokenizer-type HuggingFaceTokenizer \
    --make-vocab-size-divisible-by 1187 \
    --use-mcore-models \
    --rotary-percent 1.0 \
    --rotary-base 10000000 \
    --sequence-parallel \
    --attention-backend flash \
    --use-flash-attn \
    --recompute-activations \
"
#     --moe-layer-recompute \
#     --no-masked-softmax-fusion \
#     --no-rope-fusion \
#     --no-bias-swiglu-fusion \

    # --kv-lora-rank 128 \
    # --q-lora-rank 128 \