#!/bin/bash

# 2) JSONL → Megatron
python tools/preprocess_data.py \
  --input ~/datasets/megatron_sft_data/train.jsonl \
  --output-prefix ~/datasets/megatron_sft_data/megatron_corpus/train \
  --json-keys text  \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --append-eod --workers 8 --keep-newlines 

python tools/preprocess_data.py \
  --input ~/datasets/megatron_sft_data/test.jsonl \
  --output-prefix ~/datasets/megatron_sft_data/megatron_corpus/test \
  --json-keys text  \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --append-eod --workers 8 --keep-newlines 



#   python "${HOST_MEGATRON_LM_DIR}/tools/preprocess_data.py" \
#        --input your_dataset.json \
#        --output-prefix test_dataset \
#        --tokenizer-type HuggingFaceTokenizer \
#        --tokenizer-model /path/to/tokenizer.model \
#        --append-eod


# usage: preprocess_data.py [-h] [--vocab-size VOCAB_SIZE] [--padded-vocab-size PADDED_VOCAB_SIZE] [--vocab-file VOCAB_FILE] [--merge-file MERGE_FILE] [--vocab-extra-ids VOCAB_EXTRA_IDS]
#                           [--tokenizer-type {BertWordPieceLowerCase,BertWordPieceCase,GPT2BPETokenizer,SentencePieceTokenizer,GPTSentencePieceTokenizer,HuggingFaceTokenizer,Llama2Tokenizer,TikTokenizer,MultimodalTokenizer,NullTokenizer,NullMultimodalTokenizer,SFTTokenizer}]
#                           [--tokenizer-model TOKENIZER_MODEL] [--tokenizer-metadata TOKENIZER_METADATA] [--tiktoken-pattern TIKTOKEN_PATTERN]
#                           [--tiktoken-num-special-tokens TIKTOKEN_NUM_SPECIAL_TOKENS] [--tiktoken-special-tokens TIKTOKEN_SPECIAL_TOKENS [TIKTOKEN_SPECIAL_TOKENS ...]]
#                           [--legacy-tokenizer] [--trust-remote-code] --input INPUT [--json-keys JSON_KEYS [JSON_KEYS ...]] [--split-sentences] [--keep-newlines] [--append-eod]
#                           [--lang LANG] --output-prefix OUTPUT_PREFIX --workers WORKERS [--partitions PARTITIONS] [--log-interval LOG_INTERVAL] [--keep-sequential-samples]