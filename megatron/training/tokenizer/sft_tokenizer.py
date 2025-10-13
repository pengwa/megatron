# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""SFT tokenizer."""
from typing import Dict, List, Union
import numpy as np

nemotron_h_aligned_custom_template = """{% for message in messages %}{% if message['role'] == 'system' %}{{ '<SPECIAL_10>System\n' + message['content'].strip() + '\n' }}{% elif message['role'] == 'user' %}{{ '<SPECIAL_11>User\n' + message['content'].strip() + '\n' + '<SPECIAL_11>Assistant\n' }}{% elif message['role'] == 'assistant' %}{{ message['content'].strip() + '\n' }}{% endif %}{% endfor %}"""
nemotron_nano_v2_custom_template = """{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'system' %}{{ '<SPECIAL_10>System\n' + content.replace('/think', '').replace('/no_think', '').strip() + '\n' }}{% elif message['role'] == 'user' %}{{ '<SPECIAL_11>User\n' + content.replace('/think', '').replace('/no_think', '').strip() + '\n' }}{% elif message['role'] == 'assistant' %}{{ '<SPECIAL_11>Assistant\n' + content.strip() + '\n<SPECIAL_12>\n' }}{% endif %}{% endfor %}"""

from megatron.core.datasets.megatron_tokenizer import MegatronLegacyTokenizer
from megatron.training.datasets.sft_dataset import IGNORE_INDEX
from megatron.training.tokenizer.multimodal_tokenizer import PromptConfig

class SFTTokenizer(MegatronLegacyTokenizer):  
    """SFT Tokenizer."""

    def __init__(
        self,
        tokenizer_path: str,
        prompt_format: str,
    ):
        """
        Note: Currently, only HuggingFaceTokenizer is supported as the underlying text tokenizer.

        Args:
            tokenizer_path (str): Underlying tokenizer path.
            prompt_format (str): Prompt format for the tokenizer.
        """
        super().__init__(tokenizer_path, prompt_format=prompt_format)
        try:
            import transformers
        except ImportError:
            raise ImportError(
                "SFTTokenizer currently requires transformers library to be installed"
            )

        # Currently, only HuggingFace tokenizers are supported.
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_path,
        )

        self._vocab_size = len(tokenizer)
        self._tokenizer = tokenizer

        if prompt_format == "nemotron-nano-v2":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=3,
                pad_token_id=tokenizer.convert_tokens_to_ids("<unk>"),
                custom_chat_template=nemotron_nano_v2_custom_template,
                has_bos=False,
                has_system_role=True,
            )
        elif prompt_format == "nemotron-h-aligned":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=0,
                pad_token_id=tokenizer.convert_tokens_to_ids("<SPECIAL_233>"),
                custom_chat_template=nemotron_h_aligned_custom_template,
                has_bos=False,
                has_system_role=True,
            )
        else:
            raise NotImplementedError("unknown SFT prompt format", prompt_format)

        self._prompt_format = prompt_format


    def tokenize_conversation(
        self, conversation: List[Dict], return_target: bool, add_generation_prompt: bool
    ):
        """Convert a conversation to tokens.

        Args:
            conversation (List[Dict]): Sequence of system/user/assistant messages.
                Must be in the following format:
                [
                    {"role": "system", "content": "something"},
                    {"role": "user", "content": "something1"},
                    {"role": "assistant", "content": "something2"},
                ]
            return_target (bool): Return target tokens with system and assistant masked.
            add_generation_prompt (bool): Add assistant prefix to the end.
        """
        # Skip system message if the tokenizer doesn't have a system role.
        if not self._prompt_config.has_system_role and conversation[0]["role"] == "system":
            conversation = conversation[1:]

        tokens = self._tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_assistant_token_mask=False,
            return_tensors="np",
            # pengwa: remove the chat template
            # chat_template=self._prompt_config.custom_chat_template,
            # pengwa, to workaround:
            # [rank0]: jinja2.exceptions.UndefinedError: Caught UndefinedError in DataLoader worker process 0.
            # [rank0]: Original Traceback (most recent call last):
            # [rank0]:   File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
            # [rank0]:     data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
            # [rank0]:   File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
            # [rank0]:     data = [self.dataset[idx] for idx in possibly_batched_index]
            # [rank0]:   File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
            # [rank0]:     data = [self.dataset[idx] for idx in possibly_batched_index]
            # [rank0]:   File "/root/megatron/megatron/training/datasets/sft_dataset.py", line 77, in __getitem__
            # [rank0]:     tokens, target = tokenizer.tokenize_conversation(
            # [rank0]:   File "/root/megatron/megatron/training/tokenizer/sft_tokenizer.py", line 87, in tokenize_conversation
            # [rank0]:     tokens = self._tokenizer.apply_chat_template(
            # [rank0]:   File "/usr/local/lib/python3.10/dist-packages/transformers/tokenization_utils_base.py", line 1641, in apply_chat_template
            # [rank0]:     rendered_chat, generation_indices = render_jinja_template(
            # [rank0]:   File "/usr/local/lib/python3.10/dist-packages/transformers/utils/chat_template_utils.py", line 498, in render_jinja_template
            # [rank0]:     rendered_chat = compiled_template.render(
            # [rank0]:   File "/usr/local/lib/python3.10/dist-packages/jinja2/environment.py", line 1295, in render
            # [rank0]:     self.environment.handle_exception()
            # [rank0]:   File "/usr/local/lib/python3.10/dist-packages/jinja2/environment.py", line 942, in handle_exception
            # [rank0]:     raise rewrite_traceback_stack(source=source)
            # [rank0]:   File "<template>", line 6, in top-level template code
            # [rank0]:   File "/usr/local/lib/python3.10/dist-packages/jinja2/sandbox.py", line 399, in call
            # [rank0]:     if not __self.is_safe_callable(__obj):
            # [rank0]:   File "/usr/local/lib/python3.10/dist-packages/jinja2/sandbox.py", line 265, in is_safe_callable
            # [rank0]:     getattr(obj, "unsafe_callable", False) or getattr(obj, "alters_data", False)
            # [rank0]: jinja2.exceptions.UndefinedError: 'None' has no attribute 'strip'
            continue_final_message=False, 
        )[0]

        if not return_target:
            return tokens

        target = tokens.copy()

        # Mask system and user tokens in the target.
        idx = 0
        for turn_idx, turn in enumerate(conversation):
            # disable to workaround
            # [rank0]: Traceback (most recent call last):
            # [rank0]:   File "/root/megatron/examples/post_training/modelopt/finetune.py", line 630, in <module>
            # [rank0]:     pretrain(
            # [rank0]:   File "/root/megatron/megatron/training/training.py", line 732, in pretrain
            # [rank0]:     iteration, num_floating_point_operations_so_far = train(
            # [rank0]:   File "/root/megatron/megatron/training/training.py", line 2254, in train
            # [rank0]:     ) = train_step(
            # [rank0]:   File "/root/megatron/megatron/training/training.py", line 1252, in train_step
            # [rank0]:     losses_reduced = forward_backward_func(
            # [rank0]:   File "/root/megatron/megatron/core/pipeline_parallel/schedules.py", line 2135, in forward_backward_pipelining_without_interleaving
            # [rank0]:     output_tensor, num_tokens = forward_step(
            # [rank0]:   File "/root/megatron/megatron/core/pipeline_parallel/schedules.py", line 402, in forward_step
            # [rank0]:     output_tensor, loss_func = forward_step_func(data_iterator, model)
            # [rank0]:   File "/root/megatron/examples/post_training/modelopt/finetune.py", line 609, in forward_step
            # [rank0]:     batch = get_batch(data_iterator)
            # [rank0]:   File "/root/megatron/examples/post_training/modelopt/finetune.py", line 473, in get_batch
            # [rank0]:     data = next(data_iterator)
            # [rank0]:   File "/root/megatron/megatron/core/rerun_state_machine.py", line 1062, in __next__
            # [rank0]:     n: Any = next(self.iterable)
            # [rank0]:   File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 733, in __next__
            # [rank0]:     data = self._next_data()
            # [rank0]:   File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 1515, in _next_data
            # [rank0]:     return self._process_data(data, worker_id)
            # [rank0]:   File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py", line 1550, in _process_data
            # [rank0]:     data.reraise()
            # [rank0]:   File "/usr/local/lib/python3.10/dist-packages/torch/_utils.py", line 750, in reraise
            # [rank0]:     raise exception
            # [rank0]: TypeError: Caught TypeError in DataLoader worker process 0.
            # [rank0]: Original Traceback (most recent call last):
            # [rank0]:   File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
            # [rank0]:     data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
            # [rank0]:   File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
            # [rank0]:     data = [self.dataset[idx] for idx in possibly_batched_index]
            # [rank0]:   File "/usr/local/lib/python3.10/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
            # [rank0]:     data = [self.dataset[idx] for idx in possibly_batched_index]
            # [rank0]:   File "/root/megatron/megatron/training/datasets/sft_dataset.py", line 77, in __getitem__
            # [rank0]:     tokens, target = tokenizer.tokenize_conversation(
            # [rank0]:   File "/root/megatron/megatron/training/tokenizer/sft_tokenizer.py", line 134, in tokenize_conversation
            # [rank0]:     if turn["role"].lower() == "assistant" and len(turn["content"]) == 0:
            # [rank0]: TypeError: object of type 'NoneType' has no len()
            # if turn["role"].lower() == "assistant" and len(turn["content"]) == 0:
            #     raise ValueError(f"empty assistant turn in conversation: {conversation}.")
            
            # if turn["role"].lower() == "assistant":
            #     assert conversation[turn_idx-1]["role"].lower() == "user"

            turn_tokens = self._tokenizer.apply_chat_template(
                [turn], tokenize=True, 
                # pengwa: remove the chat template
                #chat_template=self._prompt_config.custom_chat_template,
                # pegnwa:
                continue_final_message=False, 
            )

            # There should be only one BOS at the very beginning.
            # After the first turn, skip BOS token.
            if self._prompt_config.has_bos and turn_idx > 0:
                turn_tokens = turn_tokens[1:]
            turn_len = len(turn_tokens)

            role = turn["role"].lower()
            if role in ("system", "user"):
                target[idx : idx + turn_len] = IGNORE_INDEX
            elif role == "assistant":
                if self._prompt_config.assistant_prefix_len > 0:
                    target[idx : idx + self._prompt_config.assistant_prefix_len] = IGNORE_INDEX
            # pengwa: disable to workaround
            # else:
            #     raise ValueError(f"Wrong role value.")

            assert np.allclose(
                tokens[idx : idx + turn_len], turn_tokens
            ), f"expected turn tokens to match tokens in conversation {conversation}"

            idx += turn_len
        
        assert idx == len(tokens), f"mismatch in target masking the conversation {conversation}"

        return tokens, target

    def tokenize(self, text: Union[str, List[Dict]]):
        """Tokenize conversation or string input."""
        if isinstance(text, list):
            # This code path is used by the inference code currently.
            return self.tokenize_conversation(text, return_target=False, add_generation_prompt=True).tolist()

        return self._encode(text)

    def _encode(self, text: str):
        """Tokenize text input, w/o chat template"""
        return self._tokenizer.encode(text)

    def convert_tokens_to_ids(self, tokens: List[str]):
        """Convert tokens to IDs."""
        return self._tokenizer.convert_tokens_to_ids(tokens)

    def detokenize(self, tokens: List[int]):
        """Detokenize tokens."""
        return self._tokenizer.decode(tokens)

    def get_special_tokens(self):
        """Get special tokens."""
        return self._tokenizer.get_added_vocab()

    @property
    def force_eod(self):
        """To force an EOD at the end of every data sample in SFT."""
        return self._prompt_format == "nemotron-h-aligned"

    @property
    def pad(self):
        """Pad token ID."""
        return self._prompt_config.pad_token_id

    @property
    def eod(self):
        """End of sentence token ID."""
        return self._tokenizer.eos_token_id

    @property
    def vocab(self):
        """Vocab."""
        return NotImplementedError("not used")

    @property
    def inv_vocab(self):
        """Inverse vocab."""
        return NotImplementedError("not used")

    @property
    def vocab_size(self):
        """Vocabulary size."""
        return self._vocab_size
