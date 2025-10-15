# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from typing import Any, Dict, List, Optional, Union
import logging

import numpy as np
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split

IGNORE_INDEX = -100


def compute_position_id_with_mask(mask):
    """Compute position IDs with attention mask (from VERL's implementation)."""
    return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)


def postprocess_data(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_length: int,
    pad_token_id: int,
    left_pad=True,
    truncation="error",
):
    """Process tokenizer outputs to consistent shapes via padding/truncation.

    Args:
        input_ids: Token indices [batch_size, seq_len]
        attention_mask: Mask [batch_size, seq_len]
        max_length: Target sequence length
        pad_token_id: Padding token ID
        left_pad: Pad left if True
        truncation: "left", "right", "middle" or "error"

    Returns:
        (input_ids, attention_mask) padded/truncated to max_length
    """
    assert truncation in ["left", "right", "middle", "error"]
    assert input_ids.ndim == 2

    sequence_length = input_ids.shape[-1]
    if sequence_length < max_length:
        input_ids = pad_sequence_to_length(
            input_ids, max_seq_len=max_length, pad_token_id=pad_token_id, left_pad=left_pad
        )
        attention_mask = pad_sequence_to_length(
            attention_mask, max_seq_len=max_length, pad_token_id=0, left_pad=left_pad
        )
    elif sequence_length > max_length:
        if truncation == "left":
            # actually, left truncation may not be reasonable
            input_ids = input_ids[:, -max_length:]
            attention_mask = attention_mask[:, -max_length:]
        elif truncation == "right":
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
        elif truncation == "middle":
            left_half = max_length // 2
            right_half = max_length - left_half
            input_ids = torch.cat([input_ids[:, :left_half], input_ids[:, -right_half:]], dim=-1)
            attention_mask = torch.cat([attention_mask[:, :left_half], attention_mask[:, -right_half:]], dim=-1)
        elif truncation == "error":
            raise NotImplementedError(f"{sequence_length=} is larger than {max_length=}")
        else:
            raise NotImplementedError(f"Unknown truncation method {truncation}")

    return input_ids, attention_mask


def pad_sequence_to_length(tensors, max_seq_len, pad_token_id, left_pad=False):
    """
    pad a 2D tensors (e.g. responses, logprobs) in the last dim to max_seq_length.
    input shape: [bs, seq_length]
    output shape: [bs, max_seq_length]
    """
    if tensors.shape[-1] >= max_seq_len:
        return tensors
    # (0, max_seq_len - tensors.shape[-1]) means right pad to max_seq_length and no left pad
    pad_tuple = (max_seq_len - tensors.shape[-1], 0) if left_pad else (0, max_seq_len - tensors.shape[-1])
    return torch.nn.functional.pad(tensors, pad_tuple, "constant", pad_token_id)


def convert_nested_value_to_list_recursive(data_item):
    """Convert nested numpy arrays and pandas series to lists recursively."""
    if isinstance(data_item, dict):
        return {k: convert_nested_value_to_list_recursive(v) for k, v in data_item.items()}
    elif isinstance(data_item, list):
        return [convert_nested_value_to_list_recursive(elem) for elem in data_item]
    elif isinstance(data_item, np.ndarray):
        # Convert to list, then recursively process the elements of the new list
        return convert_nested_value_to_list_recursive(data_item.tolist())
    else:
        # Base case: item is already a primitive type (int, str, float, bool, etc.)
        return data_item


class SFTLowLevelDataset:
    """The low-level dataset loading jsonl data for SFT

    Args:
        dataset_path (str): The path to jsonl data
            Each line of the jsonl must have key "messages" (List[Dict]),
            which is a sequence of system/user/assistant messages.
            Must be in the following format:
            [
                {"role": "system", "content": "something"},
                {"role": "user", "content": "something1"},
                {"role": "assistant", "content": "something2"},
            ]
    """

    def __init__(self, dataset_path: str) -> None:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "SFTDataset currently requires datasets library to be installed"
            )
        self.dataset = load_dataset("json", data_files=dataset_path, split="all")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> list:
        return self.dataset[idx]["messages"]


class MultiTurnSFTLowLevelDataset:
    """The low-level dataset loading parquet data for Multi-turn SFT
    
    Args:
        dataset_path (str or list): The path(s) to parquet file(s)
            Each record must have key "messages" (List[Dict]),
            which is a sequence of system/user/assistant messages.
            Must be in the following format:
            [
                {"role": "system", "content": "something"},
                {"role": "user", "content": "something1"},
                {"role": "assistant", "content": "something2"},
            ]
        config (dict): Configuration dictionary containing:
            - messages_key: Key name for messages (default: "messages")  
            - tools_key: Key name for tools (default: "tools")
            - enable_thinking_key: Key name for enable_thinking (default: "enable_thinking")
    """
    
    def __init__(self, dataset_path: Union[str, list], config: Optional[Dict] = None) -> None:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "MultiTurnSFTDataset requires pandas library to be installed"
            )
        
        # Set defaults from config
        config = config or {}
        self.messages_key = config.get("messages_key", "messages")
        self.tools_key = config.get("tools_key", "tools") 
        self.enable_thinking_key = config.get("enable_thinking_key", "enable_thinking")
        
        # Handle single file or list of files
        if not isinstance(dataset_path, list):
            dataset_path = [dataset_path]
        
        self.parquet_files = dataset_path
        self._read_files_and_process()
    
    def _read_files_and_process(self):
        """Read parquet files and extract data columns."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for parquet file processing")
            
        def series_to_item(ls):
            """Convert pandas series to item if it's a single element."""
            while isinstance(ls, (pd.core.series.Series, np.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls
        
        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        
        self.dataframe = pd.concat(dataframes, ignore_index=True)
        
        # Extract messages list from dataframe
        self.messages = self.dataframe[self.messages_key].apply(series_to_item).tolist()
        
        # Extract tools list from dataframe if available
        if self.tools_key in self.dataframe.columns:
            self.tools = self.dataframe[self.tools_key].apply(convert_nested_value_to_list_recursive).tolist()
        else:
            self.tools = [None] * len(self.messages)
        
        # Extract enable_thinking list from dataframe if available
        if self.enable_thinking_key in self.dataframe.columns:
            self.enable_thinking = self.dataframe[self.enable_thinking_key].tolist()
        else:
            self.enable_thinking = [None] * len(self.messages)
    
    def __len__(self) -> int:
        return len(self.messages)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "messages": self.messages[idx],
            "tools": self.tools[idx],
            "enable_thinking": self.enable_thinking[idx]
        }


class SFTDataset(MegatronDataset):
    """The dataset used during SFT"""

    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: Optional[str],
        indices: np.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> LowLevelDataset:
        return SFTLowLevelDataset(dataset_path)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        tokenizer = self.config.tokenizer
        max_seq_len = self.config.sequence_length

        conversation_list = self.dataset[int(self.indices[idx % len(self.indices)])]
        tokens, target = tokenizer.tokenize_conversation(
            conversation_list, return_target=True, add_generation_prompt=False
        )

        force_eod_length = int(tokenizer.force_eod)

        if len(tokens) > max_seq_len - force_eod_length:
            tokens = tokens[: max_seq_len - force_eod_length]
            target = target[: max_seq_len - force_eod_length]

        # padding
        num_tokens = len(tokens) + force_eod_length
        padding_len = max_seq_len - num_tokens
        assert padding_len >= 0
        filler = [tokenizer.eod] * force_eod_length + [tokenizer.pad] * (padding_len + 1)

        tokens = np.array(tokens.tolist() + filler, dtype=np.int64)
        target = np.array(target.tolist() + filler, dtype=np.int64)

        tokens = torch.tensor(tokens)
        target = torch.tensor(target)

        tokens = tokens[:-1].contiguous()
        target = target[1:].contiguous()

        loss_mask, position_ids, attention_mask = self._get_ltor_masks_and_position_ids(
            max_seq_len, target, tokenizer.pad
        )

        if self.config.create_attention_mask:
            ret = {
                'tokens': tokens,
                'labels': target,
                'attention_mask': attention_mask,
                'loss_mask': loss_mask,
                'position_ids': position_ids,
            }
        else:
            ret = {
                'tokens': tokens,
                'labels': target,
                'loss_mask': loss_mask,
                'position_ids': position_ids,
            }

        return ret

    def _get_ltor_masks_and_position_ids(self, max_seq_len, target, pad_token):
        """Build masks and position id for left to right model for SFT"""

        assert not self.config.reset_position_ids and not self.config.reset_attention_mask

        # Position ids.
        position_ids = torch.arange(max_seq_len, dtype=torch.long)

        # Loss mask.
        loss_mask = torch.ones(max_seq_len, dtype=torch.float)
        loss_mask[target == pad_token] = 0.0  # mask paddings
        loss_mask[target == IGNORE_INDEX] = 0.0  # mask prompts

        if self.config.create_attention_mask:
            attention_mask = torch.tril(
                torch.ones((max_seq_len, max_seq_len))
            ).unsqueeze(0)
            # Convert attention mask to binary:
            attention_mask = attention_mask < 0.5
        else:
            attention_mask = None

        return loss_mask, position_ids, attention_mask


class MultiTurnSFTDataset(MegatronDataset):
    """Multi-turn SFT Dataset re-implementing VERL's logic with Megatron's interface
    
    This dataset handles multi-turn conversations where each assistant response should be trained.
    It supports different padding modes and truncation strategies similar to VERL's implementation.
    """
    
    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: Optional[str],
        indices: np.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
       
    ) -> None:
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

        # Set defaults and extract parameters from config if provided (following VERL structure)
        multiturn_config = getattr(config, 'multiturn_config', {})
        self.pad_mode = multiturn_config.get("pad_mode", "right")
        assert self.pad_mode in ["right", "left_right"], (
            f"Expect pad_mode to be 'right' or 'left_right'. Got {self.pad_mode}"
        )
        self.truncation = multiturn_config.get("truncation", "left")
        assert self.truncation in ["error", "left", "right"]
        
        # for right padding
        self.max_length = multiturn_config.get("max_length", self.config.sequence_length)
        # for left right padding to be consistent with RL
        self.max_prompt_length = multiturn_config.get("max_prompt_length", 512)
        self.max_response_length = multiturn_config.get("max_response_length", 512)
        
        # Get messages_key from the new multiturn config structure
        self.messages_key = multiturn_config.get("messages_key", "messages")
        self.tools_key = multiturn_config.get("tools_key", "tools")
        self.enable_thinking_key = multiturn_config.get("enable_thinking_key", "enable_thinking")
        self.apply_chat_template_kwargs = multiturn_config.get("apply_chat_template_kwargs", {})

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> LowLevelDataset:
        # Extract multiturn config if available
        multiturn_config = getattr(config, 'multiturn_config', {})
        return MultiTurnSFTLowLevelDataset(dataset_path, multiturn_config)

    def __len__(self) -> int:
        return self.num_samples
    
    def _process_message_tokens(
        self,
        messages: List[Dict[str, Any]],
        start_idx: int,
        end_idx: int,
        is_assistant: bool = False,
        enable_thinking: Optional[bool] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[List[int], List[int], List[int]]:
        """
        Process tokens for a single message or a group of messages.

        Args:
            messages: List of message dictionaries
            start_idx: Start index in messages list
            end_idx: End index in messages list
            is_assistant: Whether this is an assistant message
            enable_thinking: Whether to enable thinking mode

        Returns:
            Tuple of (tokens, loss_mask, attention_mask)
        """
        if start_idx > 0:
            prev_applied_text = self.config.tokenizer._tokenizer.apply_chat_template(
                messages[:start_idx],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                tools=tools,
                **self.apply_chat_template_kwargs,
            )
            if is_assistant:
                prev_applied_text_w_generation_prompt = self.config.tokenizer._tokenizer.apply_chat_template(
                    messages[:start_idx],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                    tools=tools,
                    **self.apply_chat_template_kwargs,
                )

        else:
            prev_applied_text = ""

        cur_applied_text = self.config.tokenizer._tokenizer.apply_chat_template(
            messages[:end_idx],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
            tools=tools,
            **self.apply_chat_template_kwargs,
        )
        # Get tokens for the current message only
        if is_assistant:
            generation_prompt_text = prev_applied_text_w_generation_prompt[len(prev_applied_text):]
            generation_prompt_tokens = self.config.tokenizer._tokenizer.encode(
                generation_prompt_text,
                add_special_tokens=False,
            )
            _message_tokens = self.config.tokenizer._tokenizer.encode(
                cur_applied_text[len(prev_applied_text_w_generation_prompt):],
                add_special_tokens=False,
            )
            message_tokens = generation_prompt_tokens + _message_tokens
            loss_mask = [0] * (len(generation_prompt_tokens)) + [1] * (
                len(message_tokens) - len(generation_prompt_tokens)
            )
        else:
            message_tokens = self.config.tokenizer._tokenizer.encode(
                cur_applied_text[len(prev_applied_text):],
                add_special_tokens=False,
            )
            loss_mask = [0] * len(message_tokens)

        attention_mask = [1] * len(message_tokens)

        return message_tokens, loss_mask, attention_mask
    
    def _validate_and_convert_tokens(
        self,
        full_tokens: torch.Tensor,
        concat_tokens: List[int],
        concat_loss_mask: List[int],
        concat_attention_mask: List[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Validate tokenization and convert to tensors.

        Args:
            full_tokens: Full conversation tokens
            concat_tokens: Concatenated tokens
            concat_loss_mask: Concatenated loss mask
            concat_attention_mask: Concatenated attention mask

        Returns:
            Tuple of (input_ids, loss_mask, attention_mask) as tensors
        """
        full_tokens_list = full_tokens.tolist()

        if len(concat_tokens) != len(full_tokens_list) or not all(
            a == b for a, b in zip(concat_tokens, full_tokens_list, strict=True)
        ):
            logging.warning(
                f"Token mismatch detected! Full tokenization length: {len(full_tokens_list)}, Concatenated tokens "
                f"length: {len(concat_tokens)}. Using concatenated version."
                # f"full tokens text: {self.tokenizer.decode(full_tokens_list)}"
                # f"concat tokens text: {self.tokenizer.decode(concat_tokens)}"
            )
            return (
                torch.tensor(concat_tokens, dtype=torch.long),
                torch.tensor(concat_loss_mask, dtype=torch.long),
                torch.tensor(concat_attention_mask, dtype=torch.long),
            )

        return (
            full_tokens,
            torch.tensor(concat_loss_mask, dtype=torch.long),
            torch.tensor(concat_attention_mask, dtype=torch.long),
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a training sample with multi-turn conversation processing."""
        
        tokenizer = self.config.tokenizer._tokenizer  # Use underlying tokenizer directly like VERL
        
        # Get conversation data
        sample_data = self.dataset[int(self.indices[idx % len(self.indices)])]
        messages = sample_data["messages"]
        tools = sample_data.get("tools")
        enable_thinking = sample_data.get("enable_thinking")
        
        # First, get the full conversation tokens
        try:
            full_tokens = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                **self.apply_chat_template_kwargs,
            )
        except Exception as e:
            logging.error(
                f"Error applying chat template: {e}\nMessages: {messages}\nTools: {tools}\nEnable thinking: "
                f"{enable_thinking}"
            )
            raise
        
        # Track concatenated tokens for validation
        concat_tokens = []
        concat_loss_mask = []
        concat_attention_mask = []
        
        i = 0
        while i < len(messages):
            cur_messages = messages[i]
            if cur_messages["role"] == "assistant":
                # Process assistant message
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, i, i + 1, is_assistant=True, enable_thinking=enable_thinking, tools=tools
                )
                i += 1
            elif cur_messages["role"] == "tool":
                # Process consecutive tool messages
                st = i
                ed = i + 1
                while ed < len(messages) and messages[ed]["role"] == "tool":
                    ed += 1
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, st, ed, enable_thinking=enable_thinking, tools=tools
                )
                i = ed
            elif cur_messages["role"] in ["user", "system"]:
                # Process user or system message
                if cur_messages["role"] == "system" and i != 0:
                    raise ValueError("System message should be the first message")
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, i, i + 1, enable_thinking=enable_thinking, tools=tools
                )
                i += 1
            else:
                raise ValueError(f"Unknown role: {cur_messages['role']}")

            # override loss mask with mask in the dataset to handle multi-turn conversation
            override_loss_mask = cur_messages.get("loss_mask", None)
            if override_loss_mask is not None:
                if isinstance(override_loss_mask, np.ndarray):
                    override_loss_mask = override_loss_mask.item()
                assert isinstance(override_loss_mask, int), f"loss_mask should be int, got {type(override_loss_mask)}"
                assert override_loss_mask in [0, 1], f"loss_mask should be 0 or 1, got {override_loss_mask}"
                loss_mask = [override_loss_mask] * len(tokens)

            concat_tokens.extend(tokens)
            concat_loss_mask.extend(loss_mask)
            concat_attention_mask.extend(attention_mask)

        # Validate and convert tokens
        input_ids, loss_mask, attention_mask = self._validate_and_convert_tokens(
            full_tokens[0], concat_tokens, concat_loss_mask, concat_attention_mask
        )
        
        # encode prompt (following VERL structure)
        if messages[0]["role"] == "system":
            assert messages[1]["role"] == "user"
            assert messages[2]["role"] == "assistant"
            prompt_message_length = 2
        elif messages[0]["role"] == "user":
            assert messages[1]["role"] == "assistant"
            prompt_message_length = 1
        else:
            raise ValueError(f"Unknown role: {messages[0]['role']}")

        sequence_length = input_ids.shape[0]
        
        # Handle sequence length
        if self.pad_mode == "right":
            if sequence_length < self.max_length:
                # Pad sequences
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                padded_input_ids = torch.full((self.max_length - sequence_length + 1,), pad_token_id, dtype=input_ids.dtype)
                padded_attention_mask = torch.zeros((self.max_length - sequence_length + 1,), dtype=attention_mask.dtype)
                padded_loss_mask = torch.zeros((self.max_length - sequence_length + 1,), dtype=loss_mask.dtype)
                
                input_ids = torch.cat((input_ids, padded_input_ids))
                attention_mask = torch.cat((attention_mask, padded_attention_mask))
                loss_mask = torch.cat((loss_mask, padded_loss_mask))
            elif sequence_length > self.max_length:
                if self.truncation == "left":
                    input_ids = input_ids[-self.max_length - 1:]
                    attention_mask = attention_mask[-self.max_length - 1:]
                    loss_mask = loss_mask[-self.max_length - 1:]
                elif self.truncation == "right":
                    input_ids = input_ids[:self.max_length + 1]
                    attention_mask = attention_mask[:self.max_length + 1]
                    loss_mask = loss_mask[:self.max_length + 1]
                elif self.truncation == "error":
                    raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
                else:
                    raise ValueError(f"Unknown truncation method {self.truncation}")

            # Create position IDs
            position_ids = torch.arange(len(input_ids), dtype=torch.long)
            # Zero out position IDs for padding
            position_ids = position_ids * attention_mask

            if self.config.create_attention_mask:
                ret = {
                    'tokens': input_ids[:-1].contiguous(),
                    'labels': input_ids[1:].contiguous(),
                    'attention_mask': attention_mask[:-1].contiguous(),
                    'loss_mask': loss_mask[:-1].contiguous(),
                    'position_ids': position_ids[:-1].contiguous(),
                }
            else:
                ret = {
                    'tokens': input_ids[:-1].contiguous(),
                    'labels': input_ids[1:].contiguous(),
                    'loss_mask': loss_mask[:-1].contiguous(),
                    'position_ids': position_ids[:-1].contiguous(),
                }

            return ret
        elif self.pad_mode == "left_right":
            assert self.truncation == "error", "Only support error truncation for left_right pad mode"
            
            prompt_str = tokenizer.apply_chat_template(
                messages[:prompt_message_length],
                tools=tools,  
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                **self.apply_chat_template_kwargs,
            )
            prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
            prompt_length = len(prompt_ids)
            
            prompt_ids = input_ids[:prompt_length].unsqueeze(0)
            prompt_attention_mask = attention_mask[:prompt_length].unsqueeze(0)
            prompt_loss_mask = loss_mask[:prompt_length].unsqueeze(0)
            
            response_ids = input_ids[prompt_length:].unsqueeze(0)
            response_attention_mask = attention_mask[prompt_length:].unsqueeze(0)
            response_loss_mask = loss_mask[prompt_length:].unsqueeze(0)
            
            assert prompt_loss_mask.sum().item() == 0
            
            # Use postprocess_data logic for left-right padding
            prompt_ids, prompt_attention_mask = postprocess_data(
                input_ids=prompt_ids,
                attention_mask=prompt_attention_mask,
                max_length=self.max_prompt_length,
                pad_token_id=tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )
            response_ids, response_attention_mask = postprocess_data(
                input_ids=response_ids,
                attention_mask=response_attention_mask,
                max_length=self.max_response_length,
                pad_token_id=tokenizer.pad_token_id,
                left_pad=False,
                truncation=self.truncation,
            )
            response_loss_mask = pad_sequence_to_length(
                response_loss_mask, max_seq_len=self.max_response_length, pad_token_id=0, left_pad=False
            )
            
            prompt_ids = prompt_ids[0]
            prompt_attention_mask = prompt_attention_mask[0]
            response_ids = response_ids[0]
            response_attention_mask = response_attention_mask[0] 
            response_loss_mask = response_loss_mask[0]
            
            assert response_attention_mask[0].item() == 1
            assert response_loss_mask[0].item() == 1
            
            input_ids = torch.cat((prompt_ids, response_ids), dim=0)
            attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=0)
            
            # Compute position IDs with attention mask
            position_ids = compute_position_id_with_mask(attention_mask)
            
            # return {
            #     "input_ids": input_ids,
            #     "attention_mask": attention_mask,
            #     "position_ids": position_ids,
            #     "responses": response_ids,
            #     "response_mask": response_loss_mask,
            # }

            # shift to the tokens by 1 
            tokens = input_ids[:-1].contiguous()
            labels = input_ids[1:].contiguous()


            if self.config.create_attention_mask:
                ret = {
                    'tokens': tokens,
                    'labels': labels,
                    'attention_mask': attention_mask,
                    'loss_mask': loss_mask,
                    'position_ids': position_ids,
                    "responses": response_ids,
                    "response_mask": response_loss_mask,
                }
            else:
                ret = {
                    'tokens': tokens,
                    'labels': labels,
                    'loss_mask': loss_mask,
                    'position_ids': position_ids,
                    "responses": response_ids,
                    "response_mask": response_loss_mask,
                }

            return ret
            
        else:
            raise ValueError(f"Unknown pad mode {self.pad_mode}")


