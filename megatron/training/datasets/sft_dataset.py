# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from typing import Any, Dict, List, Optional, Union
import logging

import numpy as np
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split

IGNORE_INDEX = -100


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

        # MultiTurn specific configs
        pad_mode: str = "right"
        truncation: str = "error"
        max_length: int = self.config.sequence_length
        max_prompt_length: int = 512
        max_response_length: int = 512
        messages_key: str = "messages"
        tools_key: str = "tools"
        enable_thinking_key: str = "enable_thinking"
        apply_chat_template_kwargs: Optional[Dict] = None

        # Set padding and truncation configurations
        self.pad_mode = pad_mode
        assert self.pad_mode in ["right", "left_right"], (
            f"Expect pad_mode to be 'right' or 'left_right'. Got {self.pad_mode}"
        )
        
        self.truncation = truncation
        assert self.truncation in ["error", "left", "right"]
        
        # Set length configurations  
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        
        # Set key configurations
        self.messages_key = messages_key
        self.tools_key = tools_key
        self.enable_thinking_key = enable_thinking_key
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {}

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
        Process tokens for a single message or a group of messages using the underlying tokenizer directly.
        
        Args:
            messages: List of message dictionaries
            start_idx: Start index in messages list
            end_idx: End index in messages list
            is_assistant: Whether this is an assistant message
            enable_thinking: Whether to enable thinking mode
            tools: Tool definitions
            
        Returns:
            Tuple of (message_tokens, loss_mask, attention_mask)
        """
        tokenizer = self.config.tokenizer._tokenizer
        
        # Get tokens for the previous messages (context)
        if start_idx > 0:
            prev_applied_text = tokenizer.apply_chat_template(
                messages[:start_idx],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                tools=tools,
                **self.apply_chat_template_kwargs,
            )
        else:
            prev_applied_text = ""

        cur_applied_text = tokenizer.apply_chat_template(
            messages[:end_idx],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
            tools=tools,
            **self.apply_chat_template_kwargs,
        )

        # Get tokens for the current message only
        if is_assistant:
            generation_prompt_text = tokenizer.apply_chat_template(
                messages[:start_idx],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                tools=tools,
                **self.apply_chat_template_kwargs,
            )
            prev_applied_text_w_generation_prompt = generation_prompt_text
            generation_prompt_text = prev_applied_text_w_generation_prompt[len(prev_applied_text):]
            generation_prompt_tokens = tokenizer.encode(
                generation_prompt_text,
                add_special_tokens=False,
            )
            _message_tokens = tokenizer.encode(
                cur_applied_text[len(prev_applied_text_w_generation_prompt):],
                add_special_tokens=False,
            )
            message_tokens = generation_prompt_tokens + _message_tokens
            loss_mask = [0] * (len(generation_prompt_tokens)) + [1] * (
                len(message_tokens) - len(generation_prompt_tokens)
            )
        else:
            message_tokens = tokenizer.encode(
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
        """Validate concatenated tokens against full tokens and convert to tensors."""
        
        # Convert full_tokens to list for comparison
        if isinstance(full_tokens, torch.Tensor):
            full_tokens_list = full_tokens.tolist()
        else:
            full_tokens_list = full_tokens
        
        # Compare lengths and content
        if len(full_tokens_list) != len(concat_tokens):
            logging.warning(
                f"Token length mismatch: Full tokens length: {len(full_tokens_list)}, "
                f"Concatenated tokens length: {len(concat_tokens)}. Using concatenated version."
            )
            return (
                torch.tensor(concat_tokens, dtype=torch.long),
                torch.tensor(concat_loss_mask, dtype=torch.long),
                torch.tensor(concat_attention_mask, dtype=torch.long),
            )
        
        # Use full tokens if they match
        return (
            torch.tensor(full_tokens_list, dtype=torch.long) if not isinstance(full_tokens, torch.Tensor) else full_tokens,
            torch.tensor(concat_loss_mask, dtype=torch.long),
            torch.tensor(concat_attention_mask, dtype=torch.long),
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a training sample with multi-turn conversation processing."""
        
        tokenizer = self.config.tokenizer
        max_seq_len = self.config.sequence_length
        
        # Get conversation data
        sample_data = self.dataset[int(self.indices[idx % len(self.indices)])]
        messages = sample_data["messages"]
        tools = sample_data.get("tools")
        enable_thinking = sample_data.get("enable_thinking")
        
        # First, get the full conversation tokens using the underlying tokenizer
        try:
            full_tokens = tokenizer._tokenizer.apply_chat_template(
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
                f"Error applying chat template: {e}\nMessages: {messages}\n"
                f"Tools: {tools}\nEnable thinking: {enable_thinking}"
            )
            raise
        
        # Track concatenated tokens for validation
        concat_tokens = []
        concat_loss_mask = []
        concat_attention_mask = []
        
        # Process each message/group of messages
        i = 0
        while i < len(messages):
            cur_message = messages[i]
            
            if cur_message["role"] == "assistant":
                # Process assistant message
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, i, i + 1, is_assistant=True, 
                    enable_thinking=enable_thinking, tools=tools
                )
                i += 1
                
            elif cur_message["role"] == "tool":
                # Process consecutive tool messages
                st = i
                ed = i + 1
                while ed < len(messages) and messages[ed]["role"] == "tool":
                    ed += 1
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, st, ed, enable_thinking=enable_thinking, tools=tools
                )
                i = ed
                
            elif cur_message["role"] in ["user", "system"]:
                # Process user or system message
                if cur_message["role"] == "system" and i != 0:
                    raise ValueError("System message should be the first message")
                    
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, i, i + 1, enable_thinking=enable_thinking, tools=tools
                )
                i += 1
                
            else:
                raise ValueError(f"Unknown role: {cur_message['role']}")
            
            # Override loss mask if specified in the dataset
            override_loss_mask = cur_message.get("loss_mask", None)
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
        input_ids, loss_mask_tensor, attention_mask_tensor = self._validate_and_convert_tokens(
            full_tokens[0], concat_tokens, concat_loss_mask, concat_attention_mask
        )
        
        # Determine prompt length for left_right padding
        if messages[0]["role"] == "system":
            assert len(messages) >= 3 and messages[1]["role"] == "user" and messages[2]["role"] == "assistant"
            prompt_message_length = 2
        elif messages[0]["role"] == "user":
            assert len(messages) >= 2 and messages[1]["role"] == "assistant"
            prompt_message_length = 1
        else:
            raise ValueError(f"Unknown first role: {messages[0]['role']}")
        
        sequence_length = input_ids.shape[0]
        
        # Handle sequence length based on padding mode
        if self.pad_mode == "right":
            # Right padding mode - similar to original SFTDataset
            force_eod_length = int(tokenizer.force_eod)
            
            if sequence_length > max_seq_len - force_eod_length:
                # Truncate if needed
                if self.truncation == "left":
                    input_ids = input_ids[-(max_seq_len - force_eod_length):]
                    loss_mask_tensor = loss_mask_tensor[-(max_seq_len - force_eod_length):]
                elif self.truncation == "right":
                    input_ids = input_ids[:max_seq_len - force_eod_length]
                    loss_mask_tensor = loss_mask_tensor[:max_seq_len - force_eod_length]
                elif self.truncation == "error":
                    raise ValueError(f"{sequence_length=} is larger than {max_seq_len - force_eod_length=}")
                else:
                    raise ValueError(f"Unknown truncation method {self.truncation}")
            
            # Padding similar to original SFTDataset
            num_tokens = len(input_ids) + force_eod_length
            padding_len = max_seq_len - num_tokens
            assert padding_len >= 0
            filler = [tokenizer.eod] * force_eod_length + [tokenizer.pad] * (padding_len + 1)
            
            tokens = np.array(input_ids.tolist() + filler, dtype=np.int64)
            target = np.array(input_ids.tolist() + filler, dtype=np.int64)
            
            # Apply loss mask to target
            loss_filler = [IGNORE_INDEX] * len(filler)
            target_with_mask = []
            for i, (token, mask) in enumerate(zip(input_ids.tolist(), loss_mask_tensor.tolist())):
                if mask == 0:
                    target_with_mask.append(IGNORE_INDEX)
                else:
                    target_with_mask.append(token)
            target = np.array(target_with_mask + loss_filler, dtype=np.int64)
            
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
            
        elif self.pad_mode == "left_right":
            # Left-right padding mode (for RL compatibility)
            assert self.truncation == "error", "Only support error truncation for left_right pad mode"
            
            # Get prompt tokens using the underlying tokenizer
            prompt_str = tokenizer._tokenizer.apply_chat_template(
                messages[:prompt_message_length],
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                **self.apply_chat_template_kwargs,
            )
            prompt_tokens = tokenizer._tokenizer.encode(prompt_str, add_special_tokens=False)
            prompt_length = len(prompt_tokens)
            
            # Split into prompt and response parts
            prompt_ids = input_ids[:prompt_length].unsqueeze(0)
            prompt_attention_mask = attention_mask_tensor[:prompt_length].unsqueeze(0)
            prompt_loss_mask = loss_mask_tensor[:prompt_length].unsqueeze(0)
            
            response_ids = input_ids[prompt_length:].unsqueeze(0)
            response_attention_mask = attention_mask_tensor[prompt_length:].unsqueeze(0)
            response_loss_mask = loss_mask_tensor[prompt_length:].unsqueeze(0)
            
            # Ensure prompt loss mask is all zeros
            assert prompt_loss_mask.sum().item() == 0
            
            # Apply padding using helper functions
            pad_token_id = tokenizer.pad if hasattr(tokenizer, 'pad') else 0
            
            # Use helper function for prompt padding (left pad)
            prompt_ids, prompt_attention_mask = self._postprocess_data(
                input_ids=prompt_ids,
                attention_mask=prompt_attention_mask,
                max_length=self.max_prompt_length,
                pad_token_id=pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )
            
            # Use helper function for response padding (right pad)  
            response_ids, response_attention_mask = self._postprocess_data(
                input_ids=response_ids,
                attention_mask=response_attention_mask,
                max_length=self.max_response_length,
                pad_token_id=pad_token_id,
                left_pad=False,
                truncation=self.truncation,
            )
            
            # Right pad response loss mask to match response length
            if response_loss_mask.shape[1] < self.max_response_length:
                response_pad_len = self.max_response_length - response_loss_mask.shape[1]
                response_pad_loss = torch.zeros((1, response_pad_len), dtype=response_loss_mask.dtype)
                response_loss_mask = torch.cat((response_loss_mask, response_pad_loss), dim=1)
            elif response_loss_mask.shape[1] > self.max_response_length:
                if self.truncation == "error":
                    raise ValueError(f"Response loss mask length {response_loss_mask.shape[1]} > max_response_length {self.max_response_length}")
                response_loss_mask = response_loss_mask[:, :self.max_response_length]
            
            # Flatten tensors
            prompt_ids = prompt_ids[0]
            prompt_attention_mask = prompt_attention_mask[0]
            response_ids = response_ids[0]
            response_attention_mask = response_attention_mask[0]
            response_loss_mask = response_loss_mask[0]
            
            # Combine prompt and response
            input_ids = torch.cat((prompt_ids, response_ids), dim=0)
            attention_mask_combined = torch.cat((prompt_attention_mask, response_attention_mask), dim=0)
            
            # Compute position ids with mask
            position_ids = torch.cumsum(attention_mask_combined.long(), dim=0) - 1
            position_ids = torch.clamp(position_ids, min=0)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask_combined,
                "position_ids": position_ids,
                "responses": response_ids,
                "response_mask": response_loss_mask,
            }
        
        else:
            raise ValueError(f"Unknown pad_mode: {self.pad_mode}")

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
    
    def _compute_position_id_with_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute position IDs with attention mask (similar to VERL's implementation)."""
        position_ids = torch.cumsum(attention_mask.long(), dim=0) - 1
        position_ids = torch.clamp(position_ids, min=0)
        return position_ids
    
    def _postprocess_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int,
        pad_token_id: int,
        left_pad: bool = False,
        truncation: str = "error"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Post-process data with padding and truncation (similar to VERL's implementation)."""
        if input_ids.shape[1] < max_length:
            # Pad sequence
            pad_len = max_length - input_ids.shape[1]
            pad_ids = torch.full((input_ids.shape[0], pad_len), pad_token_id, dtype=input_ids.dtype)
            pad_mask = torch.zeros((attention_mask.shape[0], pad_len), dtype=attention_mask.dtype)
            
            if left_pad:
                input_ids = torch.cat((pad_ids, input_ids), dim=1)
                attention_mask = torch.cat((pad_mask, attention_mask), dim=1)
            else:
                input_ids = torch.cat((input_ids, pad_ids), dim=1)
                attention_mask = torch.cat((attention_mask, pad_mask), dim=1)
                
        elif input_ids.shape[1] > max_length:
            if truncation == "error":
                raise ValueError(f"Sequence length {input_ids.shape[1]} > max_length {max_length}")
            elif truncation == "left":
                input_ids = input_ids[:, -max_length:]
                attention_mask = attention_mask[:, -max_length:]
            elif truncation == "right":
                input_ids = input_ids[:, :max_length]
                attention_mask = attention_mask[:, :max_length]
        
        return input_ids, attention_mask
