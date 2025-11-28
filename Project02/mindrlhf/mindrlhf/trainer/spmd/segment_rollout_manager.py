"""Segment rollout orchestration helpers."""
from __future__ import annotations

from typing import Dict, List
import numpy as np

from mindformers import logger


class SegmentRolloutManager:
    """
    Manage unfinished / experience pools for the segment rollout algorithm.
    """

    def __init__(self, rl_config, sampling_config):
        self.rl_config = rl_config
        self.sampling_config = sampling_config
        self.enable_segment_rollout = bool(getattr(rl_config, "enable_segment_rollout", False))
        global_max = getattr(rl_config, "global_max_decode_len", 0)
        self.max_response_tokens = getattr(sampling_config, "max_tokens", 0)
        if global_max <= 0:
            global_max = self.max_response_tokens
        self.global_max_decode_len = int(global_max)
        max_segment_count = max(1, getattr(rl_config, "max_segment_count", 1))
        self.max_segment_count = max_segment_count
        self.segment_len = max(1, self.global_max_decode_len // max_segment_count)
        self.segment_schedule = getattr(rl_config, "segment_schedule", None)
        self.importance_sampling_mode = getattr(rl_config, "importance_sampling_mode", "legacy")
        self.pad_token_id = getattr(sampling_config, "pad_token_id", 0)
        self.bos_token_id = getattr(sampling_config, "bos_token_id", self.pad_token_id)
        eos_ids = getattr(sampling_config, "eos_token_id", None)
        if eos_ids is None:
            self.eos_token_ids = set()
        elif isinstance(eos_ids, (list, tuple)):
            self.eos_token_ids = set(int(eos) for eos in eos_ids)
        else:
            self.eos_token_ids = {int(eos_ids)}
        self.seq_length = getattr(rl_config, "seq_length", 0)
        self.prompt_window = max(0, self.seq_length - self.max_response_tokens)
        self.metrics: Dict[str, float] = {}
        self.samples: List[Dict] = []
        self.unfinished_ids: List[int] = []
        self.ready_ids: List[int] = []
        self.segment_calls = 0
        self.truncated_count = 0

        logger.info(
            "SegmentRolloutManager init: enable=%s global_max=%s segment_len=%s count=%s importance_mode=%s",
            self.enable_segment_rollout,
            self.global_max_decode_len,
            self.segment_len,
            self.max_segment_count,
            self.importance_sampling_mode,
        )

    @property
    def enabled(self) -> bool:
        """Return True if segment rollout is switched on."""
        return self.enable_segment_rollout

    def current_segment_len(self, segment_index: int) -> int:
        """Derive the target segment length for the current rollout loop."""
        if self.segment_schedule:
            safe_index = min(segment_index, len(self.segment_schedule) - 1)
            scheduled = int(self.segment_schedule[safe_index])
            if scheduled > 0:
                return scheduled
        return self.segment_len

    def decode(self, prompt_tensors, infer_callback, seq_length):
        """Run multi-segment decoding with unfinished/ready pools."""
        if not self.enable_segment_rollout:
            raise RuntimeError("Segment rollout decode called while disabled.")
        self.seq_length = seq_length
        self._initialize_samples(prompt_tensors)
        segment_index = 0
        while self.unfinished_ids and segment_index < self.max_segment_count:
            segment_len = self.current_segment_len(segment_index)
            batch_inputs = self._build_input_batch()
            results = infer_callback(batch_inputs, segment_len)
            self._process_outputs(results)
            segment_index += 1
        if self.unfinished_ids:
            self._truncate_remaining()
        generated_lengths = [sample["generated_len"] for sample in self.samples]
        mean_completion_len = float(np.mean(generated_lengths)) if generated_lengths else 0.0
        truncated_ratio = (
            float(self.truncated_count) / float(len(self.samples)) if self.samples else 0.0
        )
        segments_used = min(segment_index, self.max_segment_count)
        self.set_metrics(
            segments_used=segments_used,
            truncated_ratio=truncated_ratio,
            mean_completion_len=mean_completion_len,
        )
        return self._build_outputs()

    def get_metrics(self) -> Dict[str, float]:
        """Return diagnostic metrics captured during the most recent rollout."""
        return self.metrics.copy()

    def set_metrics(self, **kwargs):
        """Record diagnostic metrics for downstream logging."""
        self.metrics = dict(kwargs)

    def _initialize_samples(self, prompt_tensors):
        total = prompt_tensors.shape[0]
        self.samples = []
        for idx in range(total):
            prompt = prompt_tensors[idx]
            trimmed = self._trim_prompt(prompt)
            self.samples.append(
                {
                    "prompt": trimmed,
                    "generated": [],
                    "generated_len": 0,
                    "finished": False,
                    "truncated": False,
                }
            )
        self.unfinished_ids = list(range(total))
        self.ready_ids = []
        self.segment_calls = 0
        self.truncated_count = 0

    def _trim_prompt(self, prompt):
        valid = prompt[prompt != self.pad_token_id]
        if valid.size == 0:
            return np.array([self.bos_token_id], dtype=np.int32)
        return valid.astype(np.int32)

    def _build_input_batch(self):
        batch = np.full((len(self.samples), self.seq_length), self.pad_token_id, dtype=np.int32)
        for idx, sample in enumerate(self.samples):
            prompt = sample["prompt"]
            generated = (
                np.array(sample["generated"], dtype=np.int32)
                if sample["generated"]
                else np.array([], dtype=np.int32)
            )
            if sample["finished"]:
                tokens = prompt
            elif generated.size == 0:
                tokens = prompt
            else:
                tokens = np.concatenate((prompt, generated))
            if tokens.size > self.seq_length:
                tokens = tokens[-self.seq_length :]
            if tokens.size == 0:
                tokens = np.array([self.bos_token_id], dtype=np.int32)
            batch[idx, : tokens.size] = tokens
        return batch

    def _process_outputs(self, results):
        responses, responses_mask, _, _ = results
        for sample_id, sample in enumerate(self.samples):
            if sample["finished"]:
                continue
            response = responses[sample_id]
            mask = responses_mask[sample_id]
            segment_tokens = response[mask == 1]
            if segment_tokens.size == 0:
                continue
            remaining = self.global_max_decode_len - sample["generated_len"]
            if remaining <= 0:
                sample["finished"] = True
                sample["truncated"] = True
                self.ready_ids.append(sample_id)
                continue
            usable = segment_tokens[:remaining]
            eos_hit = False
            to_append = []
            for token in usable:
                token_int = int(token)
                if token_int in self.eos_token_ids:
                    eos_hit = True
                    break
                to_append.append(token_int)
            if to_append:
                sample["generated"].extend(to_append)
                sample["generated_len"] += len(to_append)
            if eos_hit or sample["generated_len"] >= self.global_max_decode_len:
                if sample["generated_len"] >= self.global_max_decode_len and not eos_hit:
                    sample["truncated"] = True
                    self.truncated_count += 1
                sample["finished"] = True
                self.ready_ids.append(sample_id)
        self.unfinished_ids = [idx for idx, sample in enumerate(self.samples) if not sample["finished"]]

    def _truncate_remaining(self):
        for sample_id in self.unfinished_ids:
            sample = self.samples[sample_id]
            sample["finished"] = True
            sample["truncated"] = True
            self.truncated_count += 1
            self.ready_ids.append(sample_id)
        self.unfinished_ids = []

    def _build_outputs(self):
        num_samples = len(self.samples)
        right_padding_responses = np.full(
            (num_samples, self.max_response_tokens), self.pad_token_id, dtype=np.int32
        )
        responses_mask = np.zeros((num_samples, self.max_response_tokens), dtype=np.int32)
        if self.prompt_window > 0:
            left_padding_prompts = np.full(
                (num_samples, self.prompt_window), self.pad_token_id, dtype=np.int32
            )
            prompts_mask = np.zeros((num_samples, self.prompt_window), dtype=np.int32)
        else:
            left_padding_prompts = np.zeros((num_samples, 0), dtype=np.int32)
            prompts_mask = np.zeros((num_samples, 0), dtype=np.int32)

        for idx, sample in enumerate(self.samples):
            generated = (
                np.array(sample["generated"], dtype=np.int32)
                if sample["generated"]
                else np.array([], dtype=np.int32)
            )
            generated = generated[: self.max_response_tokens]
            gen_len = generated.size
            if gen_len > 0:
                right_padding_responses[idx, :gen_len] = generated
                responses_mask[idx, :gen_len] = 1
            if self.prompt_window > 0:
                prompt = sample["prompt"]
                if prompt.size > self.prompt_window:
                    prompt_slice = prompt[-self.prompt_window :]
                else:
                    prompt_slice = prompt
                start = self.prompt_window - prompt_slice.size
                if prompt_slice.size > 0:
                    left_padding_prompts[idx, start:] = prompt_slice
                    prompts_mask[idx, start:] = 1

        return right_padding_responses, responses_mask, left_padding_prompts, prompts_mask
