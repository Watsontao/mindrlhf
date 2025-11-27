"""Segment rollout orchestration helpers."""
from __future__ import annotations

from typing import Dict

from mindformers import logger


class SegmentRolloutManager:
    """
    Manage unfinished / experience pools for the segment rollout algorithm.

    The full segment rollout flow will be integrated progressively.
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

    def get_metrics(self) -> Dict[str, float]:
        """
        Return diagnostic metrics captured during the most recent rollout.

        The dictionary is kept empty for now so downstream reporting code does not
        need conditional guards. Future revisions will populate this map.
        """

        return self.metrics.copy()

    def set_metrics(self, **kwargs):
        """Record diagnostic metrics for downstream logging."""
        self.metrics = dict(kwargs)
