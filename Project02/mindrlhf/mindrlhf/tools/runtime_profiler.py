"""Runtime profiling utility for logging durations and memory usage."""
import json
import os
import re
import shutil
import subprocess
import time
from contextlib import contextmanager

import psutil
from mindformers import logger


def _convert_to_mb(value, unit):
    """Convert memory value with unit string into MB."""
    unit = unit.lower()
    if unit in ("mb", "mib"):
        return float(value)
    if unit in ("gb", "gib"):
        return float(value) * 1024.0
    if unit in ("kb", "kib"):
        return float(value) / 1024.0
    return float(value)


def _query_nvidia_smi():
    """Query GPU memory via nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return None
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=5)
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning(f"RuntimeProfiler: failed to run nvidia-smi: {exc}")
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    lines = result.stdout.strip().splitlines()
    used = []
    total = []
    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            try:
                used.append(float(parts[0]))
                total.append(float(parts[1]))
            except ValueError:
                continue
    if not used:
        return None
    return {
        "backend": "nvidia-smi",
        "device_count": len(used),
        "used_mb": sum(used),
        "total_mb": sum(total),
    }


def _query_npu_smi():
    """Query Ascend device memory via npu-smi."""
    if not shutil.which("npu-smi"):
        return None
    cmd = ["npu-smi", "info"]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=5)
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning(f"RuntimeProfiler: failed to run npu-smi: {exc}")
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    used_total_pattern = re.compile(
        r"Memory-Usage\s*:\s*([0-9.]+)\s*([KMG]i?B)\s*/\s*([0-9.]+)\s*([KMG]i?B)",
        re.IGNORECASE,
    )
    used_values = []
    total_values = []
    for line in result.stdout.splitlines():
        match = used_total_pattern.search(line)
        if not match:
            continue
        used_values.append(_convert_to_mb(match.group(1), match.group(2)))
        total_values.append(_convert_to_mb(match.group(3), match.group(4)))
    if not used_values:
        return None
    return {
        "backend": "npu-smi",
        "device_count": len(used_values),
        "used_mb": sum(used_values),
        "total_mb": sum(total_values),
    }


def _collect_gpu_memory():
    """Collect GPU/NPU memory usage if possible."""
    info = _query_nvidia_smi()
    if info:
        return info
    return _query_npu_smi()


class RuntimeProfiler:
    """Record timing and memory usage for critical phases."""

    def __init__(
        self,
        enable=False,
        log_dir="/tmp/",
        record_gpu_memory=True,
        record_host_memory=False,
        step_log_interval=0,
        epoch_log_interval=0,
        rank_id=None,
    ):
        self.enable = enable
        self.record_gpu_memory = record_gpu_memory
        self.record_host_memory = record_host_memory
        self.step_log_interval = max(0, int(step_log_interval or 0))
        self.epoch_log_interval = max(0, int(epoch_log_interval or 0))
        self.rank_id = rank_id
        self.active = bool(enable) and (rank_id in (None, 0))
        self._start_time = time.perf_counter()
        self._last_step_time = self._start_time
        self._last_epoch_time = self._start_time
        if self.active:
            os.makedirs(log_dir, exist_ok=True)
            self.log_path = os.path.join(log_dir, "runtime_profile.log")
            with open(self.log_path, "w", encoding="utf-8") as file:
                file.write("")
        else:
            self.log_path = None

    def _get_host_memory(self):
        if not self.record_host_memory:
            return None
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024.0 * 1024.0)
        return round(memory_mb, 3)

    def _get_gpu_memory(self):
        if not self.record_gpu_memory:
            return None
        return _collect_gpu_memory()

    def _log_entry(self, entry):
        if not self.active or not self.log_path:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as file:
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning(f"RuntimeProfiler: failed to append log entry: {exc}")

    @contextmanager
    def profile(self, phase_name, extra_info=None):
        """Context manager for profiling a phase."""
        if not self.active:
            yield
            return
        start_time = time.perf_counter()
        entry = {
            "phase": phase_name,
            "start_ts": time.time(),
            "extra": extra_info or {},
        }
        host_before = self._get_host_memory()
        gpu_before = self._get_gpu_memory()
        yield
        duration = time.perf_counter() - start_time
        host_after = self._get_host_memory()
        gpu_after = self._get_gpu_memory()
        entry["duration_s"] = round(duration, 6)
        if host_before is not None or host_after is not None:
            entry["host_memory_mb_before"] = host_before
            entry["host_memory_mb_after"] = host_after
        if gpu_before or gpu_after:
            entry["device_memory_before"] = gpu_before
            entry["device_memory_after"] = gpu_after
        self._log_entry(entry)

    def record_step(self, step_index):
        """Record runtime every N steps as configured."""
        if not self.active or self.step_log_interval <= 0:
            return
        if step_index <= 0:
            return
        if step_index % self.step_log_interval != 0:
            return
        current = time.perf_counter()
        duration = current - self._last_step_time
        entry = {
            "phase": "step_interval",
            "step": int(step_index),
            "interval": self.step_log_interval,
            "duration_s": round(duration, 6),
            "since_start_s": round(current - self._start_time, 6),
        }
        self._log_entry(entry)
        self._last_step_time = current

    def record_epoch(self, epoch_index):
        """Record runtime every N epochs as configured."""
        if not self.active or self.epoch_log_interval <= 0:
            return
        if epoch_index <= 0:
            return
        if epoch_index % self.epoch_log_interval != 0:
            return
        current = time.perf_counter()
        duration = current - self._last_epoch_time
        entry = {
            "phase": "epoch_interval",
            "epoch": int(epoch_index),
            "interval": self.epoch_log_interval,
            "duration_s": round(duration, 6),
            "since_start_s": round(current - self._start_time, 6),
        }
        self._log_entry(entry)
        self._last_epoch_time = current
