"""K-frame sliding window for online inference [🟩 PEMTRS].

During training, K-frame sequences are sampled by the dataset directly.
During online inference, this buffer maintains the most recent K frames
and supplies them to the temporal selector each tick.

When fewer than K frames have been pushed, the buffer warms up by
replicating the first observed frame K times (per REACT guide §7).
"""
from collections import deque
from typing import Optional, Union

import numpy as np
import torch


class FrameBuffer:
    def __init__(self, K: int, device: Optional[torch.device] = None):
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        self.K = K
        self._device = device
        self._buf: deque = deque(maxlen=K)

    def reset(self) -> None:
        self._buf.clear()

    def push(self, frame: Union[torch.Tensor, np.ndarray]) -> None:
        f = frame if isinstance(frame, torch.Tensor) else torch.as_tensor(frame)
        f = f.detach()
        if self._device is not None:
            f = f.to(self._device)
        if not self._buf:
            for _ in range(self.K):
                self._buf.append(f.clone())
        else:
            self._buf.append(f)

    def get(self) -> torch.Tensor:
        """Return the K-frame window stacked along a new leading axis: (K, ...)."""
        if not self._buf:
            raise RuntimeError("FrameBuffer is empty; call push() at least once first")
        return torch.stack(list(self._buf), dim=0)

    def __len__(self) -> int:
        return len(self._buf)
