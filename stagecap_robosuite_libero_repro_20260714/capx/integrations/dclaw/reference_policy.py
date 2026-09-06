from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np


class _DummyDimension:
    def __init__(self, value: int | None = None) -> None:
        self.value = value


class _DummyTensorShape:
    def __init__(self, dims: list[_DummyDimension] | None = None) -> None:
        self.dims = dims or []


class _DummyFeedforwardGaussianPolicy:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_args = args
        self.init_kwargs = kwargs

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)


class DClawReferencePolicy:
    """Numpy-only loader for ROBEL bundled softlearning D'Claw policies."""

    def __init__(self, weights: list[np.ndarray], action_mode: str = "mean_scaled_tanh", scale: float = 0.5) -> None:
        self.weights = [np.asarray(weight, dtype=np.float32) for weight in weights]
        self.action_mode = action_mode
        self.scale = float(scale)
        if len(self.weights) != 6:
            raise ValueError(f"Expected 6 MLP weights, got {len(self.weights)}")

    @classmethod
    def from_robel_pickle(
        cls,
        env_id: str,
        *,
        action_mode: str = "mean_scaled_tanh",
        scale: float = 0.5,
    ) -> "DClawReferencePolicy":
        path = _default_policy_path(env_id)
        data = _load_policy_pickle_without_softlearning(path)
        weights = data.get("weights")
        if not isinstance(weights, list):
            raise ValueError(f"ROBEL policy pickle has no weight list: {path}")
        return cls(weights, action_mode=action_mode, scale=scale)

    def action(self, obs: np.ndarray) -> np.ndarray:
        """Return a deterministic 9-DoF action for a ROBEL flattened observation."""
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        w0, b0, w1, b1, w2, b2 = self.weights
        hidden = np.maximum(obs @ w0 + b0, 0.0)
        hidden = np.maximum(hidden @ w1 + b1, 0.0)
        output = hidden @ w2 + b2
        mean = output[:9]
        log_std = output[9:]
        if self.action_mode == "mean_clip":
            action = mean
        elif self.action_mode == "mean_tanh":
            action = np.tanh(mean)
        elif self.action_mode == "mean_scaled_clip":
            action = self.scale * mean
        elif self.action_mode == "mean_scaled_tanh":
            action = np.tanh(self.scale * mean)
        elif self.action_mode == "log_std_tanh":
            action = np.tanh(log_std)
        else:
            raise ValueError(f"Unsupported DClaw reference action mode: {self.action_mode}")
        return np.clip(action, -1.0, 1.0).astype(np.float32)


def _default_policy_path(env_id: str) -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "external" / "robel" / "robel" / "scripts" / "data" / "softlearning" / f"{env_id}-policy.pkl"


def _load_policy_pickle_without_softlearning(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    _install_pickle_stubs()
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected ROBEL policy pickle type: {type(data)!r}")
    return data


def _install_pickle_stubs() -> None:
    gaussian = types.ModuleType("softlearning.policies.gaussian_policy")
    gaussian.FeedforwardGaussianPolicy = _DummyFeedforwardGaussianPolicy
    tensor_shape = types.ModuleType("tensorflow.python.framework.tensor_shape")
    tensor_shape.Dimension = _DummyDimension
    tensor_shape.TensorShape = _DummyTensorShape

    for name in [
        "softlearning",
        "softlearning.policies",
        "tensorflow",
        "tensorflow.python",
        "tensorflow.python.framework",
    ]:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["softlearning.policies.gaussian_policy"] = gaussian
    sys.modules["tensorflow.python.framework.tensor_shape"] = tensor_shape
