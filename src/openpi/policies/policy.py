from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup. `return_prefix_activations` (used for activation
            # probing) is a Python bool that selects a branch, so it must be a
            # static jit argument; it defaults to False and is otherwise inert.
            self._sample_actions = nnx_utils.module_jit(
                model.sample_actions, static_argnames=("return_prefix_activations",)
            )
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        sample_output = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        # When `return_prefix_activations=True`, sample_actions returns a tuple of
        # (actions, prefix_activations). The activations have a leading depth axis
        # and so are handled separately from the per-element batch stripping below.
        prefix_activations = None
        if isinstance(sample_output, tuple):
            actions, prefix_activations = sample_output
        else:
            actions = sample_output
        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        if prefix_activations is not None:
            # shape (depth, batch, width) -> drop batch -> (depth, width)
            outputs["prefix_activations"] = np.asarray(prefix_activations[:, 0, :])
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results


class ActivationCapturingPolicy(_base_policy.BasePolicy):
    """Wraps a policy to capture per-layer prefix activations to disk.

    On each `infer()` call (when `capture=True`), the underlying policy is run
    with `return_prefix_activations=True`, and the resulting mean-pooled,
    per-layer PaliGemma-backbone hidden states are saved as
    `{save_dir}/act_{step_id:05d}.npz` under the key "hidden" with shape
    (depth, width) == (18, 2048). When `capture=False` the wrapper is fully
    transparent and forwards `infer()` unchanged.
    """

    def __init__(self, policy: _base_policy.BasePolicy, save_dir: str, *, capture: bool = False):
        self._policy = policy
        self._capture = capture
        self._step_id = 0
        self._save_dir = pathlib.Path(save_dir)

        if self._capture:
            logging.info(f"Capturing prefix activations to: {save_dir}")
            self._save_dir.mkdir(parents=True, exist_ok=True)
            # Request activations from the underlying Policy on every infer().
            existing_kwargs = getattr(self._policy, "_sample_kwargs", None)
            if existing_kwargs is None:
                raise ValueError("ActivationCapturingPolicy must wrap a JAX `Policy` that supports sample_kwargs.")
            existing_kwargs["return_prefix_activations"] = True

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)
        if self._capture:
            hidden = results.pop("prefix_activations", None)
            if hidden is not None:
                output_path = self._save_dir / f"act_{self._step_id:05d}.npz"
                np.savez(output_path, hidden=np.asarray(hidden))
                self._step_id += 1
        return results

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata
