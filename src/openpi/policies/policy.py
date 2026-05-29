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
            # JAX model setup. The activation-probing flags select branches in
            # sample_actions, so they must be static jit arguments; they default
            # to False and are otherwise inert.
            self._sample_actions = nnx_utils.module_jit(
                model.sample_actions,
                static_argnames=("return_prefix_activations", "return_action_activations"),
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
        # When an activation-probing flag is set, sample_actions returns a tuple of
        # (actions, activations) where activations is a dict with keys among
        # {"prefix", "action"}. The activations have a leading depth axis and so are
        # handled separately from the per-element batch stripping below.
        activations: dict[str, Any] = {}
        if isinstance(sample_output, tuple):
            actions, activations = sample_output
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
        for key, value in activations.items():
            # shape (depth, batch, width) -> drop batch -> (depth, width)
            outputs[f"{key}_activations"] = np.asarray(value[:, 0, :])
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
    """Wraps a policy to capture per-layer activations to disk.

    On each `infer()` call (when `capture=True`), the underlying policy is run
    with `return_prefix_activations=True` (and `return_action_activations=True`
    when `capture_action=True`). The resulting mean-pooled, per-layer hidden
    states are saved as `{save_dir}/act_{step_id:05d}.npz`:

    - prefix only: a single key "hidden" with shape (depth, width) == (18, 2048),
      preserved for backward compatibility.
    - prefix + action: two keys "hidden_prefix" (18, 2048) and "hidden_action"
      (18, 1024). This is the preferred format going forward.

    `step_id` is a monotonic counter incremented once per saved inference across
    the whole server lifetime (NOT reset per episode); it is the join key for the
    client-side ground-truth logger. When `capture=False` the wrapper is fully
    transparent and forwards `infer()` unchanged.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        save_dir: str,
        *,
        capture: bool = False,
        capture_action: bool = False,
    ):
        self._policy = policy
        self._capture = capture
        self._capture_action = capture_action
        self._step_id = 0
        self._save_dir = pathlib.Path(save_dir)

        if self._capture:
            logging.info(
                f"Capturing activations to: {save_dir} (action expert: {capture_action})"
            )
            self._save_dir.mkdir(parents=True, exist_ok=True)
            # Request activations from the underlying Policy on every infer().
            existing_kwargs = getattr(self._policy, "_sample_kwargs", None)
            if existing_kwargs is None:
                raise ValueError("ActivationCapturingPolicy must wrap a JAX `Policy` that supports sample_kwargs.")
            existing_kwargs["return_prefix_activations"] = True
            if self._capture_action:
                existing_kwargs["return_action_activations"] = True

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)
        if self._capture:
            prefix = results.pop("prefix_activations", None)
            action = results.pop("action_activations", None)
            if prefix is not None or action is not None:
                output_path = self._save_dir / f"act_{self._step_id:05d}.npz"
                if action is not None:
                    # Preferred dual format (prefix backbone + action expert).
                    np.savez(output_path, hidden_prefix=np.asarray(prefix), hidden_action=np.asarray(action))
                else:
                    # Backward-compatible single-key prefix-only format.
                    np.savez(output_path, hidden=np.asarray(prefix))
                self._step_id += 1
        return results

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata
