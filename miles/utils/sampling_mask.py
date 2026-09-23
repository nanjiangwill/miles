import math
from array import array
from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field

import torch


@dataclass(frozen=True, eq=False)
class RolloutSamplingMask:
    """One sample's sampling mask: for each response position, the token ids
    the rollout sampler could emit.

    Stored in CSR form so it stays two flat integer arrays end to end:
    ``ids`` is ``[total_support_size]`` (all supports concatenated), ``offsets``
    is ``[num_response_tokens + 1]``, and token ``t``'s support is
    ``ids[offsets[t] : offsets[t + 1]]``. That is the shape object-store
    transport needs, so no per-token nesting is rebuilt on the trainer side.
    Optional ``support_logprobs`` stores the rollout sampler's normalized
    behavior distribution in the same flattened order.
    """

    ids: InitVar[Sequence[int] | torch.Tensor]
    offsets: InitVar[Sequence[int] | torch.Tensor]
    support_logprobs: InitVar[Sequence[float] | torch.Tensor | None] = None
    _ids: torch.Tensor = field(init=False, repr=False)
    _offsets: torch.Tensor = field(init=False, repr=False)
    _support_logprobs: torch.Tensor | None = field(init=False, repr=False)

    def __post_init__(
        self,
        ids: Sequence[int] | torch.Tensor,
        offsets: Sequence[int] | torch.Tensor,
        support_logprobs: Sequence[float] | torch.Tensor | None,
    ):
        owned_ids = _to_owned_cpu_integer_tensor(ids, dtype=torch.int32)
        owned_offsets = _to_owned_cpu_integer_tensor(offsets, dtype=torch.long)
        if owned_offsets.numel() == 0 or owned_offsets[0] != 0 or owned_offsets[-1] != owned_ids.numel():
            raise ValueError("sampling-mask offsets must start at zero and end at the flattened id count")
        if torch.any(owned_offsets[1:] <= owned_offsets[:-1]):
            raise ValueError(
                "sampling-mask offsets must be strictly increasing: "
                "every response token needs a non-empty sampling mask"
            )
        owned_support_logprobs = None if support_logprobs is None else _to_owned_cpu_float_tensor(support_logprobs)
        if owned_support_logprobs is not None:
            _validate_normalized_rows(owned_ids, owned_offsets, owned_support_logprobs)
        object.__setattr__(self, "_ids", owned_ids)
        object.__setattr__(self, "_offsets", owned_offsets)
        object.__setattr__(self, "_support_logprobs", owned_support_logprobs)

    @classmethod
    def from_mask_list(
        cls,
        mask_list: Sequence[Sequence[int]],
        support_logprob_rows: Sequence[Sequence[float]] | None = None,
    ) -> "RolloutSamplingMask":
        """Build from one mask (the allowed token ids) per response token.

        Args:
            mask_list: ragged ``[num_response_tokens][mask_size_t]``;
                ``mask_list[t]`` lists the token ids the sampler could emit at
                response position ``t``. SGLang's ``output_token_sampling_mask``
                arrives in this shape.
            support_logprob_rows: optional behavior logprobs aligned with
                ``mask_list`` and normalized within every response position.
        """
        if support_logprob_rows is not None and len(support_logprob_rows) != len(mask_list):
            raise ValueError("sampling-support logprob rows must align with support rows")

        ids = []
        support_logprobs = [] if support_logprob_rows is not None else None
        offsets = [0]
        for row_index, mask in enumerate(mask_list):
            ids.extend(mask)
            if support_logprobs is not None:
                row_logprobs = support_logprob_rows[row_index]
                if len(row_logprobs) != len(mask):
                    raise ValueError("sampling-support logprobs must align with support ids")
                support_logprobs.extend(row_logprobs)
            offsets.append(len(ids))
        return cls(ids=ids, offsets=offsets, support_logprobs=support_logprobs)

    @classmethod
    def concatenate(cls, masks: Sequence["RolloutSamplingMask"]) -> "RolloutSamplingMask":
        """Concatenate complete per-token supports in response order."""
        if not masks:
            return cls(ids=[], offsets=[0])
        if len(masks) == 1:
            return masks[0]

        has_support_logprobs = [mask._support_logprobs is not None for mask in masks]
        if any(has_support_logprobs) and not all(has_support_logprobs):
            raise ValueError("cannot concatenate sampling masks with incomplete support logprobs")

        ids = torch.cat([mask._ids for mask in masks])
        support_logprobs = torch.cat([mask._support_logprobs for mask in masks]) if all(has_support_logprobs) else None
        offsets = [torch.zeros(1, dtype=torch.long)]
        id_count = 0
        for mask in masks:
            offsets.append(mask._offsets[1:] + id_count)
            id_count += mask._ids.numel()
        return cls(ids=ids, offsets=torch.cat(offsets), support_logprobs=support_logprobs)

    def __len__(self) -> int:
        return self._offsets.numel() - 1

    def prefix(self, response_length: int) -> "RolloutSamplingMask":
        """Return the support for the first ``response_length`` tokens."""
        if not 0 <= response_length <= len(self):
            raise ValueError(f"sampling-mask prefix length must be in [0, {len(self)}]")
        if response_length == len(self):
            return self
        id_count = self._offsets[response_length]
        support_logprobs = None if self._support_logprobs is None else self._support_logprobs[:id_count]
        return type(self)(
            ids=self._ids[:id_count],
            offsets=self._offsets[: response_length + 1],
            support_logprobs=support_logprobs,
        )

    def _as_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Borrow the private CSR tensors for immediate read-only transport."""
        return self._ids, self._offsets

    def _as_distribution_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Borrow ids, offsets, and optional support logprobs for transport."""
        return self._ids, self._offsets, self._support_logprobs

    def _select_masks(self, token_indices: Sequence[int] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Flattened masks for the given response positions.

        Args:
            token_indices: ``[num_selected]`` response positions to read.

        Returns:
            ``(ids, lengths)`` where ``ids`` is ``[sum(lengths)]``, the selected
            masks concatenated in ``token_indices`` order, and ``lengths`` is
            ``[num_selected]``, each position's mask size.

        The returned ids may share the mask's private storage and are only for
        immediate read-only use by the scoring path.
        """
        if isinstance(token_indices, range) and token_indices.step == 1:
            if len(token_indices) == 0:
                return self._ids.new_empty(0), self._offsets.new_empty(0)
            if token_indices.start < 0 or token_indices.stop > len(self):
                raise ValueError(f"response indices must be in [0, {len(self)})")
            start, stop = token_indices.start, token_indices.stop
            lengths = self._offsets[start + 1 : stop + 1] - self._offsets[start:stop]
            return self._ids[self._offsets[start] : self._offsets[stop]], lengths

        indices = _to_cpu_integer_tensor(token_indices).to(torch.long)
        if torch.any(indices < 0) or torch.any(indices >= len(self)):
            raise ValueError(f"response indices must be in [0, {len(self)})")
        lengths = self._offsets[indices + 1] - self._offsets[indices]
        if indices.numel() == 0:
            return self._ids.new_empty(0), lengths
        run_starts = [0]
        run_starts.extend((torch.nonzero(indices[1:] != indices[:-1] + 1).flatten() + 1).tolist())
        run_starts.append(indices.numel())
        parts = [
            self._ids[self._offsets[indices[start]] : self._offsets[indices[end - 1] + 1]]
            for start, end in zip(run_starts[:-1], run_starts[1:], strict=True)
        ]
        return (parts[0] if len(parts) == 1 else torch.cat(parts)), lengths

    def _select_distribution(
        self,
        token_indices: Sequence[int] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return selected support ids, normalized logprobs, and row lengths."""
        if self._support_logprobs is None:
            raise ValueError("sampling-support logprobs were not captured")
        ids, lengths = self._select_masks(token_indices)
        logprobs, logprob_lengths = _select_ragged_values(self._support_logprobs, self._offsets, token_indices)
        if not torch.equal(lengths, logprob_lengths):
            raise RuntimeError("sampling-support ids and logprobs selected different row lengths")
        return ids, logprobs, lengths


def _to_owned_cpu_integer_tensor(
    values: Sequence[int] | torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        _validate_integer_tensor(values)
        return values.to(device="cpu", dtype=dtype, copy=True)

    if dtype == torch.int32 and len(values) > 0:
        storage = array("i", values)
        # frombuffer keeps this private backing array alive without copying it.
        return torch.frombuffer(storage, dtype=torch.int32)

    return torch.tensor(values, dtype=dtype, device="cpu")


def _to_cpu_integer_tensor(values: Sequence[int] | torch.Tensor) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        tensor = values.detach().cpu()
    elif len(values) == 0:
        tensor = torch.empty(0, dtype=torch.long, device="cpu")
    elif isinstance(values, range):
        tensor = torch.arange(values.start, values.stop, values.step, device="cpu")
    else:
        tensor = torch.as_tensor(values, device="cpu")
    _validate_integer_tensor(tensor)
    return tensor


def _to_owned_cpu_float_tensor(values: Sequence[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        if values.ndim != 1 or not torch.is_floating_point(values):
            raise ValueError("sampling-support logprobs must be a one-dimensional floating-point tensor")
        return values.detach().to(device="cpu", dtype=torch.float32, copy=True)
    if len(values) > 0:
        storage = array("f", values)
        return torch.frombuffer(storage, dtype=torch.float32)
    return torch.empty(0, dtype=torch.float32, device="cpu")


def _select_ragged_values(
    values: torch.Tensor,
    offsets: torch.Tensor,
    token_indices: Sequence[int] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = _to_cpu_integer_tensor(token_indices).to(torch.long)
    lengths = offsets[indices + 1] - offsets[indices]
    if indices.numel() == 0:
        return values.new_empty(0), lengths
    parts = [values[offsets[index] : offsets[index + 1]] for index in indices]
    return (parts[0] if len(parts) == 1 else torch.cat(parts)), lengths


def _validate_normalized_rows(ids: torch.Tensor, offsets: torch.Tensor, logprobs: torch.Tensor) -> None:
    if logprobs.numel() != ids.numel():
        raise ValueError("sampling-support logprobs must align with flattened support ids")
    if not torch.isfinite(logprobs).all() or torch.any(logprobs > 1e-6):
        raise ValueError("sampling-support logprobs must be finite normalized log probabilities")
    for start, end in zip(offsets[:-1].tolist(), offsets[1:].tolist(), strict=True):
        row_ids = ids[start:end]
        if torch.unique(row_ids).numel() != row_ids.numel():
            raise ValueError("sampling-mask ids must be distinct within each response row")
        mass = logprobs[start:end].double().exp().sum().item()
        if not math.isclose(mass, 1.0, rel_tol=1e-4, abs_tol=1e-6):
            raise ValueError("sampling-support logprobs must normalize to one within each response row")


def _validate_integer_tensor(tensor: torch.Tensor) -> None:
    if tensor.ndim != 1 or tensor.dtype == torch.bool or torch.is_floating_point(tensor) or torch.is_complex(tensor):
        raise ValueError("sampling-mask ids, offsets, and response indices must be one-dimensional integers")
