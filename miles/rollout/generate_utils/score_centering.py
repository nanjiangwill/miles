import math
from collections.abc import Mapping, Sequence

from miles.utils.score_centering import RolloutScoreCenteringHead, score_centering_enabled
from miles.utils.types import Sample


def score_centering_request_fields(
    args,
    sampling_params: Mapping[str, object],
    *,
    evaluation: bool,
    openai: bool = False,
) -> dict[str, object]:
    """Return opt-in SGLang fields after validating the behavior policy."""
    if evaluation or not score_centering_enabled(args):
        return {}

    temperature = float(_value_or_default(sampling_params, "temperature", args.rollout_temperature))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("score centering requires stochastic sampling with temperature > 0")
    if temperature != float(args.rollout_temperature):
        raise ValueError(
            f"request temperature {temperature} does not match --rollout-temperature {args.rollout_temperature}"
        )
    if int(_value_or_default(sampling_params, "beam_width", 1)) != 1:
        raise ValueError("score centering requires independently sampled actions and does not support beam search")

    if args.use_sampling_support_replay:
        return {"return_sampling_support_logprobs": True}

    if temperature != 1.0:
        raise ValueError(
            "modeled-tail score centering requires temperature=1 until SGLang exposes an explicit behavior-head API"
        )

    _validate_unreplayed_logit_transforms(sampling_params)
    top_p = float(_value_or_default(sampling_params, "top_p", 1.0))
    top_k = int(_value_or_default(sampling_params, "top_k", -1))
    min_p = float(_value_or_default(sampling_params, "min_p", 0.0))
    if (top_p, top_k, min_p) != (1.0, -1, 0.0):
        raise ValueError("score centering without sampling replay requires top_p=1, top_k=-1, and min_p=0")
    top_logprobs_field = "top_logprobs" if openai else "top_logprobs_num"
    return {top_logprobs_field: int(args.score_centering_head_size)}


def append_score_centering_metadata(
    sample: Sample,
    output_token_ids: Sequence[int],
    meta_info: Mapping[str, object],
    *,
    head_size: int,
) -> None:
    """Append a validated sampler head from one SGLang response."""
    count = len(output_token_ids)
    if count == 0:
        _append_head(sample, RolloutScoreCenteringHead(ids=[], offsets=[0], logprobs=[]))
        return

    rows = meta_info.get("output_top_logprobs")
    if not isinstance(rows, Sequence) or len(rows) != count:
        raise ValueError(f"score-centering sampler head must contain {count} output rows")

    id_rows = []
    logprob_rows = []
    for row in rows:
        if not isinstance(row, Sequence) or len(row) != head_size:
            raise ValueError(f"every score-centering sampler-head row must contain {head_size} entries")
        if any(not isinstance(entry, Sequence) or len(entry) < 2 for entry in row):
            raise ValueError("score-centering sampler-head entries must contain a logprob and token id")
        logprob_rows.append([float(entry[0]) for entry in row])
        id_rows.append([int(entry[1]) for entry in row])
    _append_head(sample, RolloutScoreCenteringHead.from_rows(id_rows, logprob_rows))


def append_forced_score_centering_tokens(sample: Sample, token_ids: Sequence[int]) -> None:
    head = RolloutScoreCenteringHead.from_rows(
        [[int(token_id)] for token_id in token_ids],
        [[0.0] for _ in token_ids],
    )
    _append_head(sample, head)


def merge_score_centering_heads(
    first: Sample,
    observation_token_ids: Sequence[int],
    second: Sample,
) -> RolloutScoreCenteringHead | None:
    first_head = first.rollout_score_centering_head
    second_head = second.rollout_score_centering_head
    if first_head is None or second_head is None:
        if first_head is None and second_head is None:
            return None
        raise ValueError("cannot merge samples unless both turns carry complete score-centering heads")
    observation_head = RolloutScoreCenteringHead.from_rows(
        [[int(token_id)] for token_id in observation_token_ids],
        [[0.0] for _ in observation_token_ids],
    )
    return RolloutScoreCenteringHead.concatenate((first_head, observation_head, second_head))


def _validate_unreplayed_logit_transforms(sampling_params: Mapping[str, object]) -> None:
    defaults = {
        "frequency_penalty": (0, 0.0, None),
        "presence_penalty": (0, 0.0, None),
        "repetition_penalty": (1, 1.0, None),
        "logit_bias": ({}, None),
    }
    for name, allowed in defaults.items():
        if sampling_params.get(name) not in allowed:
            raise ValueError(f"{name} is not supported with score centering")
    for name in (
        "json_schema",
        "regex",
        "ebnf",
        "structural_tag",
        "response_format",
        "tools",
        "min_new_tokens",
        "min_tokens",
        "custom_logit_processor",
        "custom_params",
    ):
        if sampling_params.get(name):
            raise ValueError(f"constrained sampling ({name}) is not supported with score centering")


def _value_or_default(sampling_params: Mapping[str, object], name: str, default: object) -> object:
    value = sampling_params.get(name)
    return default if value is None else value


def _append_head(sample: Sample, head: RolloutScoreCenteringHead) -> None:
    if sample.rollout_score_centering_head is None:
        if sample.response_length != 0:
            raise ValueError("cannot initialize score-centering data after response tokens were appended")
        sample.rollout_score_centering_head = head
        return
    if len(sample.rollout_score_centering_head) != sample.response_length:
        raise ValueError("score-centering head is not aligned with response_length before appending")
    sample.rollout_score_centering_head = RolloutScoreCenteringHead.concatenate(
        (sample.rollout_score_centering_head, head)
    )
