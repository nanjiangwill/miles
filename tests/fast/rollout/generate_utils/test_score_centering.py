from types import SimpleNamespace

import pytest

from miles.rollout.generate_utils.score_centering import (
    append_forced_score_centering_tokens,
    append_score_centering_metadata,
    score_centering_request_fields,
)
from miles.utils.types import Sample


def _args(**overrides):
    values = dict(
        rollout_temperature=1.0,
        rollout_top_k=-1,
        rollout_top_p=1.0,
        score_centering_head_size=2,
        use_sampling_support_replay=False,
        use_score_centering=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_disabled_and_evaluation_requests_add_no_fields():
    params = {"temperature": 1.0, "top_p": 1.0, "top_k": -1}

    assert score_centering_request_fields(_args(use_score_centering=False), params, evaluation=False) == {}
    assert score_centering_request_fields(_args(), params, evaluation=True) == {}


def test_modeled_tail_request_uses_existing_output_head():
    fields = score_centering_request_fields(
        _args(),
        {"temperature": 1.0, "top_p": 1.0, "top_k": -1},
        evaluation=False,
    )

    assert fields == {"top_logprobs_num": 2}


def test_openai_mode_uses_openai_top_logprobs_name():
    fields = score_centering_request_fields(
        _args(),
        {"temperature": 1.0, "top_p": 1.0, "top_k": -1},
        evaluation=False,
        openai=True,
    )

    assert fields["top_logprobs"] == 2
    assert "top_logprobs_num" not in fields


def test_sampling_replay_requests_support_aligned_logprobs():
    fields = score_centering_request_fields(
        _args(
            rollout_temperature=0.8,
            rollout_top_p=0.9,
            rollout_top_k=16,
            use_sampling_support_replay=True,
        ),
        {
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 16,
            "tools": [{"type": "function"}],
        },
        evaluation=False,
    )

    assert fields == {"return_sampling_support_logprobs": True}


def test_sampling_replay_rejects_beam_search():
    with pytest.raises(ValueError, match="beam search"):
        score_centering_request_fields(
            _args(
                rollout_temperature=0.8,
                rollout_top_p=0.9,
                rollout_top_k=16,
                use_sampling_support_replay=True,
            ),
            {"temperature": 0.8, "top_p": 0.9, "top_k": 16, "beam_width": 2},
            evaluation=False,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("top_p", 0.9),
        ("top_k", 16),
        ("min_p", 0.1),
        ("frequency_penalty", 0.2),
        ("logit_bias", {"1": 0.1}),
        ("regex", "[a-z]+"),
        ("tools", [{"type": "function"}]),
        ("min_new_tokens", 3),
        ("min_tokens", 3),
        ("custom_logit_processor", "serialized"),
        ("custom_params", {"scale": 0.5}),
    ],
)
def test_modeled_tail_rejects_unmodeled_sampler_transform(name, value):
    params = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, name: value}

    with pytest.raises(ValueError):
        score_centering_request_fields(_args(), params, evaluation=False)


def test_modeled_tail_rejects_nonunit_temperature_without_behavior_head_contract():
    with pytest.raises(ValueError, match="behavior-head API"):
        score_centering_request_fields(
            _args(rollout_temperature=0.8),
            {"temperature": 0.8, "top_p": 1.0, "top_k": -1},
            evaluation=False,
        )


def test_sampler_head_decodes_and_forced_tokens_are_singletons():
    sample = Sample()

    append_score_centering_metadata(
        sample,
        [4, 7],
        {
            "output_top_logprobs": [
                [[-0.4, 4, None], [-1.2, 2, None]],
                [[-0.6, 7, None], [-1.0, 1, None]],
            ],
        },
        head_size=2,
    )
    sample.tokens = [4, 7]
    sample.response_length = 2
    append_forced_score_centering_tokens(sample, [9])
    sample.tokens.append(9)
    sample.response_length += 1

    head_ids, head_logprobs, lengths = sample.rollout_score_centering_head._select_rows(range(3))
    assert head_ids.tolist() == [4, 2, 7, 1, 9]
    assert lengths.tolist() == [2, 2, 1]
    assert head_logprobs[-1].item() == 0.0
    sample.validate()


def test_zero_token_response_builds_an_empty_head_without_payload_arrays():
    sample = Sample()

    append_score_centering_metadata(sample, [], {}, head_size=2)

    assert len(sample.rollout_score_centering_head) == 0


def test_sampler_head_requires_complete_output_rows():
    with pytest.raises(ValueError, match="output rows"):
        append_score_centering_metadata(
            Sample(),
            [4],
            {},
            head_size=1,
        )
