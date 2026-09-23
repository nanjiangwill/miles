---
title: Score Centering
description: Remove off-policy score drift with an additive correction computed from sampler probabilities.
---

[Score centering](https://arxiv.org/abs/2609.20807) corrects the drift that
appears when rollouts come from a sampler distribution `q` but gradients use a
different trainer distribution `p`. At each token prefix, Miles replaces the
trainer score with

$$
\nabla \log p(y) - \mathbb{E}_{v \sim q}\left[\nabla \log p(v)\right].
$$

The expectation is an additive, prefix-local correction. It is distinct from
reward or advantage centering, and it composes with token-level importance
sampling. Miles implements the correction as an autograd scalar, so gradients
flow through trainer log probabilities while sampler probabilities and
importance weights remain stop-gradient coefficients.

## Enable score centering

```bash
ROLLOUT_ARGS+=(
  --use-score-centering
  --score-centering-head-size 128
  --use-miles-router
)
```

`--score-centering-head-size` controls how many of the sampler's
highest-probability tokens are retained for the modeled-tail approximation. It
does not truncate sampling and is unused by exact-support score centering.

Miles supports two modes and selects between them from the rollout sampling
configuration.

## Modeled-tail score centering

With untruncated sampling (`--rollout-top-p 1 --rollout-top-k -1`) at
`--rollout-temperature 1`, SGLang
returns the sampler's top probability head `H`. Miles evaluates the trainer on
the sampled token and the head, then models the omitted sampler tail with the
trainer distribution:

$$
\hat q(v) =
\begin{cases}
q(v), & v \in H, \\
\rho p(v), & v \notin H,
\end{cases}
\qquad
\rho = \frac{1 - q(H)}{1 - p(H)}.
$$

The score expectation reduces to a sum over `H`; Miles never materializes a
full-vocabulary log-softmax. The default head size is 128, matching the paper.
This mode uses SGLang's existing output top-logprob API and does not depend on
sampling-support probability metadata.

Modeled-tail mode currently requires unit temperature. SGLang's existing
top-logprob response can be switched globally between pre- and post-temperature
semantics, and a separately managed endpoint does not expose which mode it is
using. Miles therefore fails closed at non-unit temperature until SGLang has an
explicit behavior-head response contract. Exact-support mode is unaffected and
supports any positive configured temperature.

Because the omitted tail assumes no additional sampler-only logit transform,
modeled-tail mode rejects top-p, top-k, min-p, penalties, logit bias,
constraints, beam search, and custom logit processors. Tool and environment
tokens receive singleton behavior distributions, which makes their policy
gradient exactly zero.

## Exact-support score centering

Positive rollout top-k, or top-p below one, enables
[sampling-support replay](/advanced/sampling-support-replay). In this mode the
actual action space is the realized support `S`. SGLang returns every `q(v)` for
`v` in `S`, and Miles renormalizes trainer logits over the same support before
computing

$$
\sum_{v \in S} q(v)\,w(v)\,\log p(v).
$$

Support IDs alone are not enough: they let the trainer reconstruct `p` over
`S`, but not the sampler's relative probabilities `q`. Exact-support mode
therefore requires an SGLang build with `return_sampling_support_logprobs` in
addition to the existing `return_sampling_mask` primitive.

## Importance-sampling composition

Without `--use-tis`, every correction weight `w(v)` is one. With `--use-tis`,
Miles applies the same clipped token-level importance-weight function to the
sampled-token term and the score-centering expectation, as derived in the
paper. The built-in IcePop custom TIS function is also supported; arbitrary
custom correction functions are rejected because Miles cannot assume their
tail reduction is valid.

## Correctness and cost

Score centering currently supports `policy_loss` and token-level objectives,
not sequence-level GSPO. It is incompatible with OPSM, OPD, unbiased KL,
mismatch-metric recomputation, custom policy-loss reducers, and prefill logprob
recomputation. Sampling-support replay retains its additional restrictions,
including the current reference-KL and teacher-distillation limitation.

The default disabled path performs no extra capture or scoring. Modeled-tail
mode transports `head_size` IDs and float32 log probabilities per generated
token and scores that sparse head on the trainer. Exact-support mode transports
one float32 sampler log probability alongside every already-replayed support
ID. Both trainer paths use a sparse vocabulary-parallel autograd primitive;
they avoid storing a full-vocabulary probability tensor and support tensor and
context parallelism.

The following metrics make the approximation and off-policy correction
visible:

- `score_centering_sampler_head_mass`
- `score_centering_trainer_head_mass`
- `score_centering_sampled_token_weight`
- `train_rollout_logprob_abs_diff`
- `train_rollout_kl`
