from argparse import Namespace


def top_p_sampling_replay_enabled(args: Namespace) -> bool:
    return float(getattr(args, "rollout_top_p", 1.0)) < 1.0
