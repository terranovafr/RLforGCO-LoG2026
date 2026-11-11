"""
Notebook-facing benchmark evaluation API built around LaGCORL's own test_utils.

Unlike a generic reward-only evaluator, this API calls
utils.test_utils.calculate_average_performances(), which uses the environment's
get_metrics() method and returns the benchmark-specific metrics defined by the
environment. It also supports the project's action-time benchmark.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


def _project_test_utils():
    """Import project utilities whether project root or its parent is on sys.path."""
    try:
        from utils.test_utils import (
            calculate_average_performances,
            calculate_average_action_time,
            summarize_df,
        )
    except ImportError:
        from LaGCORL.utils.test_utils import (
            calculate_average_performances,
            calculate_average_action_time,
            summarize_df,
        )
    try:
        from utils.math_utils import set_seeds
    except ImportError:
        from LaGCORL.utils.math_utils import set_seeds
    return calculate_average_performances, calculate_average_action_time, summarize_df, set_seeds


def test_agent_on_env(
    model,
    env,
    *,
    n_episodes: int = 20,
    seeds: Optional[list[int]] = None,
    wrapper_factory: Optional[Callable[[Any], Any]] = None,
    compare_random: bool = True,
    deterministic: bool = False,
    save_dir: Optional[str] = None,
    logger=None,
    verbose: int = 1,
    label: str = "agent",
):
    """
    Benchmark a trained agent on a fresh project environment.

    The benchmark metrics are taken from env.get_metrics() through the
    project's calculate_average_performances utility. This API expects a
    project environment whose completed episode exposes get_metrics().

    Parameters
    ----------
    model: trained SB3/LaGCORL-compatible model.
    env: fresh Gym/Gymnasium project environment (not the training instance).
    n_episodes: total episodes to evaluate.
    seeds: optional seed list; repeats cyclically if shorter than n_episodes.
    wrapper_factory: same environment wrapper used during training, if required.
    compare_random: include the project's random-agent baseline.
    deterministic: retained for API clarity. The current project benchmark
        helper selects actions with deterministic=False, matching test_utils.
        To change this, update play_agent_episode_until_done in test_utils.
    save_dir: optional directory for benchmark CSV/JSON outputs.
    logger: optional logger; a default notebook logger is created if omitted.
    verbose: verbosity forwarded to project test utilities.
    label: descriptive name added to exported result metadata.

    Returns a dict containing the per-episode benchmark DataFrame, metric lists,
    certainty statistics, and aggregate summaries.
    """
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    if not hasattr(env, "get_metrics"):
        raise TypeError(
            "env must expose get_metrics() for LaGCORL benchmark metrics. "
            "For custom environments, implement get_metrics() or use the "
            "generic evaluate_agent helper instead."
        )

    (calculate_average_performances, calculate_average_action_time,
     summarize_df, set_seeds) = _project_test_utils()

    try:
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise ImportError("Install stable-baselines3 to use the benchmark testing API.") from exc

    logger = logger or logging.getLogger("notebook_test_api")
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    eval_env = wrapper_factory(env) if wrapper_factory is not None else env
    # Do not wrap in SB3 Monitor: the project's test_utils directly calls
    # env.envs[0].get_metrics(), and Monitor does not reliably expose custom
    # environment methods across SB3/Gymnasium versions.
    vec_env = DummyVecEnv([lambda: eval_env])
    # Make iDQN obtain candidate actions from the evaluation environment.
    model.set_env(vec_env)
    # The CLI's test path applies VecNormalize when training used normalized
    # observations. Do not silently add it here: callers must provide a correctly
    # normalized environment/model pairing if normalization was used.
    base_env = vec_env.envs[0]
    if not hasattr(base_env, "current_env_index"):
        try:
            # This field is also used by the project's summarize_df() to form
            # JSON filenames. Keep the human-readable label in the `agent`
            # column, but use a filesystem-safe environment identifier here.
            safe_env_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label)).strip("._-")
            base_env.current_env_index = safe_env_id or "benchmark_env"
        except Exception:
            pass

    if seeds is None:
        seeds = list(range(1, n_episodes + 1))
    if not seeds:
        raise ValueError("seeds must not be empty")
    # Existing benchmark helper indexes seeds by episode // num_envs.
    # For this single live environment, provide exactly one seed per episode.
    episode_seeds = [int(seeds[i % len(seeds)]) for i in range(n_episodes)]

    # test_utils controls the action selection internally (including MaskablePPO,
    # iDQN and ProtoKNN paths); its current implementation uses stochastic policy
    # selection. Seed each episode through the project seed utility.
    # The helper itself calls set_seeds(seeds[episode]); so pass our episode list.
    (df, metrics), certainty_stats = calculate_average_performances(
        model,
        vec_env,
        episode_seeds,
        proto_knn=None,
        logger=logger,
        num_envs=1,
        num_episodes=n_episodes,
        avoid_random=not compare_random,
        verbose=verbose,
    )

    # Keep the display/export label out of numerical aggregation. The project's
    # summarize_df() applies np.mean to every non-boolean column, so a string
    # column such as "agent" (or any other categorical metric) would raise a
    # TypeError. Summarize only episode identifiers plus numeric benchmark metrics.
    df.insert(0, "agent", label)
    summaries = None
    summary_all_envs = None
    if len(df):
        summary_dir = Path(save_dir).expanduser().resolve() / "benchmark_summaries" if save_dir else None
        summary_columns = [c for c in ("episode", "environment_ID") if c in df.columns]
        summary_columns += [
            c for c in df.columns
            if c not in summary_columns and c != "agent"
            and pd.api.types.is_numeric_dtype(df[c].dtype)
        ]
        summary_df = df.loc[:, summary_columns].copy()
        if summary_dir is not None:
            summary_dir.mkdir(parents=True, exist_ok=True)
            if any(c not in {"episode", "environment_ID"} for c in summary_df.columns):
                summaries, summary_all_envs = summarize_df(summary_df, save_folder=str(summary_dir))
            else:
                summaries, summary_all_envs = {}, {}
        else:
            # Keep the same project-defined aggregate statistics without writing files.
            summaries = {}
            for env_id, group in summary_df.groupby("environment_ID"):
                summaries[str(env_id)] = {}
                for col in group.select_dtypes(include=[np.number]).columns:
                    if col in {"episode"}:
                        continue
                    values = group[col].dropna().to_numpy(dtype=float)
                    if len(values):
                        summaries[str(env_id)][col] = {
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                            "min": float(np.min(values)),
                            "max": float(np.max(values)),
                        }
            numeric_cols = [c for c in summary_df.select_dtypes(include=[np.number]).columns if c != "episode"]
            summary_all_envs = {
                col: {
                    "mean": float(summary_df[col].mean()),
                    "std": float(summary_df[col].std(ddof=1)) if summary_df[col].count() > 1 else 0.0,
                    "min": float(summary_df[col].min()),
                    "max": float(summary_df[col].max()),
                }
                for col in numeric_cols if summary_df[col].notna().any()
            }

    output_path = Path(save_dir).expanduser().resolve() if save_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path / "benchmark_episode_metrics.csv", index=False)
        pd.Series(metrics, dtype=object).to_json(output_path / "benchmark_metric_lists.json", indent=2)
        import json
        with (output_path / "benchmark_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary_all_envs, f, indent=2, default=lambda x: np.asarray(x).tolist())

        import yaml
        with (output_path / "certainty_stats.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                [{k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                  for k, v in item.items() if k != "raw"} for item in certainty_stats],
                f,
                sort_keys=False,
            )

    return {
        "episodes": df,
        "metric_lists": metrics,
        "certainty_stats": certainty_stats,
        "summary_per_environment": summaries,
        "summary_all_environments": summary_all_envs,
        "env": eval_env,
        "vec_env": vec_env,
        "label": label,
        "save_dir": str(output_path) if output_path else None,
    }


def test_agent_action_time(
    model,
    env,
    *,
    n_episodes: int = 20,
    wrapper_factory: Optional[Callable[[Any], Any]] = None,
    save_dir: Optional[str] = None,
    logger=None,
    verbose: int = 1,
):
    """Run LaGCORL's project-specific action-inference/episode-time benchmark."""
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    _, calculate_average_action_time, _, _ = _project_test_utils()
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise ImportError("Install stable-baselines3 to use the benchmark testing API.") from exc

    logger = logger or logging.getLogger("notebook_test_api")
    eval_env = wrapper_factory(env) if wrapper_factory is not None else env
    # Keep the environment unwrapped so project-specific methods remain visible.
    vec_env = DummyVecEnv([lambda: eval_env])

    # Make iDQN obtain candidate actions from the evaluation environment.
    model.set_env(vec_env)
    action_times, episode_times = calculate_average_action_time(
        model, vec_env, config={}, proto_knn=None, logger=logger,
        num_episodes=n_episodes, verbose=verbose
    )
    result = {
        "action_times_seconds": action_times,
        "episode_times_seconds": episode_times,
        "summary": {
            "mean_action_seconds": float(np.mean(action_times)) if action_times else 0.0,
            "std_action_seconds": float(np.std(action_times, ddof=1)) if len(action_times) > 1 else 0.0,
            "mean_episode_seconds": float(np.mean(episode_times)) if episode_times else 0.0,
            "std_episode_seconds": float(np.std(episode_times, ddof=1)) if len(episode_times) > 1 else 0.0,
        },
    }
    if save_dir:
        import json
        out = Path(save_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"action_seconds": action_times}).to_csv(out / "action_times.csv", index=False)
        pd.DataFrame({"episode_seconds": episode_times}).to_csv(out / "episode_times.csv", index=False)
        with (out / "action_time_summary.json").open("w", encoding="utf-8") as f:
            json.dump(result["summary"], f, indent=2)
    return result