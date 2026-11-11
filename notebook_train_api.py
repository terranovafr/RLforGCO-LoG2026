"""
Notebook-friendly training API for environments created in Python/Jupyter.

This deliberately bypasses the CLI's environment-folder/split pipeline. It trains
on a supplied Gymnasium environment instance, or loads one from the default corpus
configured in config.yaml when env=None.

Examples:
    result = train_agent_on_env(env, algorithm="ppo", agent_type="discrete",
                                total_timesteps=10_000)
    model = result["model"]

For project-specific projection training, pass a wrapper_factory that applies
LaGCORL's AgentTrainingWrapper to the raw environment using the encoder/config
used by your project.
"""
from __future__ import annotations

import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Callable, Optional, Any



def _read_yaml(path: Path) -> dict:
    """Read a YAML mapping, returning an empty mapping for an empty file."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load an environment from config.yaml.") from exc

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}, got {type(data).__name__}.")
    return data


def _resolve_project_root(project_root: Optional[str] = None) -> Path:
    """Resolve the LaGCORL project root from an explicit path or the working directory."""
    if project_root is not None:
        root = Path(project_root).expanduser().resolve()
    else:
        root = Path.cwd().resolve()
        if not (root / "config.yaml").is_file() and (root / "LaGCORL" / "config.yaml").is_file():
            root = root / "LaGCORL"

    if not (root / "config.yaml").is_file():
        raise FileNotFoundError(
            f"Could not find {root / 'config.yaml'}. Pass project_root explicitly "
            "(the directory containing config.yaml, agents/, envs/, and data/)."
        )
    return root


def _configured_environment_name(general_config: dict, requested: Optional[str]) -> str:
    """Choose an environment key from config.yaml when the caller omits it."""
    candidates = [
        key for key, value in general_config.items()
        if isinstance(value, dict) and value.get("default_envs")
    ]
    if requested is not None:
        if requested not in general_config or not isinstance(general_config[requested], dict):
            raise ValueError(
                f"Environment {requested!r} is not a mapping key in config.yaml. "
                f"Available environment keys: {candidates or list(general_config)}"
            )
        return requested

    # Honour an explicit project-level default if one exists.
    for key in ("default_environment", "environment"):
        value = general_config.get(key)
        if isinstance(value, str) and value in general_config and isinstance(general_config[value], dict):
            return value

    # Match the native train_agent.py CLI default when it is configured.
    if "tsp" in candidates:
        return "tsp"
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        warnings.warn(
            "No environment was specified and config.yaml has multiple environments. "
            f"Falling back to the first configured environment, {candidates[0]!r}. "
            "Pass environment=... to choose explicitly.",
            RuntimeWarning,
            stacklevel=3,
        )
        return candidates[0]

    raise ValueError(
        "Could not infer an environment from config.yaml: no environment sections "
        "with a 'default_envs' value were found. Pass env=... or environment=... explicitly."
    )


def _load_default_environment(
    *,
    environment: Optional[str] = None,
    project_root: Optional[str] = None,
) -> tuple[Any, str, Path]:
    """
    Load one environment instance from the configured default sample corpus.

    config.yaml stores the default corpus name rather than a Python environment
    instance, so this loads the first sorted ``env.pkl`` under that corpus.
    """
    root = _resolve_project_root(project_root)
    general_config = _read_yaml(root / "config.yaml")
    env_name = _configured_environment_name(general_config, environment)
    corpus_name = general_config[env_name].get("default_envs")
    if not corpus_name:
        raise ValueError(
            f"config.yaml has no 'default_envs' entry for environment {env_name!r}. "
            "Set that entry or pass a concrete env instance."
        )

    corpus_dir = root / "data" / "env_samples" / str(corpus_name)
    if not corpus_dir.is_dir():
        raise FileNotFoundError(
            f"The default corpus configured for {env_name!r} does not exist: {corpus_dir}"
        )
    # Ensure pickle can resolve LaGCORL modules even when called from a notebook
    # whose current working directory is the repository parent.
    for entry in (str(root.parent), str(root)):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    env_files = sorted(corpus_dir.glob("*/env.pkl"))
    if not env_files:
        raise FileNotFoundError(
            f"No */env.pkl files were found in configured corpus {corpus_dir}"
        )

    with env_files[0].open("rb") as handle:
        env = pickle.load(handle)

    # Mirror the core configuration steps used by LaGCORL's native training path.
    env_config_path = root / "envs" / "config" / f"{env_name}_config.yaml"
    env_config = _read_yaml(env_config_path)
    shared_schema = {
        key: value for key, value in general_config.items()
        if isinstance(value, dict) and "key" in value
    }
    env_config.update(shared_schema)

    if hasattr(env, "update_config"):
        env.update_config(env_config)
    if hasattr(env, "update_spec"):
        env.update_spec()

    return env, env_name, root


def _get_model_class(algorithm: str):
    """Resolve project algorithm classes first, then common SB3 algorithms."""
    try:
        from utils.train_utils import algorithm_models
        if algorithm in algorithm_models:
            return algorithm_models[algorithm]
    except (ImportError, ModuleNotFoundError):
        pass

    mapping = {
        "ppo": ("stable_baselines3", "PPO"),
        "a2c": ("stable_baselines3", "A2C"),
        "dqn": ("stable_baselines3", "DQN"),
        "ddpg": ("stable_baselines3", "DDPG"),
        "sac": ("stable_baselines3", "SAC"),
        "td3": ("stable_baselines3", "TD3"),
        "trpo": ("sb3_contrib", "TRPO"),
        "tqc": ("sb3_contrib", "TQC"),
        "maskable_ppo": ("sb3_contrib", "MaskablePPO"),
    }
    if algorithm not in mapping:
        raise ValueError(
            f"Unknown or unavailable algorithm {algorithm!r}. "
            f"Available fallback algorithms: {', '.join(mapping)}. "
            "If this is a LaGCORL-specific algorithm, ensure the project root "
            "is on sys.path and its dependencies are installed."
        )
    module_name, class_name = mapping[algorithm]
    try:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            f"Could not import {class_name} from {module_name}. "
            "Install the corresponding dependency or use a project-supported algorithm."
        ) from exc


def train_agent_on_env(
    env=None,
    *,
    environment: Optional[str] = None,
    project_root: Optional[str] = None,
    algorithm: str = "ppo",
    agent_type: str = "discrete",
    total_timesteps: int = 10_000,
    seed: int = 7,
    policy: Optional[str] = None,
    policy_kwargs: Optional[dict] = None,
    algorithm_kwargs: Optional[dict] = None,
    wrapper_factory: Optional[Callable[[Any], Any]] = None,
    tensorboard_log: Optional[str] = None,
    save_path: Optional[str] = None,
    checkpoint_freq: Optional[int] = None,
    verbose: int = 1,
):
    """
    Train an SB3/LaGCORL model directly on a notebook-created Gymnasium env.

    Parameters
    ----------
    env:
        Optional environment instance. If omitted, load the first ``env.pkl`` from
        the ``default_envs`` corpus configured in ``config.yaml``.
    environment:
        Optional config.yaml environment key used only when ``env`` is omitted.
        If omitted too, the configured default is inferred (preferring ``tsp`` to
        match LaGCORL's native CLI default).
    project_root:
        Optional LaGCORL project root containing ``config.yaml``.
    algorithm:
        Algorithm name such as 'ppo', 'a2c', 'dqn', 'sac', or 'td3'. Project
        algorithms in utils.train_utils.algorithm_models are used when available.
    agent_type:
        'discrete', 'projection', or 'iterative'. This is an experiment label;
        projection/iterative must be paired with the project's wrapper_factory.
    wrapper_factory:
        Optional callable raw_env -> wrapped_env. For projection, use this to
        apply AgentTrainingWrapper with the correct GNN encoders/config.
    save_path:
        Optional model path without requiring the CLI log-folder structure.
    checkpoint_freq:
        Optional number of timesteps between checkpoint saves.
    """
    if env is None:
        env, resolved_environment, resolved_root = _load_default_environment(
            environment=environment,
            project_root=project_root,
        )
        print(
            f"No env instance supplied; loaded {type(env).__name__} from "
            f"config.yaml environment={resolved_environment!r} "
            f"under {resolved_root}."
        )
    else:
        resolved_environment = environment
        resolved_root = _resolve_project_root(project_root) if project_root is not None else None

    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

    valid_types = {"discrete", "projection", "iterative"}
    if agent_type not in valid_types:
        raise ValueError(f"agent_type must be one of {sorted(valid_types)}")
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")

    if agent_type in {"projection", "iterative"} and wrapper_factory is None:
        raise ValueError(
            f"agent_type={agent_type!r} needs wrapper_factory. The CLI applies "
            "LaGCORL's AgentTrainingWrapper and, for graph-based projection, "
            "loads trained GAE/VGAE encoders. Pass the same wrapper/encoder "
            "configuration here; a raw Gymnasium env alone is not enough."
        )

    training_env = wrapper_factory(env) if wrapper_factory is not None else env

    # LaGCORL's iterative agent is the custom iDQN implementation. It needs
    # StateActionDQNPolicy and the next-action replay buffer, as in train_agent.py.
    if agent_type == "iterative":
        algorithm = "idqn"
        if policy is None:
            try:
                from sb3_sa_contrib.policies import StateActionDQNPolicy
                policy = StateActionDQNPolicy
            except ImportError as exc:
                raise ImportError(
                    "The iterative agent requires LaGCORL's sb3_sa_contrib package."
                ) from exc

    if policy is None:
        # Dict observations require SB3's multi-input policy; flat Box observations
        # use MLP. This is only a default; pass policy explicitly when needed.
        try:
            from gymnasium.spaces import Dict as DictSpace
        except ImportError:
            from gym.spaces import Dict as DictSpace
        policy = "MultiInputPolicy" if isinstance(training_env.observation_space, DictSpace) else "MlpPolicy"

    model_class = _get_model_class(algorithm)
    output_dir = None
    if save_path:
        output_dir = str(Path(save_path).expanduser().resolve().parent)
        os.makedirs(output_dir, exist_ok=True)
    if tensorboard_log:
        os.makedirs(tensorboard_log, exist_ok=True)

    # Do not put Monitor around LaGCORL wrappers: SB3 Monitor does not forward
    # custom methods such as get_metrics() or get_candidate_action_KNNs().
    # DummyVecEnv can wrap the raw Gymnasium-compatible environment directly.
    vec_env = DummyVecEnv([lambda: training_env])

    model_kwargs = dict(algorithm_kwargs or {})
    if agent_type == "iterative":
        try:
            from sb3_sa_contrib.buffers import DictReplayBufferWithNextActions
        except ImportError as exc:
            raise ImportError(
                "The iterative agent requires DictReplayBufferWithNextActions from sb3_sa_contrib."
            ) from exc
        model_kwargs.setdefault("replay_buffer_class", DictReplayBufferWithNextActions)
    model_kwargs.setdefault("seed", seed)
    model_kwargs.setdefault("verbose", verbose)
    if tensorboard_log is not None:
        model_kwargs.setdefault("tensorboard_log", tensorboard_log)
    if policy_kwargs is not None:
        model_kwargs.setdefault("policy_kwargs", policy_kwargs)

    model = model_class(policy, vec_env, **model_kwargs)

    # LaGCORL's custom wrappers may not be compatible with SB3 Monitor, but
    # SB3 only emits rollout/ep_rew_mean automatically when episode info is
    # supplied by Monitor. Track completed episodes directly instead.
    import numpy as np
    from collections import deque

    class _EpisodeRewardLoggingCallback(BaseCallback):
        """Record SB3-compatible rollout/ep_rew_mean without requiring Monitor."""

        def __init__(self, window_size: int = 100):
            super().__init__(verbose=0)
            self.window_size = window_size
            self.episode_returns = None
            self.recent_episode_rewards = deque(maxlen=window_size)

        def _on_training_start(self) -> None:
            self.episode_returns = np.zeros(self.training_env.num_envs, dtype=np.float64)

        def _on_step(self) -> bool:
            rewards = np.asarray(self.locals.get("rewards", []), dtype=np.float64).reshape(-1)
            dones = np.asarray(self.locals.get("dones", []), dtype=bool).reshape(-1)
            if self.episode_returns is None:
                self.episode_returns = np.zeros(self.training_env.num_envs, dtype=np.float64)

            n = min(len(rewards), len(dones), len(self.episode_returns))
            for env_index in range(n):
                self.episode_returns[env_index] += float(rewards[env_index])
                if dones[env_index]:
                    self.recent_episode_rewards.append(float(self.episode_returns[env_index]))
                    self.episode_returns[env_index] = 0.0
            return True

        def _on_rollout_end(self) -> None:
            if self.recent_episode_rewards:
                self.logger.record(
                    "rollout/ep_rew_mean",
                    float(np.mean(self.recent_episode_rewards)),
                )

    callbacks = []
    if tensorboard_log is not None:
        callbacks.append(_EpisodeRewardLoggingCallback(window_size=100))
    if checkpoint_freq is not None:
        if checkpoint_freq <= 0:
            raise ValueError("checkpoint_freq must be positive")
        checkpoint_dir = os.path.join(output_dir or os.getcwd(), "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, checkpoint_freq),
                save_path=checkpoint_dir,
                name_prefix=f"{agent_type}_{algorithm}",
            )
        )

    model.learn(
        total_timesteps=int(total_timesteps),
        callback=callbacks or None,
        progress_bar=False,
    )

    if save_path:
        save_target = str(Path(save_path).with_suffix(""))
        model.save(save_target)

    return {
        "model": model,
        "env": training_env,
        "vec_env": vec_env,
        "algorithm": algorithm,
        "agent_type": agent_type,
        "total_timesteps": int(total_timesteps),
        "save_path": str(Path(save_path).with_suffix("")) if save_path else None,
        "environment_name": resolved_environment,
        "project_root": str(resolved_root) if resolved_root is not None else None,
    }


def evaluate_agent(model, env=None, *, n_episodes: int = 5, deterministic: bool = True,
                   wrapper_factory: Optional[Callable[[Any], Any]] = None,
                   environment: Optional[str] = None,
                   project_root: Optional[str] = None):
    """
    Evaluate a trained model on a fresh environment instance.

    For a fair comparison, pass the same kind of wrapper used for training.
    Returns per-episode rewards and lengths. VecNormalize is not added here, so
    this helper is intended for runs trained without VecNormalize.
    """
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    if env is None:
        env, resolved_environment, resolved_root = _load_default_environment(
            environment=environment,
            project_root=project_root,
        )
        print(
            f"No env instance supplied; loaded {type(env).__name__} from "
            f"config.yaml environment={resolved_environment!r} under {resolved_root}."
        )
    eval_env = wrapper_factory(env) if wrapper_factory is not None else env
    rewards, lengths = [], []

    for _ in range(n_episodes):
        reset_result = eval_env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        done = False
        total_reward = 0.0
        length = 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            step_result = eval_env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, _info = step_result
                done = bool(terminated or truncated)
            else:
                obs, reward, done, _info = step_result
                done = bool(done)
            total_reward += float(reward)
            length += 1
        rewards.append(total_reward)
        lengths.append(length)

    return {
        "episode_rewards": rewards,
        "episode_lengths": lengths,
        "mean_reward": float(sum(rewards) / len(rewards)),
        "mean_length": float(sum(lengths) / len(lengths)),
        "environment_name": locals().get("resolved_environment", environment),
        "project_root": str(locals().get("resolved_root")) if locals().get("resolved_root") is not None else None,
    }