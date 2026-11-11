"""
Notebook API for training a LaGCORL GAE/VGAE reference encoder and loading it
for project AgentTrainingWrapper use.

Important:
- GAE training uses LaGCORL's configured graph-environment sample corpus
  (data/env_samples/<load_envs>), not an arbitrary single Gymnasium instance.
- The selected corpus/environment must match the graph/node/edge feature schema
  of the RL environment that will later consume the embeddings.
- This API calls the repository's train_gae.py entry point so that its normal
  config, train/validation split, metadata, and checkpoint-writing behavior stay
  intact. It disables the interactive "make this default" prompt.
"""
from __future__ import annotations

import os
import re
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple


def _read_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required by the LaGCORL GAE training API.") from exc
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_project_root(project_root: Optional[str] = None) -> Path:
    if project_root is not None:
        root = Path(project_root).expanduser().resolve()
    else:
        root = Path.cwd().resolve()
        if not (root / "gae" / "train_gae.py").exists() and (root / "LaGCORL" / "gae" / "train_gae.py").exists():
            root = root / "LaGCORL"
    if not (root / "gae" / "train_gae.py").is_file():
        raise FileNotFoundError(
            "Could not find <PROJECT_ROOT>/gae/train_gae.py. Pass project_root "
            "explicitly (the directory containing config.yaml, agents/, gae/, envs/)."
        )
    return root


def train_gae_for_embeddings(
    *,
    environment: str,
    load_envs: Optional[str] = None,
    project_root: Optional[str] = None,
    train_iterations: int = 1000,
    model_type: str = "gae",
    seed: int = 7,
    num_runs: int = 1,
    validation: bool = False,
    name: str = "notebook_reference",
    verbose: int = 1,
    compare_random_init: bool = False,
    train_config: Optional[str] = None,
    algo_config: Optional[str] = None,
    overrides: Optional[list] = None,
    timeout: Optional[int] = None,
) -> dict:
    """
    Train a GAE/VGAE using the repository's native training script.

    `environment` must be a key in LaGCORL/config.yaml.
    `load_envs` must be the name of a folder under LaGCORL/data/env_samples.
    When omitted, the default sample corpus for `environment` is read from
    config.yaml.

    Returns a dictionary with `output_dir`, `encoder_dirs`, `environment`,
    `load_envs`, `model_type`, `command`, and `stdout`.
    """
    if train_iterations < 0:
        raise ValueError("train_iterations must be >= 0")
    if model_type not in {"gae", "vgae"}:
        raise ValueError("model_type must be 'gae' or 'vgae'")
    if num_runs <= 0:
        raise ValueError("num_runs must be positive")

    root = _resolve_project_root(project_root)
    gae_dir = root / "gae"
    script = gae_dir / "train_gae.py"
    general_config_path = root / "config.yaml"
    general_config = _read_yaml(general_config_path)
    if environment not in general_config:
        raise ValueError(
            f"Environment {environment!r} is not a top-level key in {general_config_path}. "
            f"Available keys include: {', '.join(map(str, general_config.keys()))}"
        )
    env_general = general_config[environment]
    if load_envs is None:
        load_envs = env_general.get("default_envs")
    if not load_envs:
        raise ValueError(
            f"No load_envs supplied and config.yaml has no {environment}.default_envs value."
        )
    corpus_dir = root / "data" / "env_samples" / str(load_envs)
    if not corpus_dir.is_dir():
        raise FileNotFoundError(
            f"GAE graph corpus does not exist: {corpus_dir}. Set load_envs to an "
            "existing folder under data/env_samples."
        )

    train_config_path = Path(train_config).expanduser().resolve() if train_config else gae_dir / "config" / "train_config.yaml"
    algo_config_path = Path(algo_config).expanduser().resolve() if algo_config else gae_dir / "config" / "algo_config.yaml"
    if not train_config_path.is_file():
        raise FileNotFoundError(f"GAE train config not found: {train_config_path}")
    if not algo_config_path.is_file():
        raise FileNotFoundError(f"GAE algorithm config not found: {algo_config_path}")

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_.-") or "notebook_reference"
    run_name = f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logs_root = gae_dir / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable, str(script),
        "--train_config", str(train_config_path),
        "--algo_config", str(algo_config_path),
        "--environment", str(environment),
        "--model_type", str(model_type),
        "--name", run_name,
        "--load_envs", str(load_envs),
        "--train_iterations", str(int(train_iterations)),
        "--static_seed", str(int(seed)),
        "--num_runs", str(int(num_runs)),
        "--verbose", str(int(verbose)),
        "--force_saving",  # no interactive prompt
    ]
    if validation:
        command.append("--validation")
    if compare_random_init:
        command.append("--compare_random_init")
    if overrides:
        command.extend(["--override", *[str(x) for x in overrides]])

    # Stream the native trainer's progress into the notebook instead of buffering
    # all output until training finishes.
    # Ensure the subprocess can import the LaGCORL package.
    env = os.environ.copy()

    python_paths = [
        str(root.parent),  # /workspace: parent of LaGCORL
        str(root),  # /workspace/LaGCORL
    ]

    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        python_paths.append(existing_pythonpath)

    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env["PYTHONUNBUFFERED"] = "1"

    # Run native GAE trainer.
    completed = subprocess.run(
        command,
        cwd=str(gae_dir),
        text=True,
        timeout=timeout,
        env=env,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"LaGCORL GAE training exited with code "
            f"{completed.returncode}.\n"
            f"Project root: {root}\n"
            f"Working directory: {gae_dir}\n"
            f"PYTHONPATH: {env['PYTHONPATH']}\n"
            f"Command: {' '.join(command)}\n"
            f"Check the trainer output above and logs under {logs_root}."
        )

    candidates = [
        p for p in logs_root.iterdir()
        if p.is_dir() and p.name.startswith(run_name + "_" + model_type + "_")
    ]
    if not candidates:
        # The CLI's exact folder name is based on the name/model type/timestamp.
        candidates = [p for p in logs_root.iterdir() if p.is_dir() and run_name in p.name]
    if not candidates:
        raise FileNotFoundError(
            f"Training completed but could not locate its output folder under {logs_root}."
        )
    output_dir = max(candidates, key=lambda p: p.stat().st_mtime)
    encoder_dirs = sorted(
        p for p in output_dir.iterdir()
        if p.is_dir() and p.name.endswith("graph")
    )
    if not encoder_dirs:
        raise FileNotFoundError(
            f"Training output {output_dir} contains no '*graph' encoder folders. "
            "Check the environment/corpus and train_gae.py logs."
        )
    return {
        "output_dir": str(output_dir),
        "encoder_dirs": [str(p) for p in encoder_dirs],
        "environment": environment,
        "load_envs": str(load_envs),
        "model_type": model_type,
        "command": command,
        "stdout": None,
        "stderr": None,
    }


def load_gae_encoders(
    gae_run_dir: str,
    *,
    project_root: Optional[str] = None,
    prefer_best_validation: bool = True,
    device: str = "cpu",
) -> dict:
    """
    Load encoder objects and metadata from a train_gae_for_embeddings result
    directory. The returned `gnn_models` mapping is suitable for
    AgentTrainingWrapper when the graph names/features match the RL environment.
    """
    import torch

    root = _resolve_project_root(project_root)
    for path in (str(root), str(root / "gae")):
        if path not in sys.path:
            sys.path.insert(0, path)

    from LaGCORL.gae.model import GAEEncoder, VGAEEncoder

    try:
        from utils.file_utils import load_yaml
    except ImportError:
        import yaml
        def load_yaml(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)

    run_dir = Path(gae_run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"GAE run directory not found: {run_dir}")

    gnn_models, encoder_configs, encoder_specs, encoder_paths = {}, {}, {}, {}
    graph_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.endswith("graph"))
    if not graph_dirs:
        raise FileNotFoundError(f"No graph encoder folders found in {run_dir}")

    for graph_dir in graph_dirs:
        config_path = graph_dir / "train_config.yaml"
        spec_path = graph_dir / "model_spec.yaml"
        if not config_path.is_file() or not spec_path.is_file():
            continue
        encoder_config = load_yaml(str(config_path))
        encoder_spec = load_yaml(str(spec_path))
        best_path = graph_dir / "encoder_best_val.pth"
        final_path = graph_dir / "encoder_final.pth"
        checkpoint_path = best_path if prefer_best_validation and best_path.is_file() else final_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"No encoder checkpoint found in {graph_dir}")

        if encoder_config.get("model_type") == "gae":
            encoder = GAEEncoder(
                encoder_spec["node_feature_vector_size"],
                encoder_config["model_config"]["layers"],
                encoder_spec["edge_feature_vector_size"],
            )
        elif encoder_config.get("model_type") == "vgae":
            encoder = VGAEEncoder(
                encoder_spec["node_feature_vector_size"],
                encoder_config["model_config"]["layers"],
                encoder_spec["edge_feature_vector_size"],
                latent_dim=encoder_config.get("latent_dim_vgae", 32),
            )
        else:
            raise ValueError(f"Unsupported model_type in {config_path}: {encoder_config.get('model_type')!r}")

        state = torch.load(str(checkpoint_path), map_location=device)
        encoder.load_state_dict(state)
        encoder.to(device)
        encoder.eval()
        gnn_models[graph_dir.name] = encoder
        encoder_configs[graph_dir.name] = encoder_config
        encoder_specs[graph_dir.name] = encoder_spec
        encoder_paths[graph_dir.name] = str(checkpoint_path)

    if not gnn_models:
        raise RuntimeError(f"No usable encoder checkpoints found in {run_dir}")
    return {
        "gnn_models": gnn_models,
        "encoder_configs": encoder_configs,
        "encoder_specs": encoder_specs,
        "encoder_paths": encoder_paths,
        "gae_run_dir": str(run_dir),
    }


def make_agent_training_wrapper_factory(
    *,
    environment: str,
    gae_run_dir: str,
    agent_type: str,
    project_root: Optional[str] = None,
    logs_dir: Optional[str] = None,
    algorithm: str = "ppo",
    config_overrides: Optional[dict] = None,
    prefer_best_validation: bool = True,
):
    """
    Create raw_env -> AgentTrainingWrapper using the supplied GAE run.

    The project YAML files are merged in the same broad order as train_agent.py:
    RL train config, algorithm config, environment config, general environment
    defaults, then agent-type settings and explicit config_overrides.

    The raw environment must implement the project methods/specification expected
    by AgentTrainingWrapper (normally update_spec(), graph feature metadata, etc.).
    """
    import copy

    root = _resolve_project_root(project_root)
    bundle = load_gae_encoders(
        gae_run_dir,
        project_root=str(root),
        prefer_best_validation=prefer_best_validation,
    )
    try:
        from wrappers.agent_training_wrapper import AgentTrainingWrapper
    except ImportError:
        from LaGCORL.wrappers.agent_training_wrapper import AgentTrainingWrapper

    general_config = _read_yaml(root / "config.yaml")
    train_cfg_path = root / "agents" / "config" / "train_config.yaml"
    algo_cfg_path = root / "agents" / "config" / "algo_config.yaml"
    env_cfg_path = root / "envs" / "config" / f"{environment}_config.yaml"
    for p in (train_cfg_path, algo_cfg_path, env_cfg_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"Required project config not found: {p}. If your repository uses "
                "different config paths, pass a custom wrapper_factory in the notebook."
            )

    config = _read_yaml(train_cfg_path)
    algo_cfg = _read_yaml(algo_cfg_path)
    if agent_type in algo_cfg:
        config["algorithm_hyperparams"] = algo_cfg[agent_type].get(algorithm, {})
    else:
        config["algorithm_hyperparams"] = algo_cfg.get(algorithm, {})
    config["policy_kwargs"] = algo_cfg.get("policy_kwargs", {})
    config.update({
        "environment": environment,
        "algorithm_type": agent_type,
        "algorithm": "idqn" if agent_type == "iterative" else algorithm,
        "GNN_observations": True,
        "validation": False,
        "projection_agent": False,
        "proto_knn": 0,
        "logs_folder": logs_dir or str(Path("runs") / f"{environment}_{agent_type}_wrapper"),
    })
    environment_config = _read_yaml(env_cfg_path)
    # The CLI augments the environment YAML with project-wide schema entries and
    # run-level limits before calling environment.update_config(...).
    keys_config = {
        key: value for key, value in general_config.items()
        if isinstance(value, dict) and "key" in value
    }
    environment_config.update(keys_config)
    for key in (
        "verbose", "max_steps_overall", "max_steps_coefficient", "max_sweeps",
        "save_logs_transitions", "remove_invalid_actions",
        "terminate_at_approximate_best", "padding_invalid_action_penalty_sum",
        "no_action_penalty_sum", "no_action_support",
    ):
        if key in config:
            environment_config[key] = config[key]
    config.update(environment_config)
    if environment in config and isinstance(config[environment], dict):
        config.update(config[environment])
    if agent_type in config and isinstance(config[agent_type], dict):
        if not (agent_type == "iterative" and config.get("projection_agent")):
            config.update(config[agent_type])
    config.update(config_overrides or {})
    config.pop("logs_folder", None)  # train_agent.py removes this before **config
    config.pop("env_name", None)
    config["gae_model_type"] = next(iter(bundle["encoder_configs"].values())).get("model_type", "gae")
    for graph_name, path in bundle["encoder_paths"].items():
        config[f"graph_encoder_path_{graph_name}"] = path
        config[f"graph_encoder_config_path_{graph_name}"] = str(Path(gae_run_dir) / graph_name / "train_config.yaml")
        config[f"graph_encoder_spec_path_{graph_name}"] = str(Path(gae_run_dir) / graph_name / "model_spec.yaml")

    wrapper_logs_dir = logs_dir or str(Path("runs") / f"{environment}_{agent_type}_wrapper")
    os.makedirs(wrapper_logs_dir, exist_ok=True)

    def wrapper_factory(raw_env):
        if not hasattr(raw_env, "update_spec"):
            raise TypeError(
                f"The {type(raw_env).__name__} environment has no update_spec(). "
                "It may not be compatible with LaGCORL's AgentTrainingWrapper. "
                "Use the project's native environment for this GAE/agent experiment, "
                "or provide a custom wrapper_factory."
            )
        if hasattr(raw_env, "update_config"):
            raw_env.update_config(copy.deepcopy(environment_config))
        raw_env.update_spec()
        wrapper_config = copy.deepcopy(config)
        return AgentTrainingWrapper(
            env=raw_env,
            env_name=environment,
            gnn_models=bundle["gnn_models"],
            logs_folder=wrapper_logs_dir,
            **wrapper_config,
        )

    wrapper_factory.gae_bundle = bundle
    wrapper_factory.gae_run_dir = str(Path(gae_run_dir).expanduser().resolve())
    return wrapper_factory