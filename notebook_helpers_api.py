"""Small reusable helpers for the LaGCORL tutorial notebooks.

Keep notebook cells focused on experiment choices and results; put reusable
visualisation, artifact management, and sample-corpus plumbing here.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from IPython.display import display

import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym

LOG_ROOT = Path("logs")


def plot_tensorboard(
    log_root,
    title=None,
    metric="rollout/ep_rew_mean",
    run_selector="latest",
    show_all_runs=False,
):
    """
    Plot a scalar metric from TensorBoard event files.

    Parameters
    ----------
    log_root : str or Path
        Root directory containing TensorBoard logs. Event files may be nested
        inside algorithm-specific directories such as PPO_17, DQN_3, or A2C_2.

    title : str, optional
        Custom plot title.

    metric : str
        TensorBoard scalar tag to plot. Examples:
        - rollout/ep_rew_mean
        - train/value_loss
        - train/policy_gradient_loss
        - train/actor_loss
        - train/critic_loss
        - train/entropy_loss
        - train/graph/total_loss

    run_selector : str
        "latest": select the newest detected run directory.
        "all": combine event files from all detected runs.

    show_all_runs : bool
        If True, plot each detected run separately for comparison.
        This overrides run_selector and plots all runs.

    Returns
    -------
    pandas.DataFrame or dict
        A DataFrame containing timestep and metric values, or a dictionary
        mapping run names to DataFrames when show_all_runs=True.
    """
    log_root = Path(log_root).expanduser().resolve()

    if not log_root.exists():
        print(f"TensorBoard log folder does not exist: {log_root}")
        return None

    if not log_root.is_dir():
        print(f"Expected a directory, got: {log_root}")
        return None

    event_pattern = "events.out.tfevents.*"
    event_files = sorted(log_root.rglob(event_pattern))

    if not event_files:
        print(f"No TensorBoard event files found under: {log_root}")
        return None

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
        from tensorboard.backend.event_processing.event_file_loader import (
            LegacyEventFileLoader,
        )
        from tensorboard.util.tensor_util import make_ndarray
    except ImportError:
        print(
            "TensorBoard is not installed. Run `%pip install tensorboard` "
            "in your notebook and retry."
        )
        return None

    def read_event_file(event_file, requested_metric):
        """Read scalar values from one TensorBoard event file."""
        values = {}
        tags_found = set()

        # First try EventAccumulator.
        try:
            accumulator = EventAccumulator(
                str(event_file),
                size_guidance={"scalars": 0, "tensors": 0},
            )
            accumulator.Reload()

            tags = accumulator.Tags()
            scalar_tags = set(tags.get("scalars", []))
            tensor_tags = set(tags.get("tensors", []))

            matching_tags = [
                tag
                for tag in scalar_tags | tensor_tags
                if tag == requested_metric
                or tag.endswith("/" + requested_metric)
            ]

            for tag in matching_tags:
                if tag in scalar_tags:
                    items = accumulator.Scalars(tag)
                    for item in items:
                        values[int(item.step)] = float(item.value)
                        tags_found.add(tag)

                elif tag in tensor_tags:
                    items = accumulator.Tensors(tag)
                    for item in items:
                        try:
                            array = make_ndarray(item.tensor_proto)
                            if array.size:
                                values[int(item.step)] = float(
                                    array.reshape(-1)[0]
                                )
                                tags_found.add(tag)
                        except Exception:
                            continue

        except Exception:
            pass

        # Fallback: parse raw event records directly.
        if not values:
            try:
                for event in LegacyEventFileLoader(str(event_file)).Load():
                    if not event.HasField("summary"):
                        continue

                    for item in event.summary.value:
                        if not (
                            item.tag == requested_metric
                            or item.tag.endswith("/" + requested_metric)
                        ):
                            continue

                        if item.HasField("simple_value"):
                            values[int(event.step)] = float(item.simple_value)
                            tags_found.add(item.tag)

                        elif item.HasField("tensor"):
                            try:
                                array = make_ndarray(item.tensor)
                                if array.size:
                                    values[int(event.step)] = float(
                                        array.reshape(-1)[0]
                                    )
                                    tags_found.add(item.tag)
                            except Exception:
                                continue

            except Exception as exc:
                print(f"Could not read {event_file.name}: {exc}")

        return values, tags_found

    def discover_runs():
        """
        Group event files by their nearest algorithm-run directory.

        Recognizes directories such as PPO_17, DQN_3, A2C_2, and IDQN_4.
        If no such directory exists, uses the parent of each event file.
        """
        runs = {}

        for event_file in event_files:
            run_dir = None

            for parent in event_file.parents:
                if parent == log_root.parent:
                    break

                # Recognize common algorithm run names with numeric suffixes.
                if re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*_\d+", parent.name):
                    run_dir = parent
                    break

                if parent == log_root:
                    break

            if run_dir is None:
                run_dir = event_file.parent

            runs.setdefault(run_dir, []).append(event_file)

        return runs

    def run_sort_key(run_dir):
        """Sort numeric run names numerically; use mtime for other names."""
        match = re.fullmatch(r"(.+?)_(\d+)", run_dir.name)

        if match:
            return (
                1,
                match.group(1).lower(),
                int(match.group(2)),
                run_dir.stat().st_mtime,
            )

        return (0, run_dir.name.lower(), 0, run_dir.stat().st_mtime)

    def load_run(run_dir, files):
        """Merge values from all event files belonging to a run."""
        by_step = {}
        found_tags = set()

        # Newer event files should overwrite earlier values at duplicate steps.
        for event_file in sorted(
            files,
            key=lambda path: path.stat().st_mtime,
        ):
            values, tags = read_event_file(event_file, metric)
            by_step.update(values)
            found_tags.update(tags)

        if not by_step:
            return None, found_tags

        steps = sorted(by_step)

        frame = pd.DataFrame({
            "timesteps": steps,
            metric: [by_step[step] for step in steps],
        })

        return frame, found_tags

    runs = discover_runs()

    if not runs:
        print(f"No TensorBoard runs found under: {log_root}")
        return None

    ordered_runs = sorted(runs, key=run_sort_key)

    if show_all_runs:
        selected_runs = ordered_runs
    elif run_selector == "all":
        selected_runs = ordered_runs
    elif run_selector == "latest":
        # Prefer the most recently modified run, which also handles
        # non-numeric names and resumed/custom training runs.
        selected_runs = [
            max(
                ordered_runs,
                key=lambda run: max(
                    path.stat().st_mtime for path in runs[run]
                ),
            )
        ]
    else:
        raise ValueError("run_selector must be 'latest' or 'all'.")

    results = {}

    for run_dir in selected_runs:
        print(f"Reading TensorBoard run: {run_dir}")

        frame, found_tags = load_run(run_dir, runs[run_dir])

        if frame is None or frame.empty:
            print(f"No values found for metric '{metric}' in {run_dir}.")
            continue

        results[run_dir.name] = frame

    if not results:
        # Diagnose the available tags across the event files.
        all_tags = set()

        for event_file in event_files:
            try:
                accumulator = EventAccumulator(
                    str(event_file),
                    size_guidance={"scalars": 0, "tensors": 0},
                )
                accumulator.Reload()

                tags = accumulator.Tags()
                all_tags.update(tags.get("scalars", []))
                all_tags.update(tags.get("tensors", []))

            except Exception:
                pass

        print(
            f"No values found for '{metric}' in the selected TensorBoard run(s)."
        )

        if all_tags:
            print("\nAvailable TensorBoard scalar/tensor tags:")
            for tag in sorted(all_tags):
                print(f"  {tag}")
        else:
            print("No readable scalar or tensor tags were found.")

        print(
            "\nTry another metric, for example "
            "metric='train/value_loss', or use run_selector='all'."
        )
        return None

    # Plot the selected run(s).
    fig, ax = plt.subplots(figsize=(10, 5))

    for run_name, frame in results.items():
        ax.plot(
            frame["timesteps"],
            frame[metric],
            marker=".",
            linewidth=1.5,
            label=run_name if len(results) > 1 else None,
        )

    ax.set_xlabel("Training timestep")
    ax.set_ylabel(metric)
    ax.set_title(title or f"TensorBoard — {metric}")

    if len(results) > 1:
        ax.legend(title="Run")

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    display(fig)
    plt.close(fig)

    if show_all_runs or run_selector == "all":
        return results

    return next(iter(results.values()))

def create_run_layout(log_root, agent_type, environment="maxcut", timestamp=None):
    """Create a timestamped run directory and its standard artifact subfolders.

    Layout: logs/<agent_type>/<environment>_<timestamp>/{checkpoints,tensorboard,
    test_metrics,train_metrics}
    """
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(log_root) / str(agent_type) / f"{environment}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    paths = {"run": run_dir}
    for folder in ("checkpoints", "tensorboard", "test_metrics", "train_metrics"):
        path = run_dir / folder
        path.mkdir(parents=True, exist_ok=True)
        paths[folder] = path
    return paths


def save_agent_artifact(model, agent_name, run_dir):
    """Save a model checkpoint and provenance metadata inside the run's checkpoints folder."""
    run_dir = Path(run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint_dir / "model"))
    metadata = {
        "agent_name": agent_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir.resolve()),
        "model_file": "model.zip",
    }
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("Saved model checkpoint:", (checkpoint_dir / "model.zip").resolve())
    return checkpoint_dir


def load_corpus_env(corpus_dir, env_id):
    """Load a fresh environment instance from a generated corpus."""
    import pickle
    path = Path(corpus_dir) / str(env_id) / "env.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Corpus environment not found: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def benchmark_on_corpus(
    model,
    *,
    corpus_dir,
    env_ids,
    test_agent_on_env,
    n_episodes=20,
    seeds=None,
    wrapper_factory=None,
    test_metrics_dir,
    label="agent",
    verbose=1,
):
    """Benchmark a model on each specified corpus environment and combine results.

    Each environment is freshly unpickled, wrapped independently, and evaluated
    separately. Per-environment files and a combined episode-metrics CSV are saved
    under the run's test_metrics directory.
    """
    import pandas as pd

    test_metrics_dir = Path(test_metrics_dir)
    test_metrics_dir.mkdir(parents=True, exist_ok=True)
    per_environment = {}
    frames = []
    for env_id in env_ids:
        env = load_corpus_env(corpus_dir, env_id)
        env_result = test_agent_on_env(
            model,
            env,
            n_episodes=n_episodes,
            seeds=seeds,
            wrapper_factory=wrapper_factory,
            compare_random=False,
            save_dir=str(test_metrics_dir / f"env_{env_id}"),
            label=label,
            verbose=verbose,
        )
        episodes = env_result["episodes"].copy()
        episodes.insert(0, "corpus_env_id", int(env_id))
        frames.append(episodes)
        per_environment[str(env_id)] = env_result
    if not frames:
        raise ValueError("env_ids must contain at least one evaluation environment.")
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(test_metrics_dir / "episode_metrics_all_envs.csv", index=False)
    combined_summary = numeric_metric_means(combined)
    (test_metrics_dir / "summary_all_envs.json").write_text(
        json.dumps(combined_summary, indent=2), encoding="utf-8"
    )
    return {
        "episodes": combined,
        "per_environment": per_environment,
        "summary": combined_summary,
        "label": label,
        "save_dir": str(test_metrics_dir.resolve()),
    }


def sample_maxcut_corpus(
    project_root,
    *,
    count=40,
    graph_size_range=(8, 60),
    edge_weight_range=(1, 20),
    seed=7,
    max_sweeps=10,
    train_fraction=0.8,
    validation_fraction=0.1,
    test_fraction=0.1,
    corpus_name=None,
):
    """Generate native ExtendedMaxCutEnv pickle files plus a GAE split.yaml.

    Returns a dictionary containing the corpus path/name, split IDs, and a
    DataFrame describing the sampled graphs. The project environment class is
    imported lazily so this module can also be imported in lightweight contexts.
    """
    import pickle
    import yaml
    import networkx as nx
    import numpy as np

    if count < 3:
        raise ValueError("count must be at least 3 to create train/validation/test splits.")
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(value < 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Train/validation/test fractions must be nonnegative and sum to 1.")
    min_nodes, max_nodes = map(int, graph_size_range)
    min_weight, max_weight = map(int, edge_weight_range)
    if min_nodes < 3 or max_nodes < min_nodes:
        raise ValueError("graph_size_range must satisfy 3 <= min_nodes <= max_nodes.")
    if min_weight < 1 or max_weight < min_weight:
        raise ValueError("edge_weight_range must satisfy 1 <= min_weight <= max_weight.")

    from LaGCORL.envs.env_maxcut import ExtendedMaxCutEnv
    root = Path(project_root).expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    corpus_name = corpus_name or f"maxcut_sampled_{stamp}"
    corpus_dir = root / "data" / "env_samples" / corpus_name
    corpus_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)

    def sample_graph():
        n = int(rng.integers(min_nodes, max_nodes + 1))
        family = str(rng.choice(["erdos_renyi", "barabasi_albert", "watts_strogatz", "gnp_dense"]))
        graph_seed = int(rng.integers(0, 2**31 - 1))
        if family in {"erdos_renyi", "gnp_dense"}:
            p = float(rng.uniform(0.45, 0.85) if family == "gnp_dense" else rng.uniform(0.08, 0.45))
            graph = nx.erdos_renyi_graph(n, p, seed=graph_seed)
        elif family == "barabasi_albert":
            m = int(rng.integers(1, min(6, n - 1) + 1))
            graph = nx.barabasi_albert_graph(n, m, seed=graph_seed)
        else:
            k = int(rng.integers(2, max(3, min(n - 1, 10)) + 1))
            if k % 2:
                k = k + 1 if k + 1 < n else k - 1
            k = max(2, min(k, n - 1))
            graph = nx.watts_strogatz_graph(n, k, float(rng.uniform(0.05, 0.5)), seed=graph_seed)
        if not nx.is_connected(graph):
            components = [list(component) for component in nx.connected_components(graph)]
            for left, right in zip(components, components[1:]):
                graph.add_edge(int(rng.choice(left)), int(rng.choice(right)))
        for u, v in graph.edges:
            graph[u][v]["weight"] = int(rng.integers(min_weight, max_weight + 1))
        graph.graph["sampled_family"] = family
        return graph

    ids = list(range(count))
    rng.shuffle(ids)
    n_train = min(max(1, int(round(count * train_fraction))), count - 2)
    n_val = min(max(1, int(round(count * validation_fraction))), count - n_train - 1)
    train_ids = sorted(ids[:n_train])
    val_ids = sorted(ids[n_train:n_train + n_val])
    test_ids = sorted(ids[n_train + n_val:])

    metadata = []
    for env_id in range(count):
        graph = sample_graph()
        env = ExtendedMaxCutEnv(graph=graph, max_sweeps=max_sweeps)
        env_dir = corpus_dir / str(env_id)
        env_dir.mkdir()
        with (env_dir / "env.pkl").open("wb") as handle:
            pickle.dump(env, handle, protocol=pickle.HIGHEST_PROTOCOL)
        metadata.append({
            "id": env_id,
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "family": graph.graph["sampled_family"],
        })

    split_payload = {
        "training_set": [{"id": i} for i in train_ids],
        "validation_set": [{"id": i} for i in val_ids],
        "test_set": [{"id": i} for i in test_ids],
    }
    with (corpus_dir / "split.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(split_payload, handle, sort_keys=False)

    return {
        "corpus_name": corpus_name,
        "corpus_dir": corpus_dir,
        "train_ids": train_ids,
        "validation_ids": val_ids,
        "test_ids": test_ids,
        "graphs": pd.DataFrame(metadata),
    }


def save_metric_frame(frame, output_dir, filename="training_metrics.csv"):
    """Persist a TensorBoard metric DataFrame under train_metrics/."""
    if frame is None:
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    if isinstance(frame, dict):
        pieces = []
        for run_name, run_frame in frame.items():
            item = run_frame.copy()
            item.insert(0, "tensorboard_run", run_name)
            pieces.append(item)
        if not pieces:
            return None
        import pandas as pd
        frame = pd.concat(pieces, ignore_index=True)
    frame.to_csv(path, index=False)
    print("Saved extracted training metrics:", path.resolve())
    return path


def save_training_metadata(run_dir, *, agent_type, environment, algorithm, total_timesteps, seed):
    """Save compact, reproducible training metadata in train_metrics/."""
    run_dir = Path(run_dir)
    train_dir = run_dir / "train_metrics"
    train_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "agent_type": agent_type,
        "environment": environment,
        "algorithm": algorithm,
        "total_timesteps": int(total_timesteps),
        "seed": int(seed),
        "run_dir": str(run_dir.resolve()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (train_dir / "training_metadata.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return train_dir


def numeric_metric_means(result):
    """Return means for numeric benchmark columns, excluding identifiers."""
    frame = result["episodes"] if isinstance(result, dict) else result
    excluded = {"episode", "agent", "environment_ID", "corpus_env_id"}
    columns = [
        column for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    return {column: float(frame[column].mean()) for column in columns}



import random
import numpy as np

def run_random_valid_vmp_episode(env, seed=None, max_steps=None):
    """Run a random valid-placement rollout and return a compact result."""
    _, _ = env.reset(seed=seed)
    max_steps = env.n_vms if max_steps is None else min(max_steps, env.n_vms)

    for _ in range(max_steps):
        candidates = [
            (vm_idx, pm_idx)
            for vm_idx in range(env.n_vms)
            for pm_idx in range(env.n_pms)
            if env.is_valid_action(vm_idx, pm_idx)
            and env._get_pm_of_vm(f"VM_{vm_idx}") != f"PM_{pm_idx}"
        ]
        if not candidates:
            break

        vm_idx, pm_idx = random.choice(candidates)
        obs, reward, terminated, truncated, info = env.step(vm_idx, pm_idx)
        env.render(show=True)
        if terminated or truncated:
            break

    history = pd.DataFrame(getattr(env, "metrics_history", []))
    initial = history.iloc[0].to_dict() if not history.empty else {}
    final = history.iloc[-1].to_dict() if not history.empty else {}
    result = {
        "steps": max(0, len(history) - 1),
        "initial_reward": initial.get("reward", np.nan),
        "final_reward": final.get("reward", np.nan),
    }
    for metric in ("avg_util", "packing_efficiency", "avg_power", "avg_load",
                   "avg_security_risk", "weighted_sum"):
        if metric in initial:
            result[f"initial_{metric}"] = initial[metric]
        if metric in final:
            result[f"final_{metric}"] = final[metric]
    return result, history


def sample_vmp_config(rng, vmp_range=None, *, n_tenants=3):
    """Sample one VMP scenario, including its VM/PM counts, from inclusive ranges.

    ``vmp_range`` uses the keys ``min_VMs``, ``max_VMs``, ``min_PMs``, and
    ``max_PMs``. Each call samples its own dimensions, so corpus environments
    can have different numbers of VMs and PMs.
    """
    ranges = dict(vmp_range or {
        "min_VMs": 10, "max_VMs": 30,
        "min_PMs": 4, "max_PMs": 12,
    })
    required = {"min_VMs", "max_VMs", "min_PMs", "max_PMs"}
    missing = required - set(ranges)
    if missing:
        raise ValueError(f"vmp_range is missing required keys: {sorted(missing)}")

    min_vms, max_vms = int(ranges["min_VMs"]), int(ranges["max_VMs"])
    min_pms, max_pms = int(ranges["min_PMs"]), int(ranges["max_PMs"])
    if min_vms < 1 or max_vms < min_vms:
        raise ValueError("vmp_range must satisfy 1 <= min_VMs <= max_VMs.")
    if min_pms < 1 or max_pms < min_pms:
        raise ValueError("vmp_range must satisfy 1 <= min_PMs <= max_PMs.")

    cfg = {
        "n_vms": int(rng.integers(min_vms, max_vms + 1)),
        "n_pms": int(rng.integers(min_pms, max_pms + 1)),
        "n_tenants": max(1, min(int(n_tenants), min_vms)),
    }

    # Physical-machine capacity ranges
    cfg.update({
        "pm_capacity_mips_min": int(rng.integers(800, 1400)),
        "pm_capacity_mips_max": int(rng.integers(4000, 5600)),
        "pm_capacity_memory_min": int(rng.integers(12, 24)),
        "pm_capacity_memory_max": int(rng.integers(96, 160)),
        "pm_capacity_storage_min": int(rng.integers(80, 160)),
        "pm_capacity_storage_max": int(rng.integers(3000, 6000)),
        "pm_capacity_pe_min": int(rng.integers(2, 6)),
        "pm_capacity_pe_max": int(rng.integers(64, 129)),
    })

    # VM demand ranges
    cfg.update({
        "vm_demand_mips_min": int(rng.integers(150, 400)),
        "vm_demand_mips_max": int(rng.integers(1400, 2200)),
        "vm_demand_memory_min": int(rng.integers(1, 5)),
        "vm_demand_memory_max": int(rng.integers(40, 65)),
        "vm_demand_storage_min": int(rng.integers(5, 20)),
        "vm_demand_storage_max": int(rng.integers(700, 1100)),
        "vm_demand_pe_min": 1,
        "vm_demand_pe_max": int(rng.integers(8, 17)),
    })

    # Network, power, security, and objective parameters
    cfg.update({
        "traffic_density": float(rng.uniform(0.10, 0.40)),
        "min_traffic": float(rng.uniform(0.5, 2.0)),
        "max_traffic": float(rng.uniform(6.0, 14.0)),
        "latency_min": float(rng.uniform(0.5, 2.0)),
        "latency_max": float(rng.uniform(12.0, 30.0)),
        "p_idle": float(rng.uniform(40.0, 80.0)),
        "p_peak": float(rng.uniform(150.0, 240.0)),
        "extra_edge_probability": float(rng.uniform(0.02, 0.20)),
        "vm_vuln_prob_min": 0.01,
        "vm_vuln_prob_max": float(rng.uniform(0.60, 1.0)),
        "pm_escape_prob_min": 0.01,
        "pm_escape_prob_max": float(rng.uniform(0.20, 0.40)),
        "util_coefficient": float(rng.uniform(0.5, 1.5)),
        "packing_coefficient": float(rng.uniform(0.5, 1.5)),
        "power_coefficient": float(rng.uniform(0.5, 1.5)),
        "latency_coefficient": float(rng.uniform(0.5, 1.5)),
        "security_coefficient": float(rng.uniform(0.5, 1.5)),
        "movement_coefficient": float(rng.uniform(-0.10, -0.01)),
        "add_movement_cost": bool(rng.integers(0, 2)),
        "revisiting": False,
    })
    return cfg


def generate_random_maxcut_graphs(
    *,
    count=30,
    num_nodes=20,
    seed=7,
    edge_weight_range=(1, 20),
    max_sweeps=10,
):
    """Generate reproducible random MaxCut graph instances with a fixed node count.

    Node count is held fixed intentionally so a pretrained policy sees a consistent
    observation/action-space shape across instances. Graph family, topology, density,
    and edge weights vary. The returned records contain a NetworkX graph and metadata.
    """
    import networkx as nx
    import numpy as np

    if count < 2:
        raise ValueError("count must be at least 2 for a multi-instance benchmark.")
    if num_nodes < 3:
        raise ValueError("num_nodes must be at least 3.")
    min_weight, max_weight = map(int, edge_weight_range)
    if min_weight < 1 or max_weight < min_weight:
        raise ValueError("edge_weight_range must satisfy 1 <= min_weight <= max_weight.")

    rng = np.random.default_rng(seed)
    records = []
    families = ("erdos_renyi", "barabasi_albert", "watts_strogatz", "dense_erdos_renyi")

    for graph_id in range(count):
        family = families[graph_id % len(families)]
        graph_seed = int(rng.integers(0, 2**31 - 1))

        if family in ("erdos_renyi", "dense_erdos_renyi"):
            p = float(rng.uniform(0.08, 0.35) if family == "erdos_renyi" else rng.uniform(0.45, 0.80))
            graph = nx.erdos_renyi_graph(num_nodes, p, seed=graph_seed)
        elif family == "barabasi_albert":
            m = int(rng.integers(1, max(2, min(5, num_nodes - 1)) + 1))
            graph = nx.barabasi_albert_graph(num_nodes, m, seed=graph_seed)
        else:
            max_k = max(2, min(num_nodes - 1, 10))
            k = int(rng.integers(2, max_k + 1))
            if k % 2:
                k = k + 1 if k + 1 < num_nodes else k - 1
            k = max(2, min(k, num_nodes - 1))
            graph = nx.watts_strogatz_graph(
                num_nodes, k, float(rng.uniform(0.05, 0.45)), seed=graph_seed
            )

        # Ensure every graph is connected without changing its node count.
        if not nx.is_connected(graph):
            components = [list(component) for component in nx.connected_components(graph)]
            for left, right in zip(components, components[1:]):
                graph.add_edge(int(rng.choice(left)), int(rng.choice(right)))

        for u, v in graph.edges:
            graph[u][v]["weight"] = int(rng.integers(min_weight, max_weight + 1))

        graph.graph["sampled_family"] = family
        graph.graph["graph_id"] = graph_id
        records.append({
            "graph_id": graph_id,
            "graph": graph,
            "family": family,
            "nodes": int(graph.number_of_nodes()),
            "edges": int(graph.number_of_edges()),
            "density": float(nx.density(graph)),
            "seed": graph_seed,
            "max_sweeps": int(max_sweeps),
        })

    return records


def build_random_maxcut_env(graph_record, *, max_sweeps=None):
    """Build a fresh ExtendedMaxCutEnv from one generated graph record."""
    from LaGCORL.envs.env_maxcut import ExtendedMaxCutEnv
    sweeps = max_sweeps if max_sweeps is not None else graph_record.get("max_sweeps", 10)
    return ExtendedMaxCutEnv(graph=graph_record["graph"].copy(), max_sweeps=int(sweeps))


def evaluate_agents_on_random_maxcut_graphs(
    agents,
    graph_records,
    *,
    test_agent_on_env,
    output_dir,
    n_episodes=10,
    seed=1000,
    max_sweeps=None,
    verbose=0,
):
    """Evaluate each agent on the same randomized graph suite and save graph-level results.

    Parameters
    ----------
    agents : sequence of mappings
        Each mapping contains ``model``, ``label``, and ``wrapper_factory``.
    graph_records : sequence
        Output from :func:`generate_random_maxcut_graphs`.
    test_agent_on_env : callable
        LaGCORL notebook evaluation API.
    output_dir : str or Path
        Directory for per-agent, per-graph, combined, and summary CSV files.

    Returns
    -------
    dict with ``episodes``, ``per_graph``, ``summary``, and ``paired_differences``.
    """
    import numpy as np
    import pandas as pd

    if not graph_records:
        raise ValueError("graph_records must contain at least one graph.")
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive.")
    if not agents:
        raise ValueError("agents must contain at least one agent specification.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_frames = []

    for agent in agents:
        required = {"model", "label", "wrapper_factory"}
        missing = required - set(agent)
        if missing:
            raise ValueError(f"Agent specification is missing: {sorted(missing)}")

        label = str(agent["label"])
        agent_dir = output_dir / re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._-")
        agent_dir.mkdir(parents=True, exist_ok=True)

        for record in graph_records:
            graph_id = int(record["graph_id"])
            env = build_random_maxcut_env(record, max_sweeps=max_sweeps)
            seeds = [int(seed + graph_id * n_episodes + episode) for episode in range(n_episodes)]
            result = test_agent_on_env(
                agent["model"],
                env,
                n_episodes=n_episodes,
                seeds=seeds,
                wrapper_factory=agent["wrapper_factory"],
                compare_random=False,
                save_dir=str(agent_dir / f"graph_{graph_id:03d}"),
                label=label,
                verbose=verbose,
            )
            frame = result["episodes"].copy()
            frame.insert(0, "test_graph_id", graph_id)
            frame.insert(1, "graph_family", record["family"])
            frame.insert(2, "graph_nodes", record["nodes"])
            frame.insert(3, "graph_edges", record["edges"])
            frame.insert(4, "graph_density", record["density"])
            frame.insert(5, "graph_seed", record["seed"])
            episode_frames.append(frame)

    episodes = pd.concat(episode_frames, ignore_index=True)
    episodes.to_csv(output_dir / "randomized_episode_metrics.csv", index=False)

    excluded = {
        "episode", "environment_ID", "test_graph_id", "graph_nodes",
        "graph_edges", "graph_density", "graph_seed",
    }
    numeric_columns = [
        column for column in episodes.columns
        if column not in excluded
        and column not in {"agent", "graph_family"}
        and pd.api.types.is_numeric_dtype(episodes[column].dtype)
    ]
    group_columns = ["agent", "test_graph_id", "graph_family", "graph_nodes", "graph_edges", "graph_density"]
    per_graph = episodes.groupby(group_columns, dropna=False)[numeric_columns].mean().reset_index()
    per_graph.to_csv(output_dir / "per_graph_summary.csv", index=False)

    summary_rows = []
    for label, group in per_graph.groupby("agent"):
        for metric in numeric_columns:
            values = group[metric].dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            sem = std / np.sqrt(len(values)) if len(values) else float("nan")
            summary_rows.append({
                "agent": label,
                "metric": metric,
                "mean_across_graphs": mean,
                "std_across_graphs": std,
                "ci95_low_normal_approx": mean - 1.96 * sem,
                "ci95_high_normal_approx": mean + 1.96 * sem,
                "n_graphs": int(len(values)),
                "episodes_per_graph": int(n_episodes),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "aggregate_summary.csv", index=False)

    # Paired per-graph differences account for both agents being tested on the same instances.
    paired_rows = []
    labels = list(dict.fromkeys(str(agent["label"]) for agent in agents))
    if len(labels) >= 2:
        first, second = labels[0], labels[1]
        first_frame = per_graph[per_graph["agent"] == first].set_index("test_graph_id")
        second_frame = per_graph[per_graph["agent"] == second].set_index("test_graph_id")
        common_metrics = [m for m in numeric_columns if m in first_frame and m in second_frame]
        common_graphs = first_frame.index.intersection(second_frame.index)
        for metric in common_metrics:
            differences = (
                second_frame.loc[common_graphs, metric].astype(float)
                - first_frame.loc[common_graphs, metric].astype(float)
            ).dropna().to_numpy()
            if not len(differences):
                continue
            mean_diff = float(np.mean(differences))
            std_diff = float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
            sem_diff = std_diff / np.sqrt(len(differences)) if len(differences) else float("nan")
            paired_rows.append({
                "comparison": f"{second} minus {first}",
                "metric": metric,
                "mean_paired_difference": mean_diff,
                "std_paired_difference": std_diff,
                "ci95_low_normal_approx": mean_diff - 1.96 * sem_diff,
                "ci95_high_normal_approx": mean_diff + 1.96 * sem_diff,
                "n_paired_graphs": int(len(differences)),
            })
    paired_differences = pd.DataFrame(paired_rows)
    paired_differences.to_csv(output_dir / "paired_agent_differences.csv", index=False)

    return {
        "episodes": episodes,
        "per_graph": per_graph,
        "summary": summary,
        "paired_differences": paired_differences,
        "output_dir": str(output_dir.resolve()),
    }

# ---------------------------------------------------------------------------
# Benchmark-generic VMP adapter/evaluation helpers
# ---------------------------------------------------------------------------
def make_vmp_gymnasium_env(*, config=None, seed=None, render_mode=None):
    """Create a VMP environment with a Gymnasium-compatible single-integer action.

    ExtendedVMPlacementEnv's native step API accepts ``step(vm_index, pm_index)``.
    This adapter maps a Discrete(n_vms * n_pms) action to that pair and flattens
    the native observation into a fixed-length Box vector. Use fixed VMP_SIZE
    across training/evaluation so the observation and action spaces stay stable.
    """
    import numpy as np
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.spaces import flatten, flatten_space

    try:
        from LaGCORL.envs.env_vmp import ExtendedVMPlacementEnv
    except ImportError:
        from envs.env_vmp import ExtendedVMPlacementEnv

    kwargs = dict(config or {})
    env = ExtendedVMPlacementEnv(**kwargs)
    adapter = VMPGymnasiumAdapter(env, seed=seed, render_mode=render_mode)
    return adapter


class VMPGymnasiumAdapter(gym.Env):
    """Gymnasium adapter for LaGCORL's two-index VM placement action API."""

    metadata = {"render_modes": []}

    def __init__(self, env, seed=None, render_mode=None):
        super().__init__()
        import numpy as np
        import gymnasium as gym
        from gymnasium import spaces
        from gymnasium.spaces import flatten

        self.env = env
        self.n_vms = int(env.n_vms)
        self.n_pms = int(env.n_pms)
        self.action_space = spaces.Discrete(self.n_vms * self.n_pms)
        if not hasattr(env, "observation_space"):
            raise TypeError("ExtendedVMPlacementEnv must expose observation_space for PPO training.")

        # Some VMP configurations return observations whose flattened length does
        # not match flatten_space(env.observation_space). In the reported case,
        # flatten_space predicted 134 values while reset() actually produced 150.
        # Use a real reset observation as the contract presented to SB3, while
        # continuing to flatten subsequent observations with the native space.
        self._flatten = lambda observation: np.asarray(
            flatten(env.observation_space, observation), dtype=np.float32
        ).reshape(-1)
        sample = env.reset(seed=seed)
        raw_obs = sample[0] if isinstance(sample, tuple) and len(sample) == 2 else sample
        self._last_observation = self._flatten(raw_obs)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self._last_observation.shape,
            dtype=np.float32,
        )
        self.render_mode = render_mode

    def reset(self, *, seed=None, options=None):
        result = self.env.reset(seed=seed) if options is None else self.env.reset(seed=seed, options=options)
        if isinstance(result, tuple) and len(result) == 2:
            observation, info = result
        else:
            observation, info = result, {}
        flattened = self._flatten(observation)
        if flattened.shape != self.observation_space.shape:
            raise ValueError(
                "VMP observation length changed after reset: expected "
                f"{self.observation_space.shape}, got {flattened.shape}. "
                "Use a fixed-size VMP configuration and make the native "
                "observation_space consistent with reset()/step() output."
            )
        return flattened, (info if isinstance(info, dict) else {})

    def step(self, action):
        import numpy as np
        action_index = int(np.asarray(action).reshape(-1)[0])
        vm_index, pm_index = divmod(action_index, self.n_pms)
        result = self.env.step(vm_index, pm_index)
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
        elif len(result) == 4:
            observation, reward, done, info = result
            terminated, truncated = bool(done), False
        else:
            raise RuntimeError(f"Unexpected ExtendedVMPlacementEnv.step return: {len(result)} values")
        self._last_observation = self._flatten(observation)
        if self._last_observation.shape != self.observation_space.shape:
            raise ValueError(
                "VMP observation length changed during step: expected "
                f"{self.observation_space.shape}, got {self._last_observation.shape}. "
                "Use a fixed-size VMP configuration and make the native "
                "observation_space consistent with reset()/step() output."
            )
        return (
            self._last_observation,
            float(reward),
            bool(terminated),
            bool(truncated),
            info if isinstance(info, dict) else {},
        )

    def get_metrics(self):
        if hasattr(self.env, "get_metrics"):
            return self.env.get_metrics()
        current = getattr(self.env, "current_metrics", None)
        if isinstance(current, dict):
            return dict(current)
        history = getattr(self.env, "metrics_history", None)
        if history:
            return dict(history[-1])
        return {}

    def close(self):
        if hasattr(self.env, "close"):
            return self.env.close()

    def __getattr__(self, name):
        # Preserve access to project-specific attributes/methods.
        return getattr(self.env, name)


def evaluate_gymnasium_agent_on_envs(
    model,
    env_factories,
    *,
    labels=None,
    n_episodes=5,
    seed=50_000,
    deterministic=True,
    output_dir=None,
):
    """Evaluate an SB3-compatible model on fresh Gymnasium envs and save metrics.

    ``env_factories`` is a sequence of zero-argument callables returning fresh
    environments. Metrics from ``get_metrics()`` are included when available.
    """
    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path

    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive.")
    labels = labels or [f"env_{i}" for i in range(len(env_factories))]
    rows = []
    for env_index, (factory, label) in enumerate(zip(env_factories, labels)):
        for episode in range(n_episodes):
            env = factory()
            obs, _ = env.reset(seed=int(seed + env_index * n_episodes + episode))
            terminated = truncated = False
            episode_reward = 0.0
            episode_length = 0
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += float(reward)
                episode_length += 1
            row = {
                "environment_id": env_index,
                "environment": str(label),
                "episode": episode,
                "episode_reward": episode_reward,
                "episode_length": episode_length,
            }
            try:
                metrics = env.get_metrics()
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        if np.isscalar(value) and not isinstance(value, (str, bytes, bool)):
                            row[f"metric_{key}"] = float(value)
            except Exception:
                pass
            rows.append(row)
            env.close()
    frame = pd.DataFrame(rows)
    summary = frame.groupby("environment", as_index=False).mean(numeric_only=True)
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out / "vmp_episode_metrics.csv", index=False)
        summary.to_csv(out / "vmp_environment_summary.csv", index=False)
        (out / "vmp_summary.json").write_text(
            json.dumps(summary.to_dict(orient="records"), indent=2, default=float),
            encoding="utf-8",
        )
    return {"episodes": frame, "summary": summary}


# Padded GNN discrete-policy helpers retained for the MaxCut variable-graph tutorial.
def build_padded_gnn_discrete_corpus(
    *,
    project_root,
    corpus_dir,
    env_ids,
    train_env_id,
    wrapper_factory,
    log_root,
    run_dir=None,
    environment="maxcut",
    timestamp=None,
    padding_config=None,
):
    """Wrap a corpus and create a CLI-style padded GraphEnvSwitcher.

    Each wrapped environment is serialized as ``<env_id>.pkl`` because the
    native GraphEnvSwitcher expects a flat folder of pickled environments.
    Padding dimensions are computed across all supplied environment IDs, while
    ``train_ids=[train_env_id]`` keeps training episodes on the selected graph.
    """
    import pickle

    try:
        from LaGCORL.envs.env_switcher import GraphEnvSwitcher
    except ImportError:
        from envs.env_switcher import GraphEnvSwitcher

    corpus_dir = Path(corpus_dir).expanduser().resolve()
    log_root = Path(log_root).expanduser().resolve()
    run_dir = Path(run_dir).expanduser().resolve()
    project_root = Path(project_root).expanduser().resolve()
    env_ids = sorted({int(env_id) for env_id in env_ids})
    train_env_id = int(train_env_id)

    if train_env_id not in env_ids:
        raise ValueError(f"Training environment ID {train_env_id} is not in env_ids.")
    if wrapper_factory is None:
        raise ValueError("wrapper_factory is required to create GNN-wrapped environments.")

    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    wrapped_corpus_dir = log_root / "wrapped_corpora" / f"{environment}_gnn_{stamp}"
    wrapped_corpus_dir.mkdir(parents=True, exist_ok=False)

    for env_id in env_ids:
        env = load_corpus_env(corpus_dir, env_id)
        wrapped_env = wrapper_factory(env)
        with (wrapped_corpus_dir / f"{env_id}.pkl").open("wb") as handle:
            pickle.dump(wrapped_env, handle, protocol=pickle.HIGHEST_PROTOCOL)

    eval_ids = [env_id for env_id in env_ids if env_id != train_env_id]
    switcher = GraphEnvSwitcher(
        envs_folder=str(wrapped_corpus_dir),
        train_ids=[train_env_id],
        val_ids=[],
        test_ids=eval_ids,
        algorithm_type="discrete",
        GNN_observations=True,
        pad_spaces=True,
        padding_config=padding_config,
        switch_strategy="sequential",
        switch_interval=1,
        save_switch_logs=False,
        log_path=str(run_dir),
    )
    if padding_config is None:
        max_obs_shape, max_action_shape = switcher.get_padding_config()
        padding_config = {
            "max_obs_shape": max_obs_shape,
            "max_action_shape": max_action_shape,
        }
    else:
        max_obs_shape = padding_config["max_obs_shape"]
        max_action_shape = padding_config["max_action_shape"]


    return {
        "env": switcher,
        "wrapped_corpus_dir": wrapped_corpus_dir,
        "padding_config": padding_config,
        "max_obs_shape": max_obs_shape,
        "max_action_shape": max_action_shape,
        "eval_env_ids": eval_ids,
    }

def evaluate_padded_gnn_policy(
    model,
    *,
    env_id,
    train_env_id,
    wrapped_corpus_dir,
    padding_config,
    log_dir,
    n_episodes=1,
    seed=7,
    seeds=None,
    label="discrete PPO (GNN + padding)",
):
    """Evaluate a padded GNN policy and collect available environment metrics."""
    import numpy as np
    import pandas as pd

    try:
        from LaGCORL.envs.env_switcher import GraphEnvSwitcher
    except ImportError:
        from envs.env_switcher import GraphEnvSwitcher

    eval_env = GraphEnvSwitcher(
        envs_folder=str(Path(wrapped_corpus_dir).expanduser().resolve()),
        train_ids=[int(train_env_id)],
        val_ids=[],
        test_ids=[int(env_id)],
        algorithm_type="discrete",
        GNN_observations=True,
        pad_spaces=True,
        padding_config=padding_config,
        switch_strategy="sequential",
        switch_interval=1,
        save_switch_logs=False,
        log_path=str(Path(log_dir).expanduser().resolve()),
    )

    eval_env.set_mode("test")
    rows = []

    if seeds is None:
        seeds = [int(seed) + i for i in range(int(n_episodes))]
    else:
        seeds = list(seeds)[:int(n_episodes)]

    if len(seeds) != int(n_episodes):
        raise ValueError(
            f"Expected {n_episodes} seeds, received {len(seeds)}."
        )

    def numeric_metrics(mapping):
        """Keep scalar numeric metrics without changing their meaning."""
        result = {}
        if not isinstance(mapping, dict):
            return result

        for key, value in mapping.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                if not isinstance(value, (bool, np.bool_)):
                    if np.isfinite(value):
                        result[str(key)] = float(value)

        return result

    def get_environment_metrics():
        """Read metrics exposed by the switcher or its wrapped environment."""
        candidates = [eval_env]

        # GraphEnvSwitcher may expose its active environment through an
        # implementation-specific attribute. Inspect these common possibilities.
        for attr in ("env", "current_env", "active_env", "unwrapped"):
            try:
                obj = getattr(eval_env, attr, None)
                if obj is not None and not callable(obj):
                    candidates.append(obj)
            except Exception:
                pass

        collected = {}

        for obj in candidates:
            for method_name in ("get_metrics",):
                try:
                    method = getattr(obj, method_name, None)
                    if callable(method):
                        collected.update(numeric_metrics(method()))
                except Exception:
                    pass

            for attr in ("current_metrics",):
                try:
                    collected.update(
                        numeric_metrics(getattr(obj, attr, None))
                    )
                except Exception:
                    pass

            try:
                history = getattr(obj, "metrics_history", None)
                if history is not None and len(history) > 0:
                    collected.update(numeric_metrics(history[-1]))
            except Exception:
                pass

        return collected

    try:
        for episode_idx, episode_seed in enumerate(seeds):
            reset_result = eval_env.reset(seed=int(episode_seed))
            obs = (
                reset_result[0]
                if isinstance(reset_result, tuple)
                else reset_result
            )

            done = False
            total_reward = 0.0
            length = 0
            last_info = {}

            while not done:
                action, _ = model.predict(obs, deterministic=True)

                if isinstance(action, np.ndarray) and action.size == 1:
                    action = action.item()

                step_result = eval_env.step(action)

                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = bool(terminated or truncated)
                else:
                    obs, reward, done, info = step_result
                    done = bool(done)

                if isinstance(info, dict):
                    last_info = info

                total_reward += float(reward)
                length += 1

            row = {
                "agent": label,
                "corpus_env_id": int(env_id),
                "episode": episode_idx + 1,
                "seed": int(episode_seed),
                "episode_reward": total_reward,
                "episode_length": length,
            }

            # Collect numeric information returned by the final environment step.
            row.update(numeric_metrics(last_info))

            # Collect additional metrics exposed by the environment.
            # These values may override info values if the same key is present.
            row.update(get_environment_metrics())

            rows.append(row)

    finally:
        close = getattr(eval_env, "close", None)
        if callable(close):
            close()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Benchmark-independent environment/corpus API
# ---------------------------------------------------------------------------

def _normalize_benchmark(benchmark):
    """Validate and normalize the benchmark selector used by the notebooks."""
    value = str(benchmark).strip().lower()
    aliases = {"max-cut": "maxcut", "vm-placement": "vmp", "vm_placement": "vmp"}
    value = aliases.get(value, value)
    if value not in {"maxcut", "vmp"}:
        raise ValueError(f"Unsupported BENCHMARK={benchmark!r}; choose 'maxcut' or 'vmp'.")
    return value


def sample_vmp_corpus(
    project_root,
    *,
    count=40,
    seed=7,
    max_sweeps=10,
    train_fraction=0.8,
    validation_fraction=0.1,
    test_fraction=0.1,
    vmp_range=None,
    corpus_name=None,
):
    """Generate a split VMP corpus with independently sampled VM/PM counts.

    ``vmp_range`` accepts ``min_VMs``, ``max_VMs``, ``min_PMs``, and ``max_PMs``.
    Every corpus item samples its own VM and PM counts from these inclusive
    ranges, along with randomized resource, traffic, latency, power, and
    objective parameters.
    """
    import pickle
    import yaml
    import numpy as np

    if count < 3:
        raise ValueError("count must be at least 3 to create train/validation/test splits.")
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(value < 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Train/validation/test fractions must be nonnegative and sum to 1.")

    try:
        from LaGCORL.envs.env_vmp import ExtendedVMPlacementEnv
    except ImportError:
        from envs.env_vmp import ExtendedVMPlacementEnv

    root = Path(project_root).expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    corpus_name = corpus_name or f"vmp_sampled_{stamp}"
    corpus_dir = root / "data" / "env_samples" / corpus_name
    corpus_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    vmp_range = dict(vmp_range or {
        "min_VMs": 10, "max_VMs": 30,
        "min_PMs": 4, "max_PMs": 12,
    })
    # Validate once before creating individual environments.
    required_range_keys = {"min_VMs", "max_VMs", "min_PMs", "max_PMs"}
    missing_range_keys = required_range_keys - set(vmp_range)
    if missing_range_keys:
        raise ValueError(
            f"vmp_range is missing required keys: {sorted(missing_range_keys)}"
        )
    min_vms, max_vms = int(vmp_range["min_VMs"]), int(vmp_range["max_VMs"])
    min_pms, max_pms = int(vmp_range["min_PMs"]), int(vmp_range["max_PMs"])
    if min_vms < 1 or max_vms < min_vms:
        raise ValueError("vmp_range must satisfy 1 <= min_VMs <= max_VMs.")
    if min_pms < 1 or max_pms < min_pms:
        raise ValueError("vmp_range must satisfy 1 <= min_PMs <= max_PMs.")

    ids = list(range(count))
    rng.shuffle(ids)
    n_train = min(max(1, int(round(count * train_fraction))), count - 2)
    n_val = min(max(1, int(round(count * validation_fraction))), count - n_train - 1)
    train_ids = sorted(ids[:n_train])
    val_ids = sorted(ids[n_train:n_train + n_val])
    test_ids = sorted(ids[n_train + n_val:])

    metadata = []
    for env_id in range(count):
        config = sample_vmp_config(rng, vmp_range=vmp_range)
        env = ExtendedVMPlacementEnv(**config, max_sweeps=max_sweeps)
        env_dir = corpus_dir / str(env_id)
        env_dir.mkdir()
        with (env_dir / "env.pkl").open("wb") as handle:
            pickle.dump(env, handle, protocol=pickle.HIGHEST_PROTOCOL)
        metadata.append({
            "id": env_id,
            # VMP dimensions are sampled independently for each corpus item.
            "nodes": int(config["n_vms"]) + int(config["n_pms"]),
            "edges": np.nan,
            "density": np.nan,
            "family": "vmp_scenario",
            "n_pms": int(config["n_pms"]),
            "n_vms": int(config["n_vms"]),
            "n_tenants": int(config["n_tenants"]),
            **config,
        })

    split_payload = {
        "training_set": [{"id": i} for i in train_ids],
        "validation_set": [{"id": i} for i in val_ids],
        "test_set": [{"id": i} for i in test_ids],
    }
    with (corpus_dir / "split.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(split_payload, handle, sort_keys=False)

    return {
        "corpus_name": corpus_name,
        "corpus_dir": corpus_dir,
        "train_ids": train_ids,
        "validation_ids": val_ids,
        "test_ids": test_ids,
        "graphs": pd.DataFrame(metadata),
        "benchmark": "vmp",
    }


def sample_benchmark_corpus(
    project_root,
    benchmark,
    *,
    count=40,
    seed=7,
    max_sweeps=10,
    graph_size_range=(8, 60),
    edge_weight_range=(1, 20),
    train_fraction=0.8,
    validation_fraction=0.1,
    test_fraction=0.1,
    vmp_range=None,
    corpus_name=None,
):
    """Create a benchmark-specific corpus behind one stable notebook API.

    The training notebook does not need benchmark-specific branches: all
    subsequent sections consume the same corpus dictionary and environment API.
    """
    benchmark = _normalize_benchmark(benchmark)
    common = dict(
        project_root=project_root,
        count=count,
        seed=seed,
        max_sweeps=max_sweeps,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        corpus_name=corpus_name,
    )
    if benchmark == "maxcut":
        result = sample_maxcut_corpus(
            graph_size_range=graph_size_range,
            edge_weight_range=edge_weight_range,
            **common,
        )
    else:
        result = sample_vmp_corpus(
            vmp_range=vmp_range,
            **common,
        )
    result["benchmark"] = benchmark
    return result


def adapt_benchmark_env(env, benchmark, *, gymnasium_api=False):
    """Adapt a native benchmark environment only when standard Gymnasium API is needed."""
    benchmark = _normalize_benchmark(benchmark)
    if benchmark == "vmp" and gymnasium_api:
        return VMPGymnasiumAdapter(env)
    return env


def load_benchmark_env(corpus_dir, env_id, benchmark, *, gymnasium_api=False):
    """Load a fresh environment, optionally adapting it to a standard Gymnasium API.

    Graph/GAE wrappers receive native environments; standard SB3 policies can use
    the VMP adapter to map a single discrete action to (VM, PM).
    """
    return adapt_benchmark_env(
        load_corpus_env(corpus_dir, env_id),
        benchmark,
        gymnasium_api=gymnasium_api,
    )


def benchmark_environment_name(benchmark):
    """Return the canonical environment key used by LaGCORL config files."""
    return _normalize_benchmark(benchmark)


def newest_benchmark_artifact(log_root, agent_type, benchmark, *, artifact="model.zip"):
    """Find the newest policy artifact in flat or legacy pretrained run layouts.

    Searches both <run>/<artifact> and any nested locations such as
    <run>/checkpoints/<artifact>, while preferring the shallowest copy in a run.
    """
    benchmark = _normalize_benchmark(benchmark)
    root = Path(log_root).expanduser().resolve()
    agent_root = root / str(agent_type)
    if not agent_root.is_dir():
        return None

    run_candidates = []
    for run_dir in agent_root.glob(f"{benchmark}_*"):
        if not run_dir.is_dir():
            continue
        matches = [path for path in run_dir.rglob(artifact) if path.is_file()]
        if matches:
            selected = min(
                matches,
                key=lambda path: len(path.relative_to(run_dir).parts),
            )
            run_candidates.append((run_dir.stat().st_mtime, selected.stat().st_mtime, selected))

    return max(run_candidates, key=lambda item: (item[0], item[1]))[2] if run_candidates else None


def newest_gae_bundle(log_root, benchmark):
    """Find the newest GAE encoder root in flat or legacy pretrained layouts.

    Returns the directory whose immediate children include encoder graph
    directories (for example allocation_graph, latency_graph, or graph).
    """
    benchmark = _normalize_benchmark(benchmark)
    root = Path(log_root).expanduser().resolve()
    gae_root = root / "gae"
    if not gae_root.is_dir():
        return None

    run_candidates = []
    for run_dir in gae_root.glob(f"{benchmark}_*"):
        if not run_dir.is_dir():
            continue

        possible_dirs = [run_dir] + [path for path in run_dir.rglob("*") if path.is_dir()]
        encoder_roots = []
        for path in possible_dirs:
            try:
                if any(
                    child.is_dir() and child.name.endswith("graph")
                    for child in path.iterdir()
                ):
                    encoder_roots.append(path)
            except OSError:
                continue

        if encoder_roots:
            # If multiple folders qualify, use the shallowest root for this run.
            selected = min(
                encoder_roots,
                key=lambda path: len(path.relative_to(run_dir).parts),
            )
            run_candidates.append((run_dir.stat().st_mtime, selected))

    return max(run_candidates, key=lambda item: item[0])[1] if run_candidates else None