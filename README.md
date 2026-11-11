# Tutorial: Reinforcement Learning for Graph Combinatorial Optimization
<div align="center">

<a href="https://logconference.org/"><strong>LoG 2026</strong></a> · Northeastern University, Boston
<br>

<strong>
<a href="https://terranovafr.github.io/">Franco Terranova</a>
</strong> · UL, INRIA, LORIA, CNRS
&nbsp; | &nbsp;
<strong>
<a href="https://victor.darvariu.me/">Victor-Alexandru Darvariu</a>
</strong> · University of Sheffield

<br>

<a href="mailto:franco.terranova@inria.fr">franco.terranova@inria.fr</a>
&nbsp; | &nbsp;
<a href="mailto:victord@robots.ox.ac.uk">v.darvariu@sheffield.ac.uk</a>

<br>

<img src="https://thumb.wikimedia.org/wikipedia/commons/thumb/5/5b/Logo_Universit%C3%A9_de_Lorraine.svg/3840px-Logo_Universit%C3%A9_de_Lorraine.svg.png?utm_source=fr.wikipedia.org&utm_campaign=index&utm_content=thumbnail" height="50" alt="Université de Lorraine"/>
&nbsp;
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/95/Inr_logo_rouge.svg/1280px-Inr_logo_rouge.svg.png" height="50" alt="INRIA"/>
&nbsp;
<img src="https://upload.wikimedia.org/wikipedia/fr/6/6e/Logo_loria_abrege_couleur.png" height="50" alt="LORIA"/>
&nbsp;
<img src="https://www.pepr-cybersecurite.fr/wp-content/uploads/2023/09/pep-cybersecurite-550x250-1.png" height="50" alt="SuperViZ"/>
&nbsp;
<img src="https://sheffield.pressbooks.pub/app/uploads/2022/12/UOSLogo_Primary_Violet_RGB.png" height="50" alt="University of Sheffield"/>

<br><br>

<a href="https://github.com/terranovafr/LaGCO-RL">💻 <strong>LaGCO-RL GitHub Repository</strong></a>

</div>

This repository provides a notebook-based walkthrough of graph-based reinforcement learning using **maximum cut** and **virtual machine placement** as benchmarks and the LaGCO-RL library as a reference implementation.

It demonstrates environment interaction, domain randomization, graph autoencoder (GAE/VGAE) representations, agent training, and evaluation of pretrained agents on held-out scenarios.

## Tutorial logic

The notebooks follow a setup → understand environments → train → evaluate workflow:

1. Set up the LaGCORL codebase and install any available pretrained artifacts.
2. Inspect the MaxCut and VMP environments and run simple random-action rollouts.
3. Select a benchmark, generate a scenario corpus, compare a standard discrete PPO baseline with GNN-based approaches, train agents, and compare their results on held-out scenarios.
4. Run the evaluation workflow independently with compatible pretrained checkpoints, without having to retrain the agents.

## Notebooks

| Notebook | What you will find |
|---|---|
| `0_setup.ipynb` | Configures the LaGCORL codebase, installs notebook API helpers, and downloads configured pretrained artifacts. When using Docker, the Python environment and dependencies are already installed in the container. |
| `1_graph_envs.ipynb` | Introduces the two environments. For MaxCut, it builds weighted NetworkX graphs and runs random valid-action rollouts. For VMP, it uses a fixed problem size and explores domain randomisation by varying scenario parameters while the environment generates its graphs internally. These are environment smoke tests, not learned-agent performance claims. |
| `2_train.ipynb` | Main experiment workflow. Set `BENCHMARK = "maxcut"` or `"vmp"` once, generate a randomized corpus, inspect why an unpadded discrete policy may not transfer across size-dependent spaces, train a GAE/VGAE encoder, and train/evaluate padded GNN-observation PPO, iterative iDQN, and projection PPO. Results are compared on held-out scenarios. |
| `3_test.ipynb` | Standalone evaluation of available pretrained agents. It creates a fresh randomized test suite, searches `logs/` for compatible checkpoints and GAE bundles, loads each policy with the required wrappers, evaluates compatible agents on the same scenarios, and exports combined results. It evaluates only artifacts that are actually available and compatible. |

## Getting started

### 1. Install Docker

This tutorial is ran through Docker. This ensures that all participants use the same Python version and package dependencies, avoiding differences between local Python or Conda environments.

Install Docker Desktop:

- **Windows/macOS:** [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **Linux:** [Install Docker Engine](https://docs.docker.com/engine/install/)

Make sure Docker is running before proceeding.


### 2. Build the Docker image

Build the environment using:

```bash
docker compose build
```

This installs the required Python dependencies and configures JupyterLab inside the container.

If you are on MacOS Tahoe on a Mac with an Arm chip (M1-M5), you may be affected by an unresolved [bug in Docker Desktop](https://github.com/docker/desktop-feedback/issues/592). This may result in timed out connections to the Jupyter server despite successful setup. If this occurs, ensure that your image is natively built for the `arm64` architecture by uncommenting the `platform: linux/arm64` line in `docker-compose.yml`.

**You do not need to create a Conda environment or install packages using `requirements.txt` manually.** The Docker image already contains the dependencies required by the tutorial.

If the Dockerfile or dependency configuration changes, rebuild the image using:

```bash
docker compose build --no-cache
```

### 4. Start JupyterLab

Launch the container in the background:

```bash
docker compose up -d
```

Open your browser and navigate to:

**http://localhost:8889**

JupyterLab will open with access to the project files.

### 5. Run the setup notebook

Open `0_setup.ipynb` and follow its instructions to configure the LaGCORL codebase, install the notebook helper APIs, and download any configured pretrained artifacts.

When using Docker, the Python environment and dependencies are already installed. **Do not create or activate another Conda environment inside the container, or switch to a separate Python kernel.** Use the default Python kernel provided by JupyterLab.

### 6. Run the tutorial notebooks

Run the notebooks in the following order:

`1_graph_envs.ipynb` — inspect and test the environments.
`2_train.ipynb` — train and compare reinforcement learning agents.
`3_test.ipynb` — independently evaluate available pretrained agents.

Make sure to execute the setup notebook before running the training or testing notebooks.

## Stopping and restarting Docker

To stop the container:

```bash
docker compose down
```

To restart it later:

```bash
docker compose up -d
```

To rebuild the image after changing the Dockerfile or installed dependencies:

```bash
docker compose down
docker compose build
docker compose up -d
```

Your project files and experiment results are stored on your local machine and are not lost when the container is stopped.

## Choose or change the benchmark

In the training and testing notebooks, set the `BENCHMARK` variable to `"maxcut"` or `"vmp"` near the beginning.

The helper API handles benchmark-specific environment creation so that most experiment code is shared.

The training notebook generates the corpus once and reuses it for encoder training and agent comparisons. By default, corpus environment `0` is the training instance and the remaining instances are held out for evaluation.

Re-run the corpus-generation section only when you intentionally want a new set of scenarios.

## Where results and artifacts are saved

Experiments write outputs under `logs/`, generally grouped by agent and benchmark with a timestamped run directory.

Depending on the experiment, look for:

- `checkpoints/` — saved model weights (typically `model.zip`) and metadata.
- `tensorboard/` — TensorBoard event files for training curves.
- `train_metrics/` — training-related metrics and wrapper logs.
- `test_metrics/` — per-episode benchmark metrics and aggregate summaries, commonly CSV/JSON/YAML files.
- `gae/` — GAE/VGAE run outputs and graph encoder checkpoints.
- `pretrained_evaluation/` — outputs from the standalone pretrained-agent evaluation notebook.

The precise files depend on which runs completed. Open the corresponding timestamped run directory to find its results.

**Important:** The `logs/` directory is shared between Docker and your local project directory. This means that experiment results remain accessible on your computer even after stopping or removing the Docker container.

TensorBoard event files can be plotted from the notebooks using the helper utilities. The testing API uses LaGCORL's benchmark-specific `get_metrics()` pathway, and can also measure action/episode timing.

## Helper API files

Keep these files alongside the notebooks as instructed in the setup notebook; the installer also places project-facing APIs under `LaGCORL/agents/`.

- `notebook_helpers_api.py` — reusable corpus generation/loading, run-directory and artifact helpers, TensorBoard plotting, and multi-environment benchmark aggregation.
- `notebook_train_api.py` — trains a supported SB3/LaGCORL agent directly from a notebook on a supplied environment.
- `notebook_train_gae_api.py` — invokes LaGCORL's native GAE/VGAE trainer, loads encoder checkpoints, and creates the GAE-backed environment-wrapper factory.
- `notebook_test_api.py` — evaluates policies using LaGCORL benchmark metrics and optionally records action/episode timing.

<br>
<div align="center">
<img src="https://logconference.org/media/logo_conf_below_512_512_hu0b3b5122a793f6cbe7d94790c745769e_24489_1200x0_resize_lanczos_3.png" height="200" alt="LoG 2026"/>
</div>