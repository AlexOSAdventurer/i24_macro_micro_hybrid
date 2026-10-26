# I-24 Hybrid Macro-Micro Traffic Simulation

A macroscopic LWR (cell transmission model) simulation of a ~1 mile stretch of I-24 (mile markers 60.6 to 61.6,
eastbound road `1` and westbound road `2`, four lanes each), with an embedded, moving microscopic region that is
simulated by either SUMO or CARLA. Both are initialized from and scored against the I-24 MOTION "INCEPTION"
trajectory dataset.

This README covers the full reproduction pipeline:

1. [Download the CARLA / Unreal Engine sources and build Unreal on the host](#1-download-the-carla--unreal-engine-sources-and-build-unreal-on-the-host)
2. [Build the Docker image](#2-build-the-docker-image)
3. [Launch the container](#3-launch-the-container)
4. [Build CARLA and the CARLA Python package](#4-build-carla-and-the-carla-python-package)
5. [Import the I-24 map and launch CARLA](#5-import-the-i-24-map-and-launch-carla)
6. [Download and prepare the I-24 MOTION dataset](#6-download-and-prepare-the-i-24-motion-dataset)
7. [Run the simulations (`sim_runs.py`)](#7-run-the-simulations-sim_runspy)
8. [Generate the results table (`sim_analysis.py`)](#8-generate-the-results-table-sim_analysispy)

## Repository layout

| Path | Purpose |
|---|---|
| `i24motion_macro_micro/` | The simulation code, the per-day dataset configs (`config/`), the SUMO network (`sumo/`), and the run and analysis scripts |
| `download_i24motion_to_carla.sh` | Clones the `macro_coupled` branch of the [`i24motion_to_carla_helpers`](https://github.com/AlexOSAdventurer/i24motion_to_carla_helpers/tree/macro_coupled) repo into `i24motion_to_carla/` |
| `build_container.sh` | Builds the `carla-base:ue4-<distro>` base image, the `i24-sim-<distro>` development image, and the `i24-sim-volume-<distro>` home volume |
| `container.Dockerfile` | The development image: the CARLA build toolchain, Python 3.12 with this project's requirements, SUMO 1.27.1, LaTeX, and GDAL/PDAL |
| `launch_container.sh` | Starts the development container with the GPU, X11, and the source trees mounted |
| `.gitignore` | Ignores `i24motion_to_carla/`, the datasets, and run outputs. Several of these are tens of GB, so never `git add -A` them. The comments in `.gitignore` explain why. |

Apart from the Unreal Engine build (step 1), everything runs **inside the container**. The host needs Docker, the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html),
an NVIDIA driver, the Unreal Engine build dependencies, and an X server for the CARLA editor window.

Disk space: the Unreal Engine and CARLA builds take roughly 150 GB together, and the raw dataset is ~20 GB per day
(10 days). Preprocessing products and the results database add more on top.

## 1. Download the CARLA / Unreal Engine sources and build Unreal on the host

Unreal Engine 4.26 (CARLA fork) is only accessible after you
[link your GitHub account to Epic Games](https://www.unrealengine.com/en-US/ue-on-github). Do that first, and create a
GitHub personal access token for that account.

From the repository root:

```bash
./download_i24motion_to_carla.sh          # clones the helper repo (branch macro_coupled) into ./i24motion_to_carla
cd i24motion_to_carla
./download_carla.sh                       # CARLA 0.9.16, branch 0.9.16_TrafficWaves -> ./carla-0.9.16
./download_Unreal.sh <GITHUB_TOKEN>       # Unreal Engine 4.26 (carla branch)  -> ./UnrealEngine_4.26
cd ..
```

This produces the layout that the build and launch scripts expect:

```
i24motion_to_carla/
├── carla-0.9.16/          -> /workspaces/carla           in the container
├── UnrealEngine_4.26/     -> /workspaces/unreal-engine   in the container
└── i24motion_to_carla/    -> /workspaces/i24motion_to_carla
```

Build Unreal Engine **on the host, outside Docker**. The container mounts the prebuilt engine and does not build it.
This is a one-time build that takes hours. Follow the
[CARLA guide](https://carla.readthedocs.io/en/0.9.16/build_linux/#building-unreal-engine), which comes down to:

```bash
cd i24motion_to_carla/UnrealEngine_4.26
./Setup.sh && ./GenerateProjectFiles.sh && make
cd ../..
```

The build is done when `i24motion_to_carla/UnrealEngine_4.26/Engine/Binaries/Linux/UE4Editor` exists.

## 2. Build the Docker image

The script resolves all paths relative to its own location, so it can be run from any directory:

```bash
./build_container.sh
```

This builds:

- `carla-base:ue4-24.04` from CARLA's `Util/Docker/Base.Dockerfile`
- `i24-sim-24.04` from `container.Dockerfile`, with your host UID/GID, so files written in the container stay owned by you
- the `i24-sim-volume-24.04` Docker volume, mounted as the container user's home (`/home/carla`). Anything installed with
  `pip install --user`, such as the CARLA wheel in step 4, lives there and survives container restarts.

Useful options (`./build_container.sh --help`):

- `--ubuntu-distro 22.04` builds the 22.04 variant. The default is **24.04** (Python 3.12).
- `--force-rebuild` rebuilds without the Docker cache and **deletes and recreates the home volume**, so you will need to
  reinstall the CARLA wheel afterwards.

## 3. Launch the container

From any directory:

```bash
./launch_container.sh            # add --rebuild to run build_container.sh first
```

The container is named `i24-sim-24.04` and starts in `/workspaces/carla`. It uses host networking, so CARLA's port
2000 and the Dash viewer's port 8050 are reachable from the host. Mounts:

| Host | Container |
|---|---|
| `i24motion_to_carla/carla-0.9.16` | `/workspaces/carla` |
| `i24motion_to_carla/UnrealEngine_4.26` | `/workspaces/unreal-engine` |
| `i24motion_to_carla/i24motion_to_carla` | `/workspaces/i24motion_to_carla` |
| `i24motion_macro_micro` | `/workspaces/i24motion_macro_micro` |

The container is started with `--rm`, so it is removed when its first shell exits. To open more shells, for example
one for CARLA and one for the simulation, run:

```bash
docker exec -it i24-sim-24.04 bash
```

If the Unreal editor cannot open a window, allow local X clients on the host with `xhost +local:` and make sure
`DISPLAY` is set in the shell you launch from.

## 4. Build CARLA and the CARLA Python package

These builds run in the container, against the Unreal Engine built on the host in step 1. They are one-time builds,
and their outputs are written to the mounted CARLA tree on the host, so they persist.

**CARLA** (see the [CARLA Linux build docs](https://carla.readthedocs.io/en/0.9.16/build_linux/) for details):

```bash
cd /workspaces/carla
./Update.sh                 # downloads CARLA's content assets (large)
make PythonAPI              # builds LibCarla and the Python API for python3 (3.12 in the 24.04 image)
```

**CARLA Python package.** `make PythonAPI` writes a wheel to `PythonAPI/carla/dist/`. Install the one matching the
container's Python (`cp312` for the 24.04 image):

```bash
python3 -m pip install --user /workspaces/carla/PythonAPI/carla/dist/carla-0.9.16-cp312-*.whl
python3 -c "import carla; print(carla.__file__)"    # sanity check
```

The `i24motion_to_carla` helper package needs no install, because the simulation code adds `/workspaces` to
`sys.path`.

## 5. Import the I-24 map and launch CARLA

Download and import the I-24 map once:

```bash
cd /workspaces/carla
./I24_CARLA_download.sh      # fetches Import/FinalMapFinal.tar.gz
./I24_CARLA_import.sh        # imports it into the Unreal project
```

The archive is named `FinalMapFinal.tar.gz`, but the map it contains is **`FinalMapI24Mini`**. After importing it
lives at `/Game/Carla/Maps/FinalMapI24Mini/FinalMapI24Mini`.

Launch CARLA. Leave this shell running:

```bash
cd /workspaces/carla
make launch
```

In the Unreal editor, open `Content/Carla/Maps/FinalMapI24Mini/FinalMapI24Mini.umap` and press **Play**. That starts
the CARLA server on `localhost:2000`. The simulation connects to whatever world is loaded and switches it to
synchronous mode itself, so the map must be open and playing before you run the CARLA simulations in step 7.

SUMO needs no setup. It is pip-installed in the image, and the corridor network is checked in at
`i24motion_macro_micro/sumo/i24_corridor.net.xml`.

## 6. Download and prepare the I-24 MOTION dataset

### Download

Create an account at [i24motion.org](https://i24motion.org/) and accept the data use agreement. From
[i24motion.org/access_data](https://i24motion.org/access_data), download the ten INCEPTION days and unzip each one.
Then rename each day's trajectory JSON to its date and place it in
`i24motion_macro_micro/i24_motion_source_data/`:

| Date | INCEPTION file | Rename to |
|---|---|---|
| 2022-11-21 | `637b023440527bf2daa5932f__post1.json` | `2022-11-21.json` |
| 2022-11-22 | `637c399add50d54aa5af0cf4__post2.json` | `2022-11-22.json` |
| 2022-11-23 | `637d8ea678f0cb97981425dd__post3.json` | `2022-11-23.json` |
| 2022-11-24 | `637f0d5f78f0cb97981425de__post4.json` | `2022-11-24.json` |
| 2022-11-25 | `6380728cdd50d54aa5af0cf5__post5.json` | `2022-11-25.json` |
| 2022-11-28 | `638450a3dd50d54aa5af0cf6__post8.json` | `2022-11-28.json` |
| 2022-11-29 | `63858a2cfb3ff533c12df166__post9.json` | `2022-11-29.json` |
| 2022-11-30 | `6386d89efb3ff533c12df167__post10.json` | `2022-11-30.json` |
| 2022-12-01 | `63882be478f0cb97981425df__post11.json` | `2022-12-01.json` |
| 2022-12-02 | `63898d48d430891009401330__post12.json` | `2022-12-02.json` |

Each day has a config in `i24motion_macro_micro/config/<date>.json`. It sets the time window (4 h from 06:00 local
time), the road geometry and mile-marker mapping, and where the preprocessing outputs go
(`preprocessing_data/<date>/` and `simulation_dataset/<date>/`). Every script below loops over all configs in
`config/`. To work with a subset of days, move the other configs out of that folder.

If you use the data, cite Gloudemans et al., *I-24 MOTION: An instrument for freeway traffic science*, Transportation
Research Part C 155 (2023), as the data use agreement requires.

### Prepare

Run these in the container, in this order. Each step reads the previous step's output, and each is a one-time cost:

```bash
cd /workspaces/i24motion_macro_micro

# 1. Split each day's trajectories into per-road, per-lane parquet files
#    -> preprocessing_data/<date>/road_data/road{1,2}lane{-1..-4}.parquet
python3 process_i24_motion_data.py

# 2. Aggregate the trajectories into 100 m x 1 s macroscopic cells (Edie's definitions)
#    -> preprocessing_data/<date>/road_{1,2}/...
python3 i24_motion_macro.py

# 3. Build the simulation network and the micro/macro ground truth for each day
#    -> simulation_dataset/<date>/{network.json, micro.parquet, macro.parquet}
python3 simulation_dataset.py
```

## 7. Run the simulations (`sim_runs.py`)

`sim_runs.py` has three experiment functions. Each loops over every dataset config and stores its rollouts in the
SQLite database `run_data/results_final.db`:

| Function | Run IDs | What it runs |
|---|---|---|
| `run_lwr_triangular()` | `lwr_triangular_<date>.json` | Fully macroscopic LWR over the whole 4 h window. This is the baseline. |
| `run_sumo()` | `sumo_road(<r>)_timeorigin(<t>)_episodelength(<n>)_<date>.json` | Hybrid LWR + SUMO bubble, for both roads, in 6-minute episodes over the last 3 h |
| `run_carla()` | `carla_road(...)...` | The same episodes, with CARLA as the micro engine. Needs CARLA running (step 5). |

Run this in your terminal to generate the simulation data:

```bash
cd /workspaces/i24motion_macro_micro
python3 sim_runs.py
```

Notes:

- **Start from an empty database.** Run IDs are deterministic, and storing an ID that already exists raises an error.
  The SUMO and CARLA loops catch that error, print `... Skipping it.`, and move on, so a re-run into an existing
  database silently keeps the old results. For a fresh reproduction, move the old database aside first:
  `mv run_data/results_final.db run_data/results_final.old.db`. The analysis also expects exactly one LWR baseline per
  dataset.
- An episode that throws for any other reason is also skipped with the same message. Check the log for `Skipping it.`
  lines before trusting the table.
- The calibrated triangular fundamental diagram (`v_f`, `w`, `rho_j`) is hard-coded in `sim_runs.py` and in
  `simulation_dataset.py`.
- To browse a stored run interactively, `sim_demo.py` serves a Dash viewer on port 8050.

## 8. Generate the results table (`sim_analysis.py`)

```bash
cd /workspaces/i24motion_macro_micro
python3 sim_analysis.py
```

This reads `run_data/results_final.db` read-only and writes **`results.tex`**: a LaTeX `table*` of the density error
upstream of the microscopic bubble against the empirical data. It reports the L1 error and the Wasserstein distance,
with mean, median, SD, and max for each dataset and over all datasets. The rows are:

- **SUMO** and **CARLA**: the hybrid runs
- **LWR (SUMO reg.)** and **LWR (CARLA reg.)**: the LWR baseline scored on exactly the same cells and timesteps as
  the corresponding hybrid runs, so each pair is a like-for-like comparison

If either the SUMO or the CARLA runs are missing, their rows show `--`. The table needs `\usepackage{booktabs}` in
the including document. To change its layout (scaling, compact headers, rows vs. columns), see the parameters of
`generate_latex_for_metrics` in `sim_analysis.py`. TeX Live is installed in the container, so you can compile the
paper there too.