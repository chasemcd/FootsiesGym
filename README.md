# FootsiesGym

Implementation of the Footsies Unity game as a reinforcement learning environment.

## Installation

```bash
conda create -n footsiesgym python=3.10
conda activate footsiesgym
pip install -r requirements.txt
pre-commit install
```

On a Mac, you may need to ensure you have `cmake` installed. You can install it using Homebrew:

```bash
brew install cmake
```




## Training

### Game Servers

Before training, you'll need to launch the headless game servers. A script is provided to do so in `scripts/start_local_servers.sh`, but you must first unpack the binaries that are included in `binaries/footsies_linux_server_021725.zip`. The `start_local_servers.sh` script assumes that you have unpacked the contents into `~/footsies_binaries`.

```bash
./scripts/start_local_servers.sh <num-train-servers> <num-eval-servers>
```

The two arguments correspond to `num_env_runners` and `evaluation_num_env_runners`, which can be specified in `experiments/config.py`. You must launch a corresponding
number of servers for each. The `start_local_servers.sh` script will start training servers from a specified port number (50051) and increment from there,
and evaluation servers from a specified port number (40051) and increment from there. The environment will use workers indices to determine which game server to connect to.

_Note_: All of my training runs so far have been with an RTX 3090 and 24-core Threadripper, where I've been able to run 40 training environments and 5 evaluation environments. With this setup, I see 40-65% utilization per-core.


### Experiments

The current setup is an LSTM Torch network trained through self-play with APPO (old stack). To launch a new experiment, run:

```bash
python -m experiments.train --experiment-name <experiment-name>
```

You can optionally add the `--debug` flag to use only a single env runner and to use `local_mode`. If you do so, using the experiment name `test` will avoid restoring or saving too man new experiments in `ray_results`.


The full experiment is configured in `experiments/experiment.py`


There is a first pass at new stack migration in `experiments/experiment_rlmodule.py` (and a corresponding network in `models/rl_modules/lstm_module.py`.). I believe that they are outdated and need to be updated.


## Visualizing a Policy

To visualize gameplay, you'll need to run the windowed version of the game. This repository includes the windowed and headless Linux builds (TODO: add Windows/Mac windowed builds).
1. Unpack the windowed build binaries (`binaries/footsies_linux_windowed_021725.zip`) to your preferred location.
2. Add the trained policy specification (unless you want to use `random` or `noop`) to the to the `ModuleRepository` in `components/module_repository.py`. This assumes you've trained a policy and it's stored in `~/ray_results/<experiment-name>/checkpoint_<checkpoint-number>/checkpoint-<checkpoint-number>`.
```python
        FootsiesModuleSpec(
            module_name="<policy-nickname>,
            experiment_name="<experiment-name>",
            trial_id="<trial-id>",  # it the experiment has multiple trials, specify the trial id
            checkpoint_number=-1,  # -1 for latest, otherwise specify a checkpoint number
        ),
```
3. Run the game with `./footsies_linux_windowed_021725 --port 80051` (or any alternative port you've specified in `scripts/local_inference.py`).
4. Specify policies in `scripts/local_inference.py` using the `MODULES` variable. If you'd like to play, set `"p1"` to be `"human"`. Run `python -m scripts.local_inference`.

---
###  gRPC / Protobuf Updates

If you are updating the proto, you'll need to generate `Footsies.cs` and `FootsiesGrpc.cs`. This repo includes what you need on a __Windows__ machine:

```
.\protoc\bin\protoc.exe --csharp_out=.\env\game\proto\ --grpc_out=.\env\game\proto\ --plugin=protoc-gen-grpc=.\plugins\grpc_csharp_plugin.exe .\env\game\proto\footsies_service.proto
```

The corresponding python files are also necessary (make sure to `pip install grpcio-tools grpcio`):

```
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. .\env\game\proto\footsies_service.proto
```
