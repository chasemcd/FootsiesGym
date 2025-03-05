import gzip
import json
import os
from typing import Any

import numpy as np
import ray
from ray.rllib import env as rllib_env
from ray.rllib import policy as rllib_policy
from ray.rllib.examples._old_api_stack.policy import random_policy
from ray.rllib.utils import policy as rllib_policy_utils
from ray.rllib.utils import typing as rllib_typing

from footsies import footsies_env
from utils import module_repository

NUM_GAMES = 7_500
DATA_NAME = "test-trajectory-export"
REMOTES = 30
GAMES_PER_REMOTE = NUM_GAMES // REMOTES
MODULES_TO_SAMPLE = [
    "4fs-16od-082992f-0.01to0.001-vsrandom",
    "4fs-16od-082992f-0.03to0.01-sp",
    "4fs-16od-13c7f7b-0.05to0.01-sp-00",
    "4fs-16od-13c7f7b-0.05to0.01-sp-01",
    "4fs-16od-13c7f7b-0.05to0.01-sp-02",
    "4fs-16od-13c7f7b-0.05to0.01-sp-03",
    "random",
]


def rollout_episode(
    env: rllib_env.MultiAgentEnv,
    env_id: str | int,
    modules: dict[rllib_typing.AgentID, rllib_policy.Policy],
    game_id: int | str,
) -> dict[str, Any]:

    obs, _ = env.reset()

    terminateds = {"__all__": False}
    truncateds = {"__all__": False}

    data = {
        "last_encoding": [],
        "p1_obs": [],
        "p1_actions": [],
        "p1_rewards": [],
        "p1_terminateds": [],
        "p1_truncateds": [],
        "p2_obs": [],
        "p2_actions": [],
        "p2_rewards": [],
        "p2_terminateds": [],
        "p2_truncateds": [],
        "t": [],
        "prop_T": [],
    }

    frame = 0
    while not terminateds["__all__"] and not truncateds["__all__"]:
        actions = {}
        for agent_id, ob in obs.items():

            if isinstance(modules[agent_id], random_policy.RandomPolicy):
                actions[agent_id] = env.action_space[agent_id].sample()
            else:
                actions[agent_id], *_ = (
                    rllib_policy_utils.local_policy_inference(
                        modules[agent_id],
                        env_id=env_id,
                        agent_id=agent_id,
                        obs=ob,
                    )[0]
                )
        frame += 1

        obs_new, reward, terminateds, truncateds, _ = env.step(actions)

        data["last_encoding"].append(
            np.concatenate([*env.encoder.get_last_encoding().values()])
        )
        data["p1_obs"].append(obs["p1"])
        data["p1_actions"].append(actions["p1"])
        data["p1_rewards"].append(reward["p1"])
        data["p1_terminateds"].append(terminateds["p1"])
        data["p1_truncateds"].append(truncateds["p1"])
        data["p2_obs"].append(obs["p2"])
        data["p2_actions"].append(actions["p2"])
        data["p2_rewards"].append(reward["p2"])
        data["p2_terminateds"].append(terminateds["p2"])
        data["p2_truncateds"].append(truncateds["p2"])
        data["t"].append(frame)

        obs = obs_new

    p1_win = reward["p1"] >= 1.0
    data["p1_win"] = [p1_win] * len(data["t"])
    data["game_id"] = [game_id] * len(data["t"])
    data["prop_T"] = [t / len(data["t"]) for t in data["t"]]

    return data


@ray.remote
def collect_n_trajectories(remote_id: int, port: int):
    env = footsies_env.FootsiesEnv(
        config={
            "frame_skip": 4,
            "observation_delay": 16,
            "max_t": 4000,
            "port": port,
        }
    )

    sampled_module_indices = np.random.choice(
        range(len(MODULES_TO_SAMPLE)), 2, replace=True
    )

    sampled_modules = [MODULES_TO_SAMPLE[i] for i in sampled_module_indices]

    modules = {
        agent_id: module_repository.ModuleRepository.get(policy_id)
        for agent_id, policy_id in zip(env.agents, sampled_modules)
    }

    def convert_to_serializable(data):
        for key, value in data.items():
            if isinstance(value, list):
                data[key] = [
                    (
                        v.item()
                        if isinstance(v, np.integer)
                        else v.tolist() if isinstance(v, np.ndarray) else v
                    )
                    for v in value
                ]
            elif isinstance(value, np.ndarray):
                data[key] = value.tolist()
            elif isinstance(value, np.integer):
                data[key] = value.item()
        return data

    # Collect all trajectories for this remote
    all_trajectories = []
    for game_num in range(GAMES_PER_REMOTE):

        sampled_module_indices = np.random.choice(
            range(len(MODULES_TO_SAMPLE)), 2, replace=True
        )

        sampled_modules = [
            MODULES_TO_SAMPLE[i] for i in sampled_module_indices
        ]

        modules = {
            agent_id: module_repository.ModuleRepository.get(policy_id)
            for agent_id, policy_id in zip(env.agents, sampled_modules)
        }

        data = rollout_episode(
            env,
            env_id=remote_id,
            modules=modules,
            game_id=f"{remote_id}-{game_num}",
        )
        data = convert_to_serializable(data)
        all_trajectories.append(data)

    # Save all trajectories from this remote in one file
    os.makedirs(f"FootsiesTrajectories/{DATA_NAME}/", exist_ok=True)
    json_data = json.dumps(all_trajectories).encode("utf-8")
    # Write directly using gzip.open() - no need for separate compression
    with gzip.open(
        f"FootsiesTrajectories/{DATA_NAME}/remote_{remote_id}.json.gz",
        "wt",  # Changed to text mode
        encoding="utf-8",
    ) as f:
        json.dump(all_trajectories, f)  # Write JSON directly to gzipped file


if __name__ == "__main__":
    ray.init()

    futures = []

    for i in range(REMOTES):
        futures.append(collect_n_trajectories.remote(i, 40051 + i))

    _ = [ray.get(f) for f in futures]
