import functools

import ray
from absl import app, flags
from policies import esr
from ray import tune
from ray.rllib.algorithms import appo
from ray.rllib.algorithms import callbacks as rllib_callbacks
from ray.rllib.examples._old_api_stack.policy import random_policy
from ray.rllib.policy import policy as rllib_policy

from callbacks import add_policies, script_metrics, winrates
from experiments import experiment
from footsies import footsies_env
from models.modelv2 import esr_model, lstm_model
from utils import matchmaking

FLAGS = flags.FLAGS
flags.DEFINE_string("experiment_name", None, "Name of the experiment")
flags.DEFINE_boolean("debug", False, "Debug mode flag")
flags.DEFINE_boolean("tune", False, "Tune mode flag")


class ESRExperiment(experiment.Experiment):

    def construct_model_config(self, as_dict=True):

        policy_observation_space = footsies_env.FootsiesEnv.observation_space[
            "p1"
        ]
        policy_action_space = footsies_env.FootsiesEnv.action_space["p1"]

        eval_policies = [
            "4fs-16od-13c7f7b-0.05to0.01-sp-02",
            "4fs-16od-082992f-0.03to0.01-sp",
        ]
        train_policies = [
            # "4fs-16od-082992f-0.01to0.001-vsrandom",
            # "4fs-16od-082992f-0.03to0.01-sp",
            # "4fs-16od-13c7f7b-0.05to0.01-sp-00",
            "4fs-16od-13c7f7b-0.05to0.01-sp-01",
            "4fs-16od-13c7f7b-0.05to0.01-sp-02",
            # "4fs-16od-13c7f7b-0.05to0.01-sp-03",
            # "random",
        ]

        config = (
            appo.APPOConfig()
            .environment(
                "FootsiesEnv",
                env_config={
                    "max_t": 4000,
                    "frame_skip": 4,
                    "observation_delay": 16,
                    # "port": 80051,
                },
            )
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .resources(
                num_learner_workers=1,
                num_gpus_per_learner_worker=(
                    1 if not self.config.get("debug", False) else 0
                ),
                num_cpus_for_local_worker=1,
            )
            .env_runners(
                num_env_runners=(
                    40 if not self.config.get("debug", False) else 1
                ),
                num_envs_per_env_runner=1,
            )
            .env_runners(
                rollout_fragment_length=256,
                batch_mode="truncate_episodes",
            )
            .multi_agent(
                policies={
                    "focal_policy": rllib_policy.PolicySpec(
                        config={
                            "model": {
                                "custom_model": esr_model.ESRModel,
                                "custom_model_config": {
                                    "lstm_cell_size": 128,
                                    "policy_dense_widths": [128, 128],
                                },
                            },
                            "max_seq_len": 64,
                        },
                        observation_space=policy_observation_space,
                        action_space=policy_action_space,
                    ),
                    "training_opponent": rllib_policy.PolicySpec(
                        policy_class=appo.APPOTorchPolicy,
                        config={
                            "model": {
                                "custom_model": lstm_model.LSTMModel,
                                "custom_model_config": {
                                    "lstm_cell_size": 128,
                                    "policy_dense_widths": [128, 128],
                                },
                            },
                            "max_seq_len": 64,
                        },
                        observation_space=policy_observation_space,
                        action_space=policy_action_space,
                    ),
                    "random": rllib_policy.PolicySpec(
                        policy_class=random_policy.RandomPolicy,
                        observation_space=policy_observation_space,
                        action_space=policy_action_space,
                    ),
                },
                policy_mapping_fn=matchmaking.Matchmaker(
                    [
                        matchmaking.Matchup(
                            "focal_policy",
                            # train_policy,
                            # 1 / len(train_policies),
                            "training_opponent",
                            0.8,
                        ),
                        matchmaking.Matchup(
                            "focal_policy",
                            "4fs-16od-13c7f7b-0.05to0.01-sp-02",
                            0.2,
                        ),
                        # for train_policy in train_policies
                    ]
                ).policy_mapping_fn,
                policies_to_train=[
                    "focal_policy",
                    "training_opponent",
                ],
            )
            .evaluation(
                evaluation_num_env_runners=(
                    5 if not self.config.get("debug", False) else 1
                ),
                evaluation_interval=1,
                evaluation_duration=30,
                evaluation_duration_unit=(
                    "episodes"
                    if not self.config.get("debug", False)
                    else "timesteps"
                ),
                evaluation_parallel_to_training=True,
                evaluation_config={
                    "env_config": {
                        "evaluation": True,
                    },  # "port": 80052},
                    "multiagent": {
                        "policy_mapping_fn": matchmaking.Matchmaker(
                            [
                                matchmaking.Matchup(
                                    "focal_policy",
                                    eval_policy,
                                    1 / (len(eval_policies) + 1),
                                )
                                for eval_policy in eval_policies + ["random"]
                            ]
                        ).policy_mapping_fn,
                    },
                },
            )
            .callbacks(
                rllib_callbacks.make_multi_callbacks(
                    [
                        winrates.Winrates,
                        functools.partial(
                            add_policies.AddPolicies,
                            policies=set(eval_policies + train_policies),
                        ),
                        script_metrics.ScriptMetrics,
                    ]
                    # [matchup_performance.MatchupPerformance, winrates.Winrates]
                )
            )
        )

        config.training(
            train_batch_size=1024,
            # lr_schedule=[[0, 0.001], [5_000_000, 0.00075], [10_000_000, 3e-4]],
            lr=4e-4,
            # entropy_coeff=0.005,
            entropy_coeff_schedule=[[0, 0.03], [200_000_000, 0.01]],
            gamma=0.995,
            vf_loss_coeff=1.0,
            tau=4e-4,
        )

        return config

    def run(self):
        ray.init(
            local_mode=self.config.get("debug", False),
        )

        ray.tune.registry.register_env(
            "FootsiesEnv",
            env_creator=self.env_creator,
        )

        model_config = self.construct_model_config()
        tune_config = self.construct_tune_config()
        run_config = self.construct_run_config()

        tuner = tune.Tuner(
            trainable=esr.ESR,
            param_space=model_config,
            tune_config=tune_config,
            run_config=run_config,
        )

        tuner.fit()


def main(*args, **kwargs):
    ESRExperiment(
        config={
            "debug": FLAGS.debug,
            "experiment_name": FLAGS.experiment_name,
            "tune": FLAGS.tune,
        }
    ).run()


if __name__ == "__main__":
    app.run(main)
