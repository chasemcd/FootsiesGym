import dataclasses
import logging
import os
from typing import TYPE_CHECKING

import natsort
import tree  # pip install dm_tree
from ray.rllib import policy as rllib_policy
from ray.rllib.core.rl_module import rl_module
from ray.rllib.examples._old_api_stack.policy import random_policy

# from models.rl_modules import noop
from ray.rllib.utils.framework import try_import_tf, try_import_torch

from footsies import footsies_env
from models.modelv2 import noop

tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class FootsiesModuleSpec:
    module_name: str
    experiment_name: str
    checkpoint_number: int = -1
    policy_id: str = "focal_policy"
    is_rlmodule: bool = False
    trial_id: str = None


class ModuleRepository:

    modules = [
        # Self-play policies
        FootsiesModuleSpec(
            module_name="4fs-16od-082992f-0.01to0.001-vsrandom",
            experiment_name="082992f-16od-noIB-0",
            checkpoint_number=67,
        ),
        FootsiesModuleSpec(
            module_name="4fs-16od-082992f-0.03to0.01-sp",
            experiment_name="ae15419-16od-sp-0",
            trial_id="ae15419-16od-sp-0-00000_0_2025-02-19_00-01-52",
            checkpoint_number=39,
        ),
        FootsiesModuleSpec(
            module_name="4fs-16od-13c7f7b-0.05to0.01-sp-00",
            experiment_name="13c7f7b-0.05to0.01-sp-1",
            trial_id="13c7f7b-0.05to0.01-sp-1-00000_0_2025-02-20_09-40-36",
            checkpoint_number=39,
        ),
        FootsiesModuleSpec(
            module_name="4fs-16od-13c7f7b-0.05to0.01-sp-01",
            experiment_name="13c7f7b-0.05to0.01-sp-1",
            trial_id="13c7f7b-0.05to0.01-sp-1-00001_1_2025-02-20_23-56-46",
            checkpoint_number=39,
        ),
        FootsiesModuleSpec(
            module_name="4fs-16od-13c7f7b-0.05to0.01-sp-02",
            experiment_name="13c7f7b-0.05to0.01-sp-1",
            trial_id="13c7f7b-0.05to0.01-sp-1-00002_2_2025-02-21_13-55-58",
            checkpoint_number=39,
        ),
        FootsiesModuleSpec(
            module_name="4fs-16od-13c7f7b-0.05to0.01-sp-03",
            experiment_name="13c7f7b-0.05to0.01-sp-1",
            trial_id="13c7f7b-0.05to0.01-sp-1-00003_3_2025-02-22_04-04-54",
            checkpoint_number=40,
        ),
        # ESR policies
        FootsiesModuleSpec(
            module_name="ESR-0",
            experiment_name="6e32536-ESR-vsSPRandom-0",
            checkpoint_number=-1,
        ),
        FootsiesModuleSpec(
            module_name="ESR-0.1alpha-0",
            experiment_name="6378da0-ESR-0.1alpha-0",
            checkpoint_number=46,
        ),
    ]

    static_modules = {
        "random": random_policy.RandomPolicy(
            observation_space=footsies_env.FootsiesEnv.observation_space["p1"],
            action_space=footsies_env.FootsiesEnv.action_space["p1"],
            config={},
        ),
        "noop": noop.NoOpPolicy(
            observation_space=footsies_env.FootsiesEnv.observation_space["p1"],
            action_space=footsies_env.FootsiesEnv.action_space["p1"],
            config={},
        ),
        # "random_rlmodule": rl_module.SingleAgentRLModuleSpec(
        #     module_class=random_rlm.RandomRLModule,
        #     observation_space=footsies_env.FootsiesEnv.observation_space,
        #     action_space=footsies_env.FootsiesEnv.action_space,
        #     model_config_dict={},
        # ),
        # "noop_rlmodule": rl_module.SingleAgentRLModuleSpec(
        #     module_class=noop.NoOpRLModule,
        #     observation_space=footsies_env.FootsiesEnv.observation_space,
        #     action_space=footsies_env.FootsiesEnv.action_space,
        #     model_config_dict={},
        # ),
    }

    @classmethod
    def get(
        cls, module_spec_name: str
    ) -> rllib_policy.Policy | rl_module.RLModule:
        """Retrieve the policy from the policy repository."""

        if module_spec_name in cls.static_modules:
            return (
                cls.static_modules[module_spec_name].build()
                if "rlmodule" in module_spec_name
                else cls.static_modules[module_spec_name]
            )

        for module in cls.modules:
            if module.module_name == module_spec_name:
                return get_local_checkpoint(module)

        raise ValueError(
            f"Module {module_spec_name} not found in the policy repository."
        )


def get_local_checkpoint(
    module_spec: FootsiesModuleSpec,
) -> rllib_policy.Policy:
    """Retrieve the checkpoint from the local filesystem. If checkpoint_number is -1, the latest checkpoint is retrieved."""
    base_dir = os.path.expanduser(
        f"~/ray_results/{module_spec.experiment_name}"
    )

    # For windows
    # Get absolute path and add file:// prefix
    # base_dir = (
    #     f"{os.path.abspath(os.path.expanduser(module_spec.experiment_name))}"
    # )

    trial_name = module_spec.trial_id

    if trial_name is None:
        num_dirs = 0
        for fname in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, fname)):
                num_dirs += 1
                trial_name = fname

        if num_dirs > 1:
            raise ValueError(
                f"More than one trial found in {base_dir}. Please specify the trial ID with FootsiesModuleSpec.trial_id."
            )

    if trial_name is None:
        raise FileNotFoundError(f"No trials found in {base_dir}")

    if module_spec.checkpoint_number == -1:
        checkpoints = natsort.natsorted(
            [
                ckpt
                for ckpt in os.listdir(os.path.join(base_dir, trial_name))
                if ckpt.startswith("checkpoint_")
            ]
        )
        checkpoint_dir = os.path.join(base_dir, trial_name, checkpoints[-1])
    else:
        checkpoint_dir = os.path.join(
            base_dir,
            trial_name,
            f"checkpoint_{module_spec.checkpoint_number:06d}",
        )
    assert os.path.exists(
        checkpoint_dir
    ), f"Checkpoint {checkpoint_dir} does not exist."

    if module_spec.is_rlmodule:
        module_dir = os.path.join(
            checkpoint_dir, "learner/module_state", module_spec.policy_id
        )

        module = rl_module.RLModule.from_checkpoint(module_dir)
    else:
        module = rllib_policy.Policy.from_checkpoint(
            checkpoint_dir, policy_ids=module_spec.policy_id
        )[module_spec.policy_id]

    return module
