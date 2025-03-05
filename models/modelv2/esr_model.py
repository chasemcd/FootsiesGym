from gymnasium import spaces

# from ray.rllib.models.torch import torch_modelv2
from ray.rllib.utils import framework
from ray.rllib.utils import typing as rllib_typing

from models.modelv2 import supplementary_component_model

torch, nn = framework.try_import_torch()


class ESRModel(supplementary_component_model.SupplementaryComponentModel):
    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: rllib_typing.ModelConfigDict,
        name: str,
        **kwargs,
    ):

        # Get the representation dimension used as the output of phi, phi' and psi.
        repr_dim = kwargs.get("repr_dim", 32)
        hiddens = kwargs.get("hiddens", [64, 64])

        component_configs = {
            # Used for ϕ(s, a^R, a^H), ϕ'(s, a^R), where the latter
            # we just leave out the human action instead of providing a one hot.
            "phi": {
                "input_size": obs_space.shape[0] + 2 * action_space.n,
                "hiddens": hiddens,
                "output_size": repr_dim,
            },
            # Used for ψ(s+), where we just use the state as the input.
            "psi": {
                "input_size": obs_space.shape[0],
                "hiddens": hiddens,
                "output_size": repr_dim,
            },
        }

        super().__init__(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            supplementary_component_configs=component_configs,
            **kwargs,
        )
