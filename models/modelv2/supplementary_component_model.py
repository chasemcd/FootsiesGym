import numpy as np
from gymnasium import spaces

# from ray.rllib.models.torch import torch_modelv2
from ray.rllib.utils import framework
from ray.rllib.utils import typing as rllib_typing

from models.modelv2 import lstm_model

torch, nn = framework.try_import_torch()


class SupplementaryComponentModel(lstm_model.LSTMModel):
    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: rllib_typing.ModelConfigDict,
        name: str,
        supplementary_component_configs: dict[str, dict] | None = None,
        **kwargs,
    ):
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name, **kwargs
        )

        if supplementary_component_configs is None:
            supplementary_component_configs = {}

        self.supplementary_components = nn.ModuleDict()
        for (
            component_name,
            component_config,
        ) in supplementary_component_configs.items():
            self.supplementary_components[component_name] = (
                self._build_component(component_config)
            )

    def _build_component(self, component_config: dict) -> nn.Module:
        subcomponents = []
        input_size = component_config.get("input_size")
        output_size = component_config.get("output_size")
        hiddens = component_config.get("hiddens", [])
        act_fn = component_config.get("act_fn", "tanh")
        use_lstm = component_config.get("use_lstm", False)

        assert input_size is not None, "input_size must be provided"
        assert output_size is not None, "output_size must be provided"

        if use_lstm:
            assert hiddens, "hiddens must be provided if use_lstm is True"
            lstm_head = nn.LSTM(input_size, hiddens[0])
            subcomponents.append(lstm_head)
            input_size = hiddens[0]
        if not hiddens:
            layer = nn.Linear(input_size, output_size)
            self._init_layer(layer, is_final_layer=True)
            subcomponents.append(layer)
        else:
            prev_size = input_size
            for i, hidden_size in enumerate(hiddens):
                layer = nn.Linear(prev_size, hidden_size)
                self._init_layer(layer, is_final_layer=False)
                subcomponents.append(layer)
                subcomponents.append(self._get_act_fn(act_fn))
                prev_size = hidden_size

            layer = nn.Linear(hiddens[-1], output_size)
            self._init_layer(layer, is_final_layer=True)
            subcomponents.append(layer)

        return nn.Sequential(*subcomponents)

    def _init_layer(self, layer: nn.Linear, is_final_layer: bool = False):
        """Initialize layer weights and biases."""
        gain = 1.0 if is_final_layer else np.sqrt(2)
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0.0)

    @staticmethod
    def _get_act_fn(act_fn: str) -> nn.Module:
        if act_fn == "relu":
            return nn.ReLU()
        elif act_fn == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
