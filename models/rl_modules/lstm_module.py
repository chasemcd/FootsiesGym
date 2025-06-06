from typing import Any, Dict, Optional

import numpy as np
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.utils import make_target_network
from ray.rllib.core.rl_module.apis.target_network_api import (
    TARGET_NETWORK_ACTION_DIST_INPUTS,
    TargetNetworkAPI,
)
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import NetworkType, TensorType

torch, nn = try_import_torch()


class LSTMModule(TorchRLModule, ValueFunctionAPI, TargetNetworkAPI):
    """An example TorchRLModule that contains an LSTM layer.

    .. testcode::

        import numpy as np
        import gymnasium as gym

        B = 10  # batch size
        T = 5  # seq len
        e = 25  # embedding dim
        CELL = 32  # LSTM cell size

        # Construct the RLModule.
        my_net = LSTMContainingRLModule(
            observation_space=gym.spaces.Box(-1.0, 1.0, (e,), np.float32),
            action_space=gym.spaces.Discrete(4),
            model_config={"lstm_cell_size": CELL}
        )

        # Create some dummy input.
        obs = torch.from_numpy(
            np.random.random_sample(size=(B, T, e)
        ).astype(np.float32))
        state_in = my_net.get_initial_state()
        # Repeat state_in across batch.
        state_in = tree.map_structure(
            lambda s: torch.from_numpy(s).unsqueeze(0).repeat(B, 1), state_in
        )
        input_dict = {
            Columns.OBS: obs,
            Columns.STATE_IN: state_in,
        }

        # Run through all 3 forward passes.
        print(my_net.forward_inference(input_dict))
        print(my_net.forward_exploration(input_dict))
        print(my_net.forward_train(input_dict))

        # Print out the number of parameters.
        num_all_params = sum(int(np.prod(p.size())) for p in my_net.parameters())
        print(f"num params = {num_all_params}")
    """

    @override(TorchRLModule)
    def setup(self):
        """Use this method to create all the model components that you require.

        Feel free to access the following useful properties in this class:
        - `self.model_config`: The config dict for this RLModule class,
        which should contain flxeible settings, for example: {"hiddens": [256, 256]}.
        - `self.observation|action_space`: The observation and action space that
        this RLModule is subject to. Note that the observation space might not be the
        exact space from your env, but that it might have already gone through
        preprocessing through a connector pipeline (for example, flattening,
        frame-stacking, mean/std-filtering, etc..).
        """
        # Assume a simple Box(1D) tensor as input shape.
        in_size = self.observation_space.shape[0]

        # Get the LSTM cell size from the `model_config` attribute:
        self._lstm_cell_size = self.model_config.get("lstm_cell_size", 256)
        self._lstm = nn.LSTM(in_size, self._lstm_cell_size, batch_first=True)
        in_size = self._lstm_cell_size

        # Build a sequential stack.
        layers = []
        # Get the dense layer pre-stack configuration from the same config dict.
        dense_layers = self.model_config.get("dense_layers", [128, 128])
        for out_size in dense_layers:
            # Dense layer.
            layers.append(nn.Linear(in_size, out_size))
            # ReLU activation.
            layers.append(nn.ReLU())
            in_size = out_size

        self._fc_net = nn.Sequential(*layers)

        # Logits layer (no bias, no activation).
        self._pi_head = nn.Linear(in_size, self.action_space.n)
        # Single-node value layer.
        self._values = nn.Linear(in_size, 1)

    @override(TorchRLModule)
    def get_initial_state(self) -> Any:
        return {
            "h": np.zeros(shape=(self._lstm_cell_size,), dtype=np.float32),
            "c": np.zeros(shape=(self._lstm_cell_size,), dtype=np.float32),
        }

    @override(TorchRLModule)
    def _forward_inference(self, batch, **kwargs):
        embeddings, state_outs = self._compute_embeddings_and_state_outs(batch)
        logits = self._pi_head(embeddings)
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.STATE_OUT: state_outs,
        }

    @override(TorchRLModule)
    def _forward_exploration(self, batch, **kwargs):
        return self._forward_inference(batch, **kwargs)

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        embeddings, state_outs = self._compute_embeddings_and_state_outs(batch)
        logits = self._pi_head(embeddings)
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.STATE_OUT: state_outs,
            Columns.EMBEDDINGS: embeddings,
        }

    # We implement this RLModule as a ValueFunctionAPI RLModule, so it can be used
    # by value-based methods like PPO or IMPALA.
    @override(ValueFunctionAPI)
    def compute_values(
        self, batch: Dict[str, Any], embeddings: Optional[Any] = None
    ) -> TensorType:
        if embeddings is None:
            embeddings, _ = self._compute_embeddings_and_state_outs(batch)
        values = self._values(embeddings).squeeze(-1)
        return values

    def _compute_embeddings_and_state_outs(
        self, batch, use_target_networks: bool = False
    ):
        obs = batch[Columns.OBS]
        state_in = batch[Columns.STATE_IN]
        h, c = state_in["h"], state_in["c"]
        # Unsqueeze the layer dim (we only have 1 LSTM layer).
        lstm = self._lstm if not use_target_networks else self._target_lstm
        embeddings, (h, c) = lstm(obs, (h.unsqueeze(0), c.unsqueeze(0)))
        # Push through our FC net.
        fc_net = (
            self._fc_net if not use_target_networks else self._target_fc_net
        )
        embeddings = fc_net(embeddings)
        # Squeeze the layer dim (we only have 1 LSTM layer).
        return embeddings, {"h": h.squeeze(0), "c": c.squeeze(0)}

    @override(TargetNetworkAPI)
    def make_target_networks(self) -> None:
        self._target_lstm = make_target_network(self._lstm)
        self._target_fc_net = make_target_network(self._fc_net)
        self._target_pi_head = make_target_network(self._pi_head)

    @override(TargetNetworkAPI)
    def get_target_network_pairs(
        self,
    ) -> list[tuple[NetworkType, NetworkType]]:
        return [
            (self._lstm, self._target_lstm),
            (self._fc_net, self._target_fc_net),
            (self._pi_head, self._target_pi_head),
        ]

    @override(TargetNetworkAPI)
    def forward_target(
        self, batch: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        embeddings, state_outs = self._compute_embeddings_and_state_outs(
            batch, use_target_networks=True
        )
        logits = self._target_pi_head(embeddings)
        return {
            TARGET_NETWORK_ACTION_DIST_INPUTS: logits,
        }
