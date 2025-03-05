# from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.algorithms.ppo.torch import ppo_torch_rl_module
from ray.rllib.core.models import configs as rlmodule_configs
from ray.rllib.core.rl_module.rl_module import RLModuleConfig


class FootsiesLSTMPPOModule(ppo_torch_rl_module.PPOTorchRLModule):
    def __init__(self, config: RLModuleConfig) -> None:
        super().__init__(config)

    def setup(self):

        catalog = self.config.get_catalog()

        catalog.actor_critic_encoder_config.base_encoder_config = (
            rlmodule_configs.RecurrentEncoderConfig()
        )

        if self.inference_only and self.framework == "torch":
            # catalog._model_config_dict["vf_share_layers"] = True
            # We need to set the shared flag in the encoder config
            # b/c the catalog has already been built at this point.
            catalog.actor_critic_encoder_config.shared = True

        # Build models from catalog
        self.encoder = catalog.build_actor_critic_encoder(
            framework=self.framework
        )
        self.pi = catalog.build_pi_head(framework=self.framework)
        # Only build the critic network when this is a learner module.
        if not self.inference_only or self.framework != "torch":
            self.vf = catalog.build_vf_head(framework=self.framework)
            # Holds the parameter names to be removed or renamed when synching
            # from the learner to the inference module.
            self._inference_only_state_dict_keys = {}

        self.action_dist_cls = catalog.get_action_dist_cls(
            framework=self.framework
        )


# class FootsiesModule(ppo_torch_rl_module.PPOTorchRLModule):
#     def __init__(self, config: RLModuleConfig) -> None:
#         super().__init__(config)

#     def setup(self):

#         input_dim = self.config.observation_space.shape[0]
#         hidden_dims = self.config.model_config_dict["fcnet_hiddens"]
#         output_dim = self.config.action_space.n

#         self.lstm = nn.LSTM(input_dim, hidden_dims[0], batch_first=True)

#         self.hiddens = [nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ReLU()]
#         for i, hidden_dim in enumerate(hidden_dims[1:-1]):
#             self.hiddens.append(nn.Linear(hidden_dim, hidden_dims[i + 1]))
#             self.hiddens.append(nn.ReLU())

#         self.policy_out = nn.Linear(hidden_dims[-1], output_dim)
#         self.value_out = nn.Linear(hidden_dims[-1], 1)

#     def get_initial_state(self):
#         return (
#             torch.zeros(1, 1, self.config.model_config_dict["fcnet_hiddens"][0]),
#             torch.zeros(1, 1, self.config.model_config_dict["fcnet_hiddens"][0]),
#         )

#     def _forward_inference(self, batch: NestedDict) -> dict[str, Any]:
#         output = {}

#         # Encoder forward pass.
#         encoder_outs = self.encoder(batch)
#         if Columns.STATE_OUT in encoder_outs:
#             output[Columns.STATE_OUT] = encoder_outs[Columns.STATE_OUT]

#         # Pi head.
#         output[Columns.ACTION_DIST_INPUTS] = self.pi(encoder_outs[ENCODER_OUT][ACTOR])

#         return output

#     def _forward_exploration(self, batch: NestedDict, **kwargs) -> dict[str, Any]:
#         """PPO forward pass during exploration.

#         Besides the action distribution, this method also returns the parameters of
#         the policy distribution to be used for computing KL divergence between the old
#         policy and the new policy during training.
#         """
#         # TODO (sven): Make this the only behavior once PPO has been migrated
#         #  to new API stack (including EnvRunners).
#         if self.config.model_config_dict.get("uses_new_env_runners"):
#             return self._forward_inference(batch)

#         output = {}

#         # Shared encoder
#         encoder_outs = self.encoder(batch)
#         if Columns.STATE_OUT in encoder_outs:
#             output[Columns.STATE_OUT] = encoder_outs[Columns.STATE_OUT]

#         # Value head
#         if not self.inference_only:
#             # If not for inference/exploration only, we need to compute the
#             # value function.
#             vf_out = self.vf(encoder_outs[ENCODER_OUT][CRITIC])
#             output[Columns.VF_PREDS] = vf_out.squeeze(-1)

#         # Policy head
#         action_logits = self.pi(encoder_outs[ENCODER_OUT][ACTOR])
#         output[Columns.ACTION_DIST_INPUTS] = action_logits

#         return output

#     def _forward_train(self, batch: NestedDict) -> dict[str, Any]:
#         if self.inference_only:
#             raise RuntimeError(
#                 "Trying to train a module that is not a learner module. Set the "
#                 "flag `inference_only=False` when building the module."
#             )
#         output = {}

#         # Shared encoder.
#         encoder_outs = self.encoder(batch)
#         if Columns.STATE_OUT in encoder_outs:
#             output[Columns.STATE_OUT] = encoder_outs[Columns.STATE_OUT]

#         # Value head.
#         vf_out = self.vf(encoder_outs[ENCODER_OUT][CRITIC])
#         # Squeeze out last dim (value function node).
#         output[Columns.VF_PREDS] = vf_out.squeeze(-1)

#         # Policy head.
#         action_logits = self.pi(encoder_outs[ENCODER_OUT][ACTOR])
#         output[Columns.ACTION_DIST_INPUTS] = action_logits

#         return output

#     @override(PPORLModule)
#     def _compute_values(self, batch, device=None):
#         infos = batch.pop(Columns.INFOS, None)
#         batch = convert_to_torch_tensor(batch, device=device)
#         if infos is not None:
#             batch[Columns.INFOS] = infos

#         # Separate vf-encoder.
#         if hasattr(self.encoder, "critic_encoder"):
#             if self.is_stateful():
#                 # The recurrent encoders expect a `(state_in, h)`  key in the
#                 # input dict while the key returned is `(state_in, critic, h)`.
#                 batch[Columns.STATE_IN] = batch[Columns.STATE_IN][CRITIC]
#             encoder_outs = self.encoder.critic_encoder(batch)[ENCODER_OUT]
#         # Shared encoder.
#         else:
#             encoder_outs = self.encoder(batch)[ENCODER_OUT][CRITIC]
#         # Value head.
#         vf_out = self.vf(encoder_outs)
#         # Squeeze out last dimension (single node value head).
#         return vf_out.squeeze(-1)
