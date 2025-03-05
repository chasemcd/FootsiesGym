from collections import namedtuple

import numpy as np
from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer

ESRBatch = namedtuple(
    "ESRBatch",
    [
        "state",
        "action",
        "other_agent_action",
        "future_state",
        "reward",
        "terminated",
        "truncated",
    ],
)


class ESRBuffer(ReplayBuffer):
    """
    Replay buffer for Empowerment via Successor Representation (ESR).
    This varies slightly from the original implementation in Myers et al. (2024),
    where we store _complete episodes_ and ensure that sampled future states are
    within the same episode and <=T (rather than wrapping around the buffer).

    """

    def __init__(self, capacity, gamma, obs_dim):
        super().__init__(capacity=capacity, storage_unit="timesteps")
        self.gamma = gamma
        self.capacity = capacity

        # Buffers for observations and actions of all agents.
        self.state_buffer = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action_buffer = np.zeros((capacity,), dtype=np.int32)
        self.other_agent_action_buffer = np.zeros((capacity,), dtype=np.int32)

        self.reward_buffer = np.zeros((capacity,), dtype=np.float32)
        self.done_buffer = np.zeros((capacity,), dtype=bool)

        self.pos = 0
        self.current_size = 0

    def add_batch(self, batch):
        for (
            state,
            action,
            other_agent_action,
            reward,
            terminated,
            truncated,
        ) in zip(
            batch[batch.OBS],
            batch[batch.ACTIONS],
            batch[batch.OTHER_AGENT_ACTIONS],
            batch[batch.REWARDS],
            batch[batch.TERMINATED],
            batch[batch.TRUNCATEDS],
        ):
            self.add(
                state,
                action,
                other_agent_action,
                reward,
                terminated or truncated,
            )

    def add(self, state, action, other_agent_action, reward, done):
        # Add observation, actions, and other data.
        self.state_buffer[self.pos] = state
        self.action_buffer[self.pos] = action
        self.other_agent_action_buffer[self.pos] = other_agent_action
        self.reward_buffer[self.pos] = reward
        self.done_buffer[self.pos] = done

        # Update position and size.
        self.pos = (self.pos + 1) % self.size
        self.current_size = min(self.current_size + 1, self.size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.current_size, size=batch_size)
        delta = np.random.geometric(1 - self.gamma, size=batch_size)
        future_idxs = (idxs + delta) % self.current_size

        # Gather sampled data.
        batch = {
            "state": self.state_buffer[idxs],
            "next_state": self.state_buffer[(idxs + 1) % self.current_size],
            "future_state": self.state_buffer[future_idxs],
            "action": self.action_buffer[idxs],
            "other_agent_action": self.other_agent_action_buffer[idxs],
            "reward": self.reward_buffer[idxs],
            "done": self.done_buffer[idxs],
        }
        return ESRBatch(**batch)
