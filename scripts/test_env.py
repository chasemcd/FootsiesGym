import time

from footsies import footsies_env

env = footsies_env.FootsiesEnv(
    config={"max_t": 4000, "frame_skip": 4, "observation_delay": 0}
)

obs, _ = env.reset()
while True:
    actions = {
        agent: env.action_spaces[agent].sample() for agent in env.agents
    }
    observations, rewards, terminateds, truncateds, _ = env.step(actions)

    if truncateds["__all__"] or terminateds["__all__"]:
        obs, _ = env.reset()

    time.sleep(1 / 60)
