from gymnasium.envs.registration import register

register(
    id="ShepherdingSwarmsim-v0",
    entry_point="swarmsim.GymEnvs.shepherding_env.envs:ShepherdingEnv",
    max_episode_steps=2000,
)
