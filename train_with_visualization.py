import gym
import pybullet_envs
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def train():
    env_id = "HumanoidBulletEnv-v0"
    env = gym.make(env_id)
    env.render(mode="human")

    # Initialize environment to ensure connection to PyBullet server
    obs = env.reset()

    # Set the time step and enable real-time simulation
    timeStep = 1.0 / 60.0  # Setting to 60 FPS
    p.setTimeStep(timeStep)
    p.setRealTimeSimulation(1)

    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1e6)
    model.save("ppo_humanoid")

    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        env.render(mode="human")
        if done:
            obs = env.reset()

if __name__ == "__main__":
    train()
