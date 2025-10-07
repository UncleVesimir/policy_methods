import gymnasium as gym
import ale_py
# import gymnasium.envs.atari

def main():
    print("Hello from deep-q-learning!")
    print([k for k in gym.envs.registry.keys() if "Pong" in k])  # sanity check
    
    env = gym.make("ALE/Pong-v5", render_mode="human")
                   
    print(env.observation_space.shape)

if __name__ == "__main__":
    main()

# make the agent
"""
    - networks: eval, target
    - memory
"""

# instantiate env

# training loop


