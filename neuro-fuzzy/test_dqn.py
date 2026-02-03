import gymnasium as gym
from stable_baselines3 import DQN

def watch_teacher():
    # Load the trained model
    model_path = "models/best_model"  # Ensure this matches your save file
    try:
        model = DQN.load(model_path)
        print(f"Loaded {model_path} successfully.")
    except FileNotFoundError:
        print("Error: Model file not found. Did you finish training?")
        return

    # Create environment with render_mode='human' to see the window
    env = gym.make("LunarLander-v3", render_mode="human")
    
    for episode in range(5):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # deterministic=True is safer for evaluation (uses the best action, no randomness)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, _ = env.step(action)
            total_reward += reward
            
            if done or trunc:
                print(f"Episode {episode + 1}: Score = {total_reward:.2f}")
                break
                
    env.close()

if __name__ == "__main__":
    watch_teacher()
