import gymnasium as gym
import torch
from neuro_fuzzy import NeuroFuzzyController

# Load trained model
student = NeuroFuzzyController(6, 6, 4)
student.load_state_dict(torch.load("fuzzy_controller.pth"))

# Watch it land
env = gym.make("LunarLander-v3", render_mode="human")
obs, _ = env.reset()

while True:
    with torch.no_grad():
        q, _, _ = student(torch.FloatTensor(obs[:6]).unsqueeze(0))
        action = torch.argmax(q).item()
    
    obs, _, done, trunc, _ = env.step(action)
    
    if done or trunc:
        obs, _ = env.reset()
