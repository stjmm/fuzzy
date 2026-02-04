import gymnasium as gym
import torch
from neuro_fuzzy import NeuroFuzzyController

# Load model
model = NeuroFuzzyController(num_inputs=6, num_rules=6, num_actions=4)
model.load_state_dict(torch.load("fuzzy_controller.pth"))

# Setup
env = gym.make("LunarLander-v3", render_mode="human")
actions = ['Do Nothing', 'Fire Left', 'Fire Main', 'Fire Right']
obs, _ = env.reset()
done = False
total_reward = 0
step = 0

print("\nFUZZY CONTROLLER W/ RULES")

while not done:
    with torch.no_grad():
        state = torch.FloatTensor(obs[:6]).unsqueeze(0)
        q_values, firing, _ = model(state)
        action = torch.argmax(q_values).item()
        
        # Find most active rule
        max_rule = torch.argmax(firing).item()
        rule_strength = firing[0, max_rule].item()
    
    obs, reward, done, trunc, _ = env.step(action)
    total_reward += reward
    step += 1
    
    print(f"Step {step:3d} | Rule {max_rule+1} ({rule_strength:.2f}) | {actions[action]:13s} | Reward: {reward:6.2f}")
    
    if trunc:
        done = True

env.close()
