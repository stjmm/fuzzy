from stable_baselines3 import DQN
from neuro_fuzzy import NeuroFuzzyController
import torch

teacher = DQN.load("models/best_model")
student = NeuroFuzzyController(6, 6, 4)
student.load_state_dict(torch.load("fuzzy_controller.pth"))

env = gym.make("LunarLander-v3")

# Test Teacher
print("TEACHER:")
for i in range(5):
    obs, _ = env.reset()
    reward = 0
    done = False
    while not done:
        action, _ = teacher.predict(obs, deterministic=True)
        obs, r, done, trunc, _ = env.step(action)
        reward += r
        if trunc: done = True
    print(f"  Episode {i+1}: {reward:.1f}")

# Test Student
print("\nSTUDENT:")
for i in range(5):
    obs, _ = env.reset()
    reward = 0
    done = False
    while not done:
        with torch.no_grad():
            q, _, _ = student(torch.FloatTensor(obs[:6]).unsqueeze(0))
            action = torch.argmax(q).item()
        obs, r, done, trunc, _ = env.step(action)
        reward += r
        if trunc: done = True
    print(f"  Episode {i+1}: {reward:.1f}")
