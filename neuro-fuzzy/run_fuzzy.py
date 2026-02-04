import sys
import os
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN

# Import necessary components
from neuro_fuzzy import NeuroFuzzyController
from post_process import print_rules, analyze_rules

# --- CONFIGURATION ---
TEACHER_PATH = "models/best_model"
STUDENT_PATH = "fuzzy_controller.pth"
ENV_NAME = "LunarLander-v3"
NUM_RULES = 6
EPISODES = 10
# ---------------------

def show_learned_rules(student):
    """Display the fuzzy rules and statistics"""
    print("\n" + "="*70)
    print("LEARNED FUZZY RULES")
    print("="*70 + "\n")
    
    print_rules(student)
    analyze_rules(student)

def run_comparison(teacher, student, env_name, n_episodes):
    """Run side-by-side comparison"""
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    
    env = gym.make(env_name)
    
    # Teacher Evaluation
    print(f"\nDQN (Teacher) - {n_episodes} episodes:")
    teacher_rewards = []
    
    for i in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = teacher.predict(obs, deterministic=True)
            obs, reward, done, trunc, _ = env.step(action)
            episode_reward += reward
            if trunc: done = True
            
        teacher_rewards.append(episode_reward)
        print(f"  Episode {i+1}: {episode_reward:.2f}")

    # Student Evaluation
    print(f"\nFuzzy (Student) - {n_episodes} episodes:")
    student_rewards = []
    
    for i in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(obs[:6]).unsqueeze(0)
                q_values, _, _ = student(state_tensor)
                action = torch.argmax(q_values).item()
            
            obs, reward, done, trunc, _ = env.step(action)
            episode_reward += reward
            if trunc: done = True
            
        student_rewards.append(episode_reward)
        print(f"  Episode {i+1}: {episode_reward:.2f}")

    env.close()

    # Summary
    t_mean = np.mean(teacher_rewards)
    s_mean = np.mean(student_rewards)
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Teacher Mean: {t_mean:.2f}")
    print(f"Student Mean: {s_mean:.2f}")
    print("="*70 + "\n")

def main():
    if not os.path.exists(TEACHER_PATH + ".zip"):
        print(f"Error: Teacher model not found at {TEACHER_PATH}.zip")
        return
    
    print(f"Loading Teacher: {TEACHER_PATH}")
    teacher = DQN.load(TEACHER_PATH)

    if not os.path.exists(STUDENT_PATH):
        print(f"Error: Student model not found at {STUDENT_PATH}")
        return

    print(f"Loading Student: {STUDENT_PATH}")
    student = NeuroFuzzyController(num_inputs=6, num_rules=NUM_RULES, num_actions=4)
    
    try:
        student.load_state_dict(torch.load(STUDENT_PATH))
        student.eval() # Set to evaluation mode
    except Exception as e:
        print(f"Error loading student weights: {e}")
        return

    show_learned_rules(student)

    try:
        run_comparison(teacher, student, ENV_NAME, EPISODES)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nRuntime Error: {e}")

if __name__ == "__main__":
    main()
