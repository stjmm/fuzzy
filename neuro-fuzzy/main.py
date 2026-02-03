import gymnasium as gym
import numpy as np
import torch
import os
import sys
from stable_baselines3 import DQN

# Import the FIXED version
from neuro_fuzzy import (
    NeuroFuzzyController, 
    initialize_student, 
    distill, 
    get_sb3_q_values,
    check_model_health
)
from post_process import optimize_rules, print_rules, analyze_rules
from visualize import (
    plot_membership_functions, 
    plot_rule_importance,
    plot_q_value_heatmap,
    create_all_visualizations
)


def get_teacher_data(teacher, env_name, n_samples=5000):
    """Collect state-Q pairs from teacher for initialization"""
    print(f"Collecting {n_samples} samples from teacher...")
    env = gym.make(env_name)
    states, q_vals = [], []
    
    obs, _ = env.reset()
    device = teacher.device
    
    collected = 0
    episodes = 0
    
    while collected < n_samples:
        # Get Teacher Q-Values
        q_np = get_sb3_q_values(teacher, obs, device)
            
        states.append(obs[:6])  # Exclude ground sensors
        q_vals.append(q_np)
        collected += 1
        
        # Step Environment
        action, _ = teacher.predict(obs, deterministic=True)
        obs, _, done, trunc, _ = env.step(action)
        
        if done or trunc:
            obs, _ = env.reset()
            episodes += 1
        
        if collected % 1000 == 0:
            print(f"  Collected: {collected}/{n_samples} samples from {episodes} episodes")
    
    env.close()
    print(f"✓ Collected {len(states)} samples from {episodes} episodes")
    
    return np.array(states), np.array(q_vals)


def evaluate_controller(controller, env_name, n_episodes=10, verbose=True):
    """Evaluate the fuzzy controller"""
    env = gym.make(env_name)
    rewards = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(obs[:6]).unsqueeze(0)
                q, _, _ = controller(state_tensor)
                
                if torch.isnan(q).any():
                    print(f"⚠️  NaN in Q-values during evaluation, using random action")
                    action = env.action_space.sample()
                else:
                    action = torch.argmax(q).item()
            
            obs, reward, done, trunc, _ = env.step(action)
            episode_reward += reward
            steps += 1
            
            if trunc:
                done = True
        
        rewards.append(episode_reward)
        
        if verbose:
            print(f"  Episode {ep+1}: Reward = {episode_reward:.2f}, Steps = {steps}")
    
    env.close()
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    print(f"\n{'='*70}")
    print(f"Evaluation Results ({n_episodes} episodes):")
    print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Min/Max: {np.min(rewards):.2f} / {np.max(rewards):.2f}")
    print(f"{'='*70}\n")
    
    return rewards


def compare_teacher_student(teacher, student, env_name, n_episodes=10):
    """Compare teacher DQN and student fuzzy controller"""
    print("\n" + "="*70)
    print("TEACHER vs STUDENT COMPARISON")
    print("="*70)
    
    # Evaluate teacher
    print("\n[Teacher DQN Evaluation]")
    env = gym.make(env_name)
    teacher_rewards = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = teacher.predict(obs, deterministic=True)
            obs, reward, done, trunc, _ = env.step(action)
            episode_reward += reward
            
            if trunc:
                done = True
        
        teacher_rewards.append(episode_reward)
        print(f"  Episode {ep+1}: {episode_reward:.2f}")
    
    env.close()
    
    # Evaluate student
    print("\n[Student Fuzzy Controller Evaluation]")
    student_rewards = evaluate_controller(student, env_name, n_episodes, verbose=True)
    
    # Summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"Teacher DQN:           {np.mean(teacher_rewards):.2f} ± {np.std(teacher_rewards):.2f}")
    print(f"Student Fuzzy:         {np.mean(student_rewards):.2f} ± {np.std(student_rewards):.2f}")
    
    ratio = (np.mean(student_rewards) / np.mean(teacher_rewards)) * 100
    print(f"Performance Ratio:     {ratio:.1f}%")
    print("="*70 + "\n")
    
    return teacher_rewards, student_rewards


def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("NEURO-FUZZY CONTROLLER FOR LUNARLANDER-V3")
    print("Policy Distillation from DQN Teacher")
    print("="*70 + "\n")
    
    # Configuration
    ENV_NAME = "LunarLander-v3"
    TEACHER_PATH = "models/best_model"  # Points to models/best_model.zip
    NUM_RULES = 6
    DISTILL_EPISODES = 600
    LEARNING_RATE = 0.003
    
    # Step 1: Load Teacher
    print("[Step 1] Loading Teacher DQN...")
    if os.path.exists(TEACHER_PATH + ".zip"):
        print(f"  Loading from {TEACHER_PATH}.zip")
        teacher = DQN.load(TEACHER_PATH)
        print(f"  ✓ Teacher loaded successfully")
    else:
        print(f"  ❌ Model not found at {TEACHER_PATH}.zip")
        print(f"  Please train a teacher model first using train_teacher.py")
        return
    
    # Step 2: Collect Initialization Data
    print(f"\n[Step 2] Collecting Initialization Data...")
    X, Y = get_teacher_data(teacher, ENV_NAME, n_samples=3000)
    print(f"  State shape: {X.shape}, Q-values shape: {Y.shape}")
    
    # Step 3: Initialize Student
    print(f"\n[Step 3] Initializing Student with {NUM_RULES} Rules...")
    student = NeuroFuzzyController(
        num_inputs=6, 
        num_rules=NUM_RULES, 
        num_actions=4
    )
    
    initialize_student(student, X, Y, num_rules=NUM_RULES)
    
    # Check initial health
    print("\n  Checking initial model health...")
    check_model_health(student)
    
    # Step 4: Distillation
    print(f"\n[Step 4] Distilling Policy...")
    print(f"  Episodes: {DISTILL_EPISODES}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    
    env = gym.make(ENV_NAME)
    student = distill(
        student, 
        teacher, 
        env, 
        episodes=DISTILL_EPISODES,
        lr=LEARNING_RATE
    )
    env.close()
    
    # Check health after training
    print("\n  Checking model health after training...")
    if not check_model_health(student):
        print("\n  ⚠️  WARNING: Model has health issues. Results may be poor.")
    
    # Step 5: Post-Processing
    print(f"\n[Step 5] Post-Processing...")
    student = optimize_rules(student, threshold=0.92, weight_threshold=0.01)
    
    # Step 6: Analysis
    print(f"\n[Step 6] Analyzing Learned Rules...")
    print_rules(student)
    analyze_rules(student)
    
    # Step 7: Evaluation
    print(f"\n[Step 7] Evaluating Controller...")
    student_rewards = evaluate_controller(student, ENV_NAME, n_episodes=20, verbose=False)
    
    # Step 8: Comparison
    print(f"\n[Step 8] Comparing with Teacher...")
    teacher_rewards, student_rewards = compare_teacher_student(
        teacher, student, ENV_NAME, n_episodes=10
    )
    
    # Step 9: Visualization
    print(f"\n[Step 9] Generating Visualizations...")
    create_all_visualizations(student, save_dir="./outputs")
    
    # Step 10: Save Model
    print(f"\n[Step 10] Saving Model...")
    torch.save(student.state_dict(), "fuzzy_controller.pth")
    print(f"  ✓ Saved to fuzzy_controller.pth")
    
    # Step 11: Demo (optional)
    print(f"\n[Step 11] Demo")
    demo = input("Run visual demo? (y/n): ").strip().lower()
    
    if demo == 'y':
        print("\nRunning visual demo (press Ctrl+C to stop)...")
        env = gym.make(ENV_NAME, render_mode="human")
        
        try:
            for episode in range(5):
                obs, _ = env.reset()
                episode_reward = 0
                done = False
                
                print(f"\n  Episode {episode + 1}...")
                
                while not done:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(obs[:6]).unsqueeze(0)
                        q, _, _ = student(state_tensor)
                        action = torch.argmax(q).item()
                    
                    obs, reward, done, trunc, _ = env.step(action)
                    episode_reward += reward
                    
                    if trunc:
                        done = True
                
                print(f"    Reward: {episode_reward:.2f}")
        
        except KeyboardInterrupt:
            print("\n  Demo interrupted")
        
        finally:
            env.close()
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - fuzzy_controller.pth (trained model)")
    print("  - outputs/membership_functions.png")
    print("  - outputs/rule_importance.png")
    print("  - outputs/q_values.png")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
