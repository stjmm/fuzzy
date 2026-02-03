#!/usr/bin/env python3
"""
Neuro-Fuzzy Controller for LunarLander-v3
Command-line interface for all operations
"""

import argparse
import sys
import os
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN

from neuro_fuzzy import NeuroFuzzyController, initialize_student, distill, get_sb3_q_values, check_model_health
from post_process import optimize_rules, print_rules, analyze_rules
from visualize import plot_membership_functions, plot_rule_importance, plot_q_value_heatmap, create_all_visualizations


def train(args):
    """Train a fuzzy controller from scratch"""
    print("\n" + "="*70)
    print("TRAINING FUZZY CONTROLLER")
    print("="*70 + "\n")
    
    # Load teacher
    if not os.path.exists(args.teacher + ".zip"):
        print(f"❌ Teacher model not found: {args.teacher}.zip")
        return
    
    teacher = DQN.load(args.teacher)
    print(f"✓ Loaded teacher from {args.teacher}.zip")
    
    # Collect data
    print(f"\nCollecting {args.init_samples} initialization samples...")
    env = gym.make(args.env)
    states, q_vals = [], []
    obs, _ = env.reset()
    
    for i in range(args.init_samples):
        q_np = get_sb3_q_values(teacher, obs, teacher.device)
        states.append(obs[:6])
        q_vals.append(q_np)
        
        action, _ = teacher.predict(obs, deterministic=True)
        obs, _, done, trunc, _ = env.step(action)
        if done or trunc:
            obs, _ = env.reset()
        
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{args.init_samples}")
    
    env.close()
    X, Y = np.array(states), np.array(q_vals)
    print(f"✓ Collected data: {X.shape}")
    
    # Initialize student
    print(f"\nInitializing {args.rules} rules...")
    student = NeuroFuzzyController(
        num_inputs=6,
        num_rules=args.rules,
        num_actions=4
    )
    initialize_student(student, X, Y, args.rules)
    check_model_health(student)
    
    # Distill
    print(f"\nDistilling for {args.episodes} episodes...")
    env = gym.make(args.env)
    student = distill(student, teacher, env, episodes=args.episodes, lr=args.lr)
    env.close()
    
    # Post-process
    print("\nPost-processing...")
    student = optimize_rules(student, threshold=args.merge_threshold)
    
    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(student.state_dict(), args.output)
    print(f"\n✓ Saved model to {args.output}")
    
    # Show rules
    if not args.no_show:
        print_rules(student)
        analyze_rules(student)
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE")
    print("="*70 + "\n")


def show_rules(args):
    """Display the learned fuzzy rules"""
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return
    
    student = NeuroFuzzyController(
        num_inputs=6,
        num_rules=args.rules,
        num_actions=4
    )
    student.load_state_dict(torch.load(args.model))
    
    print("\n" + "="*70)
    print("FUZZY CONTROLLER RULES")
    print("="*70 + "\n")
    
    print_rules(student)
    
    if args.analyze:
        analyze_rules(student)
    
    print()


def export_rules(args):
    """Export rules to human-readable text file"""
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return
    
    student = NeuroFuzzyController(
        num_inputs=6,
        num_rules=args.rules,
        num_actions=4
    )
    student.load_state_dict(torch.load(args.model))
    
    # Redirect print to file
    import io
    from contextlib import redirect_stdout
    
    output = io.StringIO()
    with redirect_stdout(output):
        print("="*70)
        print("FUZZY CONTROLLER RULES")
        print("Exported from:", args.model)
        print("="*70 + "\n")
        print_rules(student)
        analyze_rules(student)
    
    # Write to file
    with open(args.output, 'w') as f:
        f.write(output.getvalue())
    
    print(f"✓ Exported rules to {args.output}")
    
    # Also export as Python code
    if args.code:
        code_path = args.output.replace('.txt', '.py')
        export_as_code(student, code_path)
        print(f"✓ Exported as Python code to {code_path}")


def export_as_code(student, output_path):
    """Export fuzzy rules as executable Python code"""
    with torch.no_grad():
        mu = student.mu.squeeze(0).cpu().numpy()
        sigma = torch.exp(student.log_sigma).squeeze(0).cpu().numpy()
        weights = student.weights.squeeze(0).cpu().numpy()
        consequents = student.consequents.cpu().numpy()
    
    code = '''"""
Exported Fuzzy Controller Rules
Auto-generated from trained model
"""

import numpy as np

class FuzzyController:
    """Standalone fuzzy controller - no PyTorch needed!"""
    
    def __init__(self):
        # Rule parameters (mu, sigma, weights)
        self.mu = np.array(''' + repr(mu.tolist()) + ''')
        
        self.sigma = np.array(''' + repr(sigma.tolist()) + ''')
        
        self.weights = np.array(''' + repr(weights.tolist()) + ''')
        
        # Q-values for each rule
        self.consequents = np.array(''' + repr(consequents.tolist()) + ''')
    
    def predict(self, state):
        """
        Predict action from state
        
        Args:
            state: [p_x, p_y, v_x, v_y, angle, v_a, leg1, leg2]
                   (only first 6 dimensions used)
        
        Returns:
            action: 0=Nothing, 1=Left, 2=Main, 3=Right
        """
        # Use only first 6 dimensions
        x = np.array(state[:6])
        
        # Calculate membership for each rule
        num_rules = self.mu.shape[0]
        memberships = np.exp(-((x[np.newaxis, :] - self.mu) / self.sigma) ** 2)
        
        # Weighted T-norm (product)
        w_norm = self.weights / (np.max(self.weights, axis=1, keepdims=True) + 1e-8)
        w_norm = np.clip(w_norm, 0, 2)
        
        weighted_mem = np.power(np.clip(memberships, 1e-6, 1.0), w_norm)
        rule_firing = np.prod(weighted_mem, axis=1)
        
        # Normalize firing strengths
        firing_sum = np.sum(rule_firing)
        if firing_sum > 0:
            normalized_firing = rule_firing / firing_sum
        else:
            normalized_firing = np.ones(num_rules) / num_rules
        
        # Calculate Q-values
        q_values = np.dot(normalized_firing, self.consequents)
        
        # Return best action
        return np.argmax(q_values)
    
    def get_q_values(self, state):
        """Get Q-values for all actions"""
        x = np.array(state[:6])
        
        num_rules = self.mu.shape[0]
        memberships = np.exp(-((x[np.newaxis, :] - self.mu) / self.sigma) ** 2)
        
        w_norm = self.weights / (np.max(self.weights, axis=1, keepdims=True) + 1e-8)
        w_norm = np.clip(w_norm, 0, 2)
        
        weighted_mem = np.power(np.clip(memberships, 1e-6, 1.0), w_norm)
        rule_firing = np.prod(weighted_mem, axis=1)
        
        firing_sum = np.sum(rule_firing)
        if firing_sum > 0:
            normalized_firing = rule_firing / firing_sum
        else:
            normalized_firing = np.ones(num_rules) / num_rules
        
        q_values = np.dot(normalized_firing, self.consequents)
        
        return q_values


# Example usage
if __name__ == "__main__":
    import gymnasium as gym
    
    controller = FuzzyController()
    env = gym.make("LunarLander-v3", render_mode="human")
    
    for episode in range(5):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = controller.predict(obs)
            obs, reward, done, trunc, _ = env.step(action)
            episode_reward += reward
            if trunc:
                done = True
        
        print(f"Episode {episode+1}: {episode_reward:.1f}")
    
    env.close()
'''
    
    with open(output_path, 'w') as f:
        f.write(code)


def evaluate(args):
    """Evaluate a trained fuzzy controller"""
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return
    
    student = NeuroFuzzyController(
        num_inputs=6,
        num_rules=args.rules,
        num_actions=4
    )
    student.load_state_dict(torch.load(args.model))
    
    print(f"\nEvaluating for {args.episodes} episodes...")
    env = gym.make(args.env)
    rewards = []
    
    for ep in range(args.episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(obs[:6]).unsqueeze(0)
                q, _, _ = student(state_tensor)
                action = torch.argmax(q).item()
            
            obs, reward, done, trunc, _ = env.step(action)
            episode_reward += reward
            steps += 1
            
            if trunc:
                done = True
        
        rewards.append(episode_reward)
        if args.verbose:
            print(f"  Episode {ep+1:3d}: {episode_reward:7.2f} ({steps:3d} steps)")
    
    env.close()
    
    print(f"\n{'='*70}")
    print(f"Results ({args.episodes} episodes):")
    print(f"  Mean:   {np.mean(rewards):7.2f}")
    print(f"  Std:    {np.std(rewards):7.2f}")
    print(f"  Min:    {np.min(rewards):7.2f}")
    print(f"  Max:    {np.max(rewards):7.2f}")
    print(f"  Median: {np.median(rewards):7.2f}")
    print(f"{'='*70}\n")


def compare(args):
    """Compare teacher and student performance"""
    if not os.path.exists(args.teacher + ".zip"):
        print(f"❌ Teacher not found: {args.teacher}.zip")
        return
    
    if not os.path.exists(args.student):
        print(f"❌ Student not found: {args.student}")
        return
    
    teacher = DQN.load(args.teacher)
    student = NeuroFuzzyController(6, args.rules, 4)
    student.load_state_dict(torch.load(args.student))
    
    env = gym.make(args.env)
    
    print("\n" + "="*70)
    print("TEACHER vs STUDENT COMPARISON")
    print("="*70)
    
    # Teacher evaluation
    print(f"\n[Teacher DQN] {args.episodes} episodes:")
    teacher_rewards = []
    for ep in range(args.episodes):
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
        if args.verbose:
            print(f"  Episode {ep+1:3d}: {episode_reward:7.2f}")
    
    # Student evaluation
    print(f"\n[Student Fuzzy] {args.episodes} episodes:")
    student_rewards = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                q, _, _ = student(torch.FloatTensor(obs[:6]).unsqueeze(0))
                action = torch.argmax(q).item()
            obs, reward, done, trunc, _ = env.step(action)
            episode_reward += reward
            if trunc:
                done = True
        
        student_rewards.append(episode_reward)
        if args.verbose:
            print(f"  Episode {ep+1:3d}: {episode_reward:7.2f}")
    
    env.close()
    
    # Summary
    teacher_mean = np.mean(teacher_rewards)
    student_mean = np.mean(student_rewards)
    ratio = (student_mean / teacher_mean) * 100
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Teacher DQN:       {teacher_mean:7.2f} ± {np.std(teacher_rewards):6.2f}")
    print(f"Student Fuzzy:     {student_mean:7.2f} ± {np.std(student_rewards):6.2f}")
    print(f"Performance Ratio: {ratio:6.1f}%")
    print("="*70 + "\n")


def demo(args):
    """Run visual demonstration"""
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return
    
    student = NeuroFuzzyController(6, args.rules, 4)
    student.load_state_dict(torch.load(args.model))
    
    env = gym.make(args.env, render_mode="human")
    
    print("\nRunning visual demo...")
    print("Press Ctrl+C to stop\n")
    
    try:
        episode = 0
        while episode < args.episodes:
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            
            print(f"Episode {episode+1}...")
            
            while not done:
                with torch.no_grad():
                    q, _, _ = student(torch.FloatTensor(obs[:6]).unsqueeze(0))
                    action = torch.argmax(q).item()
                
                obs, reward, done, trunc, _ = env.step(action)
                episode_reward += reward
                
                if trunc:
                    done = True
            
            print(f"  Reward: {episode_reward:.2f}\n")
            episode += 1
    
    except KeyboardInterrupt:
        print("\nDemo stopped")
    
    finally:
        env.close()


def visualize(args):
    """Generate visualizations"""
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return
    
    student = NeuroFuzzyController(6, args.rules, 4)
    student.load_state_dict(torch.load(args.model))
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    if args.type == 'all' or args.type == 'membership':
        plot_membership_functions(
            student,
            save_path=os.path.join(args.output_dir, "membership_functions.png")
        )
    
    if args.type == 'all' or args.type == 'importance':
        plot_rule_importance(
            student,
            save_path=os.path.join(args.output_dir, "rule_importance.png")
        )
    
    if args.type == 'all' or args.type == 'qvalues':
        plot_q_value_heatmap(
            student,
            save_path=os.path.join(args.output_dir, "q_values.png")
        )
    
    print(f"\n✓ Saved visualizations to {args.output_dir}/\n")


def main():
    parser = argparse.ArgumentParser(
        description="Neuro-Fuzzy Controller for LunarLander-v3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new controller
  python run_fuzzy.py train --teacher models/best_model --output fuzzy.pth
  
  # Show learned rules
  python run_fuzzy.py show-rules --model fuzzy.pth
  
  # Export rules to text file and Python code
  python run_fuzzy.py export --model fuzzy.pth --output rules.txt --code
  
  # Evaluate performance
  python run_fuzzy.py evaluate --model fuzzy.pth --episodes 20
  
  # Compare with teacher
  python run_fuzzy.py compare --teacher models/best_model --student fuzzy.pth
  
  # Watch it fly
  python run_fuzzy.py demo --model fuzzy.pth
  
  # Generate plots
  python run_fuzzy.py visualize --model fuzzy.pth --output-dir ./plots
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new fuzzy controller')
    train_parser.add_argument('--teacher', default='models/best_model', help='Teacher DQN model path')
    train_parser.add_argument('--output', default='fuzzy_controller.pth', help='Output model path')
    train_parser.add_argument('--env', default='LunarLander-v3', help='Environment name')
    train_parser.add_argument('--rules', type=int, default=6, help='Number of fuzzy rules')
    train_parser.add_argument('--episodes', type=int, default=600, help='Training episodes')
    train_parser.add_argument('--init-samples', type=int, default=3000, help='Initialization samples')
    train_parser.add_argument('--lr', type=float, default=0.003, help='Learning rate')
    train_parser.add_argument('--merge-threshold', type=float, default=0.92, help='Fuzzy set merge threshold')
    train_parser.add_argument('--no-show', action='store_true', help='Don\'t show rules after training')
    
    # Show rules command
    show_parser = subparsers.add_parser('show-rules', help='Display learned fuzzy rules')
    show_parser.add_argument('--model', default='fuzzy_controller.pth', help='Model path')
    show_parser.add_argument('--rules', type=int, default=6, help='Number of rules in model')
    show_parser.add_argument('--analyze', action='store_true', help='Show detailed analysis')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export rules to text file')
    export_parser.add_argument('--model', default='fuzzy_controller.pth', help='Model path')
    export_parser.add_argument('--output', default='rules.txt', help='Output text file')
    export_parser.add_argument('--rules', type=int, default=6, help='Number of rules in model')
    export_parser.add_argument('--code', action='store_true', help='Also export as Python code')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate controller performance')
    eval_parser.add_argument('--model', default='fuzzy_controller.pth', help='Model path')
    eval_parser.add_argument('--env', default='LunarLander-v3', help='Environment name')
    eval_parser.add_argument('--rules', type=int, default=6, help='Number of rules in model')
    eval_parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
    eval_parser.add_argument('--verbose', action='store_true', help='Show each episode result')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare teacher and student')
    compare_parser.add_argument('--teacher', default='models/best_model', help='Teacher model path')
    compare_parser.add_argument('--student', default='fuzzy_controller.pth', help='Student model path')
    compare_parser.add_argument('--env', default='LunarLander-v3', help='Environment name')
    compare_parser.add_argument('--rules', type=int, default=6, help='Number of rules in student')
    compare_parser.add_argument('--episodes', type=int, default=10, help='Episodes per model')
    compare_parser.add_argument('--verbose', action='store_true', help='Show each episode')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Visual demonstration')
    demo_parser.add_argument('--model', default='fuzzy_controller.pth', help='Model path')
    demo_parser.add_argument('--env', default='LunarLander-v3', help='Environment name')
    demo_parser.add_argument('--rules', type=int, default=6, help='Number of rules in model')
    demo_parser.add_argument('--episodes', type=int, default=5, help='Number of episodes')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Generate visualizations')
    viz_parser.add_argument('--model', default='fuzzy_controller.pth', help='Model path')
    viz_parser.add_argument('--rules', type=int, default=6, help='Number of rules in model')
    viz_parser.add_argument('--output-dir', default='./outputs', help='Output directory')
    viz_parser.add_argument('--type', choices=['all', 'membership', 'importance', 'qvalues'],
                          default='all', help='Type of visualization')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Route to appropriate function
    commands = {
        'train': train,
        'show-rules': show_rules,
        'export': export_rules,
        'evaluate': evaluate,
        'compare': compare,
        'demo': demo,
        'visualize': visualize
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()
