import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_membership_functions(student, save_path="fuzzy_sets.png", x_range=(-2, 2), n_points=200):
    """
    Visualize the learned fuzzy membership functions
    """
    state_names = ['p_x', 'p_y', 'v_x', 'v_y', 'angle', 'v_a']
    
    print(f"\nGenerating membership function plot...")
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    with torch.no_grad():
        mu = student.mu.squeeze(0).cpu().numpy()  # [rules, inputs]
        sigma = torch.exp(student.log_sigma).squeeze(0).cpu().numpy()
        weights = student.weights.squeeze(0).cpu().numpy()
    
    # Sample points
    x = np.linspace(x_range[0], x_range[1], n_points)
    
    # Colors for different rules
    colors = plt.cm.tab10(np.linspace(0, 1, student.num_rules))
    
    for input_idx in range(min(student.num_inputs, len(state_names))):
        ax = axes[input_idx]
        
        for rule_idx in range(student.num_rules):
            # Get parameters for this fuzzy set
            mu_val = mu[rule_idx, input_idx]
            sigma_val = sigma[rule_idx, input_idx]
            
            # Compute normalized weight for this input in this rule
            max_weight = np.max(weights[rule_idx])
            if max_weight > 0:
                norm_weight = weights[rule_idx, input_idx] / max_weight
            else:
                norm_weight = 0
            
            # Compute membership function
            membership = np.exp(-((x - mu_val) / sigma_val) ** 2)
            
            # Plot with alpha based on weight importance
            alpha = 0.3 + 0.7 * norm_weight  # Scale between 0.3 and 1.0
            linewidth = 1.5 if norm_weight > 0.5 else 1.0
            linestyle = '-' if norm_weight > 0.1 else '--'
            
            label = f'Rule {rule_idx+1} (w={norm_weight:.2f})'
            
            ax.plot(x, membership, 
                   color=colors[rule_idx], 
                   alpha=alpha,
                   linewidth=linewidth,
                   linestyle=linestyle,
                   label=label)
            
            # Mark the center
            if norm_weight > 0.1:
                ax.scatter([mu_val], [1.0], 
                          color=colors[rule_idx], 
                          s=50, 
                          alpha=alpha,
                          zorder=5)
        
        ax.set_xlabel(state_names[input_idx], fontsize=11)
        ax.set_ylabel('Membership', fontsize=11)
        ax.set_title(f'{state_names[input_idx]} Fuzzy Sets', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {save_path}")
    plt.close()


def plot_rule_importance(student, save_path="rule_importance.png"):
    """
    Visualize the importance of each input in each rule
    """
    print(f"\nGenerating rule importance heatmap...")
    
    state_names = ['p_x', 'p_y', 'v_x', 'v_y', 'angle', 'v_a']
    
    with torch.no_grad():
        weights = student.weights.squeeze(0).cpu().numpy()  # [rules, inputs]
        
        # Normalize weights per rule
        normalized_weights = np.zeros_like(weights)
        for rule_idx in range(student.num_rules):
            max_weight = np.max(weights[rule_idx])
            if max_weight > 0:
                normalized_weights[rule_idx] = weights[rule_idx] / max_weight
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(normalized_weights.T, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(student.num_rules))
    ax.set_yticks(np.arange(student.num_inputs))
    ax.set_xticklabels([f'Rule {i+1}' for i in range(student.num_rules)])
    ax.set_yticklabels(state_names[:student.num_inputs])
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Weight', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(student.num_rules):
        for j in range(student.num_inputs):
            text = ax.text(i, j, f'{normalized_weights[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title('Input Importance by Rule', fontsize=14, fontweight='bold')
    ax.set_xlabel('Fuzzy Rule', fontsize=12)
    ax.set_ylabel('Input Feature', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {save_path}")
    plt.close()


def plot_q_value_heatmap(student, save_path="q_values.png"):
    """
    Visualize the Q-values (consequents) for each rule
    """
    print(f"\nGenerating Q-value heatmap...")
    
    action_names = ['Do Nothing', 'Fire Left', 'Fire Main', 'Fire Right']
    
    with torch.no_grad():
        consequents = student.consequents.cpu().numpy()  # [rules, actions]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(consequents, cmap='RdBu_r', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(student.num_actions))
    ax.set_yticks(np.arange(student.num_rules))
    ax.set_xticklabels(action_names)
    ax.set_yticklabels([f'Rule {i+1}' for i in range(student.num_rules)])
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Q-Value', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(student.num_rules):
        for j in range(student.num_actions):
            # Highlight max Q-value
            if j == np.argmax(consequents[i]):
                weight = 'bold'
                color = 'white'
            else:
                weight = 'normal'
                color = 'black'
            
            text = ax.text(j, i, f'{consequents[i, j]:.2f}',
                          ha="center", va="center", 
                          color=color, fontsize=10,
                          weight=weight)
    
    ax.set_title('Q-Values by Rule', fontsize=14, fontweight='bold')
    ax.set_xlabel('Action', fontsize=12)
    ax.set_ylabel('Fuzzy Rule', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {save_path}")
    plt.close()


def plot_training_progress(losses, save_path="training_progress.png"):
    """
    Plot training loss over episodes
    """
    print(f"\nGenerating training progress plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    episodes = np.arange(len(losses))
    
    # Plot raw losses
    ax.plot(episodes, losses, alpha=0.3, linewidth=0.5, label='Episode Loss')
    
    # Plot smoothed losses
    window = min(50, len(losses) // 10)
    if window > 1:
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], smoothed, linewidth=2, 
               color='red', label=f'{window}-Episode Moving Average')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Progress', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {save_path}")
    plt.close()


def create_all_visualizations(student, save_dir="."):
    """
    Generate all visualizations
    """
    import os
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate plots
    plot_membership_functions(student, 
                             save_path=os.path.join(save_dir, "membership_functions.png"))
    
    plot_rule_importance(student, 
                        save_path=os.path.join(save_dir, "rule_importance.png"))
    
    plot_q_value_heatmap(student, 
                        save_path=os.path.join(save_dir, "q_values.png"))
    
    print("\n✓ All visualizations generated")
    print("="*70)


def plot_single_trajectory(student, env, save_path="trajectory.png", max_steps=500):
    """
    Plot a single trajectory showing state evolution and actions
    """
    print(f"\nGenerating trajectory visualization...")
    
    state_names = ['p_x', 'p_y', 'v_x', 'v_y', 'angle', 'v_a']
    action_names = ['Nothing', 'Left', 'Main', 'Right']
    
    # Collect trajectory
    obs, _ = env.reset()
    states = []
    actions = []
    rewards = []
    q_values = []
    
    done = False
    steps = 0
    
    with torch.no_grad():
        while not done and steps < max_steps:
            state_vector = obs[:6]
            states.append(state_vector)
            
            # Get Q-values
            q, _, _ = student(torch.FloatTensor(state_vector).unsqueeze(0))
            q_np = q.squeeze(0).cpu().numpy()
            q_values.append(q_np)
            
            action = np.argmax(q_np)
            actions.append(action)
            
            obs, reward, done, trunc, _ = env.step(action)
            rewards.append(reward)
            steps += 1
            
            if trunc:
                done = True
    
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    q_values = np.array(q_values)
    
    # Create visualization
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # Plot states
    for i in range(6):
        ax = fig.add_subplot(gs[i//2, i%2])
        ax.plot(states[:, i], linewidth=2)
        ax.set_ylabel(state_names[i], fontsize=11)
        ax.set_xlabel('Step', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{state_names[i]} Evolution', fontsize=11)
    
    # Plot actions
    ax = fig.add_subplot(gs[3, 0])
    colors = ['gray', 'blue', 'red', 'green']
    for i, action_name in enumerate(action_names):
        mask = actions == i
        ax.scatter(np.where(mask)[0], np.ones(mask.sum()) * i, 
                  c=colors[i], label=action_name, s=20, alpha=0.6)
    ax.set_yticks(range(4))
    ax.set_yticklabels(action_names)
    ax.set_xlabel('Step', fontsize=10)
    ax.set_title('Actions Taken', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot cumulative reward
    ax = fig.add_subplot(gs[3, 1])
    ax.plot(np.cumsum(rewards), linewidth=2, color='purple')
    ax.set_xlabel('Step', fontsize=10)
    ax.set_ylabel('Cumulative Reward', fontsize=11)
    ax.set_title(f'Total Reward: {np.sum(rewards):.1f}', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Single Trajectory ({steps} steps)', fontsize=14, fontweight='bold')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved trajectory plot to: {save_path}")
    plt.close()
    
    return states, actions, rewards
