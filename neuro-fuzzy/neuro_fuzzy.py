import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.mixture import GaussianMixture

# --- HELPER FOR SB3 DEVICE HANDLING ---
def get_sb3_q_values(model, obs, device):
    """
    Safely gets Q-values from a Stable-Baselines3 model, handling device placement.
    """
    with torch.no_grad():
        # Convert numpy obs to tensor and move to the model's device
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
            
        # Forward pass through the Q-Network
        q_values = model.policy.q_net(obs_tensor)
        
        # Move back to CPU for numpy usage
        return q_values.cpu().numpy().flatten()


class NeuroFuzzyController(nn.Module):
    def __init__(self, num_inputs, num_rules, num_actions):
        super(NeuroFuzzyController, self).__init__()
        self.num_inputs = num_inputs
        self.num_rules = num_rules
        self.num_actions = num_actions
        
        # Parameters - Initialize more carefully
        self.mu = nn.Parameter(torch.randn(1, num_rules, num_inputs) * 0.5)
        
        # Initialize sigma in a safer range (0.3 to 1.0)
        # Using log parameterization: sigma = exp(log_sigma)
        self.log_sigma = nn.Parameter(torch.ones(1, num_rules, num_inputs) * (-0.5))
        
        self.weights = nn.Parameter(torch.ones(1, num_rules, num_inputs))
        self.consequents = nn.Parameter(torch.randn(num_rules, num_actions) * 0.1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(1)  # [batch, 1, inputs]
        
        sigma = torch.exp(self.log_sigma).clamp(min=0.1, max=5.0)
        
        # Gaussian Activation
        numerator = (x - self.mu) ** 2
        denominator = sigma ** 2
        # Add epsilon to denominator for stability
        membership = torch.exp(-numerator / (denominator + 1e-8))
        
        membership = membership.clamp(min=1e-8, max=1.0)
        
        # Weighted T-Norm
        w_pos = F.relu(self.weights)
        w_max, _ = torch.max(w_pos, dim=2, keepdim=True)
        w_norm = w_pos / (w_max + 1e-8)
        
        w_norm_safe = w_norm.clamp(min=0.0, max=2.0)
        
        # Power operation with clamped base
        weighted_mem = torch.pow(membership.clamp(min=1e-6), w_norm_safe)
        
        # Product across input dimensions
        rule_firing = torch.prod(weighted_mem, dim=2)  # [batch, rules]
        
        firing_sum = torch.sum(rule_firing, dim=1, keepdim=True)
        normalized_firing = rule_firing / (firing_sum + 1e-6)
        
        if torch.isnan(normalized_firing).any():
            print("Warning: NaN detected in firing, using uniform distribution")
            normalized_firing = torch.ones_like(normalized_firing) / self.num_rules
        
        # Output Q-values
        output = torch.matmul(normalized_firing, self.consequents)
        
        return output, normalized_firing, w_norm


def initialize_student(student, inputs, outputs, num_rules):
    """Initialize fuzzy controller using GMM clustering"""
    print(f"Initializing {num_rules} Rules via GMM Clustering...")
    
    inputs_mean = inputs.mean(axis=0)
    inputs_std = inputs.std(axis=0) + 1e-8
    inputs_normalized = (inputs - inputs_mean) / inputs_std
    
    data = np.hstack([inputs_normalized, outputs])
    
    # Fit GMM with regularization
    gmm = GaussianMixture(
        n_components=num_rules, 
        covariance_type='diag', 
        random_state=42, 
        reg_covar=1e-4,  # Increased regularization
        max_iter=200,
        n_init=5  # Multiple initializations for stability
    )
    
    try:
        gmm.fit(data)
    except Exception as e:
        print(f" GMM fitting failed: {e}")
        print("Using random initialization instead...")
        return
    
    means = gmm.means_
    covariances = gmm.covariances_
    
    # Denormalize the means
    mu_init = means[:, :student.num_inputs] * inputs_std + inputs_mean
    q_init = means[:, student.num_inputs:]
    
    sigma_init = np.sqrt(covariances[:, :student.num_inputs]) * inputs_std
    sigma_init = np.clip(sigma_init, 0.2, 2.0)  # Reasonable range
    
    with torch.no_grad():
        student.mu.copy_(torch.from_numpy(mu_init).float().unsqueeze(0))
        
        # Use log parameterization for sigma
        student.log_sigma.copy_(torch.log(torch.from_numpy(sigma_init).float().unsqueeze(0)))
        
        # Initialize consequents with small values
        student.consequents.copy_(torch.from_numpy(q_init).float() * 0.1)
    
    print("Initialization complete")


def distill(student, teacher, env, episodes=1000, batch_size=64, lr=0.003):
    """Distill teacher policy into student fuzzy controller"""
    
    optimizer = optim.Adam(student.parameters(), lr=lr)
    replay_buffer = []
    
    # Hyperparameters
    TAU = 0.1  # Temperature for softmax
    LAMBDA_M = 0.1  # Reduced merge regularization
    LAMBDA_T = 0.05  # Reduced T-norm regularization
    EPSILON = 0.05  # Exploration rate
    
    # Detect Teacher Device (SB3 specific)
    teacher_device = teacher.device
    
    print(f"Starting Distillation for {episodes} episodes...")
    print(f"Hyperparameters: lr={lr}, tau={TAU}, lambda_m={LAMBDA_M}, lambda_t={LAMBDA_T}")
    
    episode_losses = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_loss = 0
        steps = 0
        
        while not done:
            state_vector = obs[:6]  # Ignore ground contact legs
            
            # Epsilon-greedy student action
            if np.random.rand() < EPSILON:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    t_obs = torch.FloatTensor(state_vector).unsqueeze(0)
                    q, _, _ = student(t_obs)
                    
                    if torch.isnan(q).any():
                        print(f" NaN in Q-values at episode {ep}, using random action")
                        action = env.action_space.sample()
                    else:
                        action = torch.argmax(q).item()
            
            next_obs, reward, done, trunc, _ = env.step(action)
            
            # Get teacher Q-values
            teacher_q_np = get_sb3_q_values(teacher, obs, teacher_device)
            
            # Store transition
            replay_buffer.append((state_vector, teacher_q_np))
            if len(replay_buffer) > 10000:
                replay_buffer.pop(0)
            
            obs = next_obs
            steps += 1
            
            # Training step
            if len(replay_buffer) >= batch_size:
                idx = np.random.choice(len(replay_buffer), batch_size, replace=False)
                batch = [replay_buffer[i] for i in idx]
                
                b_states = torch.FloatTensor(np.array([x[0] for x in batch]))
                b_targets = torch.FloatTensor(np.array([x[1] for x in batch]))
                
                optimizer.zero_grad()
                
                # Forward pass
                q_s, firing, w_norm = student(b_states)
                
                if torch.isnan(q_s).any() or torch.isnan(firing).any():
                    print(f"⚠️  NaN detected in forward pass at episode {ep}, skipping batch")
                    continue
                
                # KL Divergence Loss
                log_p_s = F.log_softmax(q_s / TAU, dim=1)
                p_t = F.softmax(b_targets / TAU, dim=1)
                
                l_kl = F.kl_div(log_p_s, p_t, reduction='batchmean')
                
                # Check for NaN in loss
                if torch.isnan(l_kl):
                    print(f"⚠️  NaN in KL loss at episode {ep}, skipping batch")
                    continue
                
                mu = student.mu.squeeze(0)  # [rules, inputs]
                sigma = torch.exp(student.log_sigma).squeeze(0)  # [rules, inputs]
                
                l_merge = 0.0
                if LAMBDA_M > 0:
                    for i in range(student.num_inputs):
                        # Pairwise distances between rule centers
                        mu_i = mu[:, i]  # [rules]
                        
                        # Compute pairwise differences
                        diff = mu_i.unsqueeze(0) - mu_i.unsqueeze(1)  # [rules, rules]
                        distances = diff ** 2
                        
                        # Penalize similar centers with different sigmas
                        sigma_i = sigma[:, i]
                        sigma_diff = (sigma_i.unsqueeze(0) - sigma_i.unsqueeze(1)) ** 2
                        
                        # Only penalize if centers are close (within 0.5 units)
                        mask = (distances < 0.25).float()
                        l_merge += torch.sum(mask * sigma_diff) / (student.num_rules ** 2)
                
                # T-norm regularization (encourage sparsity)
                l_tnorm = torch.mean(w_norm) if LAMBDA_T > 0 else 0.0
                
                loss = l_kl + LAMBDA_M * l_merge + LAMBDA_T * l_tnorm
                
                if torch.isnan(loss):
                    print(f"NaN in total loss at episode {ep}, skipping batch")
                    continue
                
                # Backward pass
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                
                has_nan_grad = False
                for name, param in student.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"NaN gradient in {name}")
                        has_nan_grad = True
                        break
                
                if not has_nan_grad:
                    optimizer.step()
                    episode_loss += loss.item()
                else:
                    optimizer.zero_grad()
            
            if done or trunc:
                break
        
        # Store episode loss
        avg_loss = episode_loss / max(steps, 1)
        episode_losses.append(avg_loss)
        
        # Logging
        if ep % 50 == 0:
            recent_loss = np.mean(episode_losses[-50:]) if len(episode_losses) >= 50 else avg_loss
            print(f"Episode {ep:4d} | Avg Loss: {recent_loss:.6f} | Steps: {steps}")
            
            # Additional diagnostics
            with torch.no_grad():
                sigma_vals = torch.exp(student.log_sigma).squeeze(0)
                print(f"  Sigma range: [{sigma_vals.min():.3f}, {sigma_vals.max():.3f}]")
                print(f"  Mu range: [{student.mu.min():.3f}, {student.mu.max():.3f}]")
                print(f"  Weight range: [{student.weights.min():.3f}, {student.weights.max():.3f}]")
    
    print("Distillation complete")
    return student


# Utility function for debugging
def check_model_health(student):
    """Check if model parameters are healthy (no NaN, reasonable ranges)"""
    issues = []
    
    with torch.no_grad():
        if torch.isnan(student.mu).any():
            issues.append("NaN in mu")
        if torch.isnan(student.log_sigma).any():
            issues.append("NaN in log_sigma")
        if torch.isnan(student.weights).any():
            issues.append("NaN in weights")
        if torch.isnan(student.consequents).any():
            issues.append("NaN in consequents")
        
        sigma = torch.exp(student.log_sigma)
        if (sigma < 0.01).any():
            issues.append("Sigma too small (< 0.01)")
        if (sigma > 10).any():
            issues.append("Sigma too large (> 10)")
    
    if issues:
        print(" Model health issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("Model parameters are healthy")
        return True
