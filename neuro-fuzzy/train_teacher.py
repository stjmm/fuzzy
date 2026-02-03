import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

def train_robust_teacher(resume_path=None):
    # 1. Setup paths
    log_dir = "./logs/"
    model_dir = "./models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 2. Create Env with Monitoring
    env = gym.make("LunarLander-v3")
    env = Monitor(env, log_dir)

    # 3. Callback for "Best Model"
    eval_callback = EvalCallback(
        env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=10_000,
        deterministic=True,
        render=False,
        n_eval_episodes=10, 
    )

    # 4. Hyperparameters (Aligned with your success metrics)
    # We pass AdamW and SmoothL1Loss explicitly here
    policy_kwargs = dict(
        net_arch=[256, 256],
        optimizer_class=optim.AdamW,
        optimizer_kwargs=dict(weight_decay=1e-5), # Slight regularization
    )

    if resume_path and os.path.exists(resume_path):
        print(f"--- Resuming Training from {resume_path} ---")
        # Load the model and attach the new environment
        model = DQN.load(resume_path, env=env)
        
        # Update hyperparameters if you want to change them during resume
        # (Note: Optimizer state is loaded from zip, so changing LR might need extra steps, 
        # but usually resuming continues as is)
        model.learning_rate = 1e-4
    else:
        print("--- Starting Fresh Robust Teacher Training ---")
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-4, 
            buffer_size=500_000,
            learning_starts=10_000,
            batch_size=128,
            tau=0.005,                   # Soft update factor (User Spec)
            target_update_interval=1,    # Update target every step (for Soft Updates)
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            max_grad_norm=100,           # Gradient Clipping (User Spec)
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir
        )

    # Hack to inject SmoothL1Loss (SB3 uses Huber by default which is similar, but let's be strict)
    # We replace the criterion after initialization
    model.policy.q_net.criterion = nn.SmoothL1Loss()

    total_steps = 500_000
    print(f"Targeting {total_steps} steps. Best model will be saved to {model_dir}/best_model.zip")

    try:
        model.learn(total_timesteps=total_steps, callback=eval_callback, reset_num_timesteps=(resume_path is None))
    except KeyboardInterrupt:
        print("\nTraining interrupted manually. Saving current model...")
        model.save(f"{model_dir}/interrupted_model")

    model.save(f"{model_dir}/final_model")
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to .zip file to resume training")
    args = parser.parse_args()
    
    train_robust_teacher(args.resume)
