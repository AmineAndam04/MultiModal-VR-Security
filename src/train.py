from stable_baselines3 import PPO
from custom_env import loadCustomEnv
import glob
import os


def train(policy, env_path, env_hyperparameters, max_episode_length, seed_env, learning_rate, clip_range, ent_coef, n_steps, batch_size,
          n_epochs,target_kl,max_grad_norm, callback, tensorboard_log, device, verbose, tb_log_name, timesteps, seed_ppo, save_path):
    
    env = loadCustomEnv(env_path=env_path, hyperparameters=env_hyperparameters, max_episode_length=max_episode_length, seed=seed_env)
    if target_kl == "None":
        target_kl = None
    # Find the latest saved model, to continue training (in HPC for example after time limit) 
    list_of_files = glob.glob(os.path.join(save_path, '*.zip')) 
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"Loading saved model from {latest_file}")
        model = PPO.load(latest_file,env=env)
        model.learn(total_timesteps=timesteps, callback=callback, tb_log_name=tb_log_name,reset_num_timesteps=False)
    else:
        print("Creating new model")
        model = PPO(policy=policy, env=env, verbose=verbose, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
                    learning_rate=learning_rate, clip_range=clip_range, ent_coef=ent_coef,target_kl=target_kl,max_grad_norm=max_grad_norm,tensorboard_log=tensorboard_log, device=device, seed=seed_ppo)
        model.learn(total_timesteps=timesteps, callback=callback, tb_log_name=tb_log_name)
    model.save(save_path)
    env.close()

