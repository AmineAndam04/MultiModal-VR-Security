from stable_baselines3 import PPO
from custom_env import loadCustomEnv
import glob
import os
"""def train(policy, env_path, env_hyperparameters,max_episode_length,seed_env,learning_rate,clip_range,ent_coef,n_steps,batch_size,
          n_epochs, gamma,custom_reward_logging, tensorboard_log,device,verbose,tb_log_name,timesteps,seed_ppo,save_path):
    # load the env
    env = loadCustomEnv(env_path=env_path, hyperparameters =env_hyperparameters,max_episode_length = max_episode_length,seed=seed_env)
    model = PPO(policy=policy, env=env, verbose=verbose,n_steps=n_steps, batch_size=batch_size,n_epochs = n_epochs,gamma = gamma,
            learning_rate=learning_rate,clip_range = clip_range,ent_coef = ent_coef,tensorboard_log=tensorboard_log, device=device, seed= seed_ppo)
    TIMESTEPS = timesteps
    #model.learn(total_timesteps=TIMESTEPS,callback=CustomRewardLoggingCallback(),tb_log_name=tb_log_name)
    #for i in range(totalTimesteps):
    #    model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False, callback=CustomRewardLoggingCallback(),tb_log_name=tb_log_name)
    model.learn(total_timesteps=TIMESTEPS, callback=custom_reward_logging(),tb_log_name=tb_log_name)
    # Store the model
    model.save(save_path)

    # close the env
    env.close()"""

def train(policy, env_path, env_hyperparameters, max_episode_length, seed_env, learning_rate, clip_range, ent_coef, n_steps, batch_size,
          n_epochs,target_kl,max_grad_norm, callback, tensorboard_log, device, verbose, tb_log_name, timesteps, seed_ppo, save_path):
    # Load the environment
    env = loadCustomEnv(env_path=env_path, hyperparameters=env_hyperparameters, max_episode_length=max_episode_length, seed=seed_env)
    if target_kl == "None":
        target_kl = None
    # Find the latest saved model
    list_of_files = glob.glob(os.path.join(save_path, '*.zip')) # Assumes the saved models are in .zip format
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

    # Save the model
    model.save(save_path)

    # Close the environment
    env.close()

