import yaml
import argparse
from train import train
from utils import env_config, policy_config, lr_config,CustomTensorBoard_config,Checkpoint_config
from stable_baselines3.common.callbacks import CallbackList
def load_config(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main(config):
    env_path,env_hyperparameters,max_episode_length,seed_env = env_config(config=config)
    policy = policy_config(config=config)
    learning_rate = lr_config(config)
    custom_reward_logging = CustomTensorBoard_config(config=config)
    checkpoint_callback= Checkpoint_config(config=config)
    combined_callback = CallbackList([checkpoint_callback,custom_reward_logging()])

    train(policy, env_path,env_hyperparameters,max_episode_length,seed_env=seed_env,
        learning_rate = learning_rate,clip_range=config["ppo"]["clip_range"],ent_coef=config["ppo"]["ent_coef"],n_steps= config["ppo"]["n_steps"],batch_size =config["ppo"]["batch_size"] ,
        n_epochs=config["ppo"]["n_epochs"],target_kl =config["ppo"]["target_kl"],max_grad_norm = config["ppo"]["max_grad_norm"] ,callback=combined_callback, tensorboard_log=config["ppo"]["tensorboard_log"],device=config["ppo"]["device"],
        verbose = config["ppo"]["verbose"],tb_log_name= config["ppo"]["tb_log_name"], timesteps= config["ppo"]["timesteps"],
        seed_ppo= config["ppo"]["seed"], save_path = config["model"]["save_path"])
    print("Done training")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to configuration file', required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
