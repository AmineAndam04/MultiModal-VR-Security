from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
from custom_policy import CustomActorCriticPolicy,CustomActorCriticPolicySGD
from custom_network import SimpleCustomNetwork, DeepCustomNetwork, DeepCustomNetworkWithAdaptiveAvgPool1d,DeepCustomNetworkTanh,DeepCustomNetworkWithAttention,SimpleCustomNetworkTanh, EnhancedCustomNetwork,SimpleDeepFusionCustomNetwork,DeepCustomNetworkLV,DeepCustomNetworkRR,AblationNetwork
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float) Initial learning rate.
    :return: (Callable[[float], float]) schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        :param progress_remaining: (float) Remaining progress
        :return: (float) current learning rate
        """
        return progress_remaining * initial_value

    return func

def exponential_schedule(initial_value, decay_steps, decay_rate, min_lr):
    """
    Creates a function that returns the learning rate for the current step given the initial value,
    decay steps, decay rate, and minimum learning rate.
    """
    def func(current_step):
        """Calculates the learning rate."""
        current_lr = initial_value * (decay_rate ** (current_step / decay_steps))
        return max(current_lr, min_lr)
    
    return func


class CustomRewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomRewardLoggingCallback, self).__init__(verbose)
        self.reward_keys = ['Reward1', 'Reward2', 'Reward3', 'Reward4', 'Reward5', 'Reward6']
        self.reward_weights= self.weights()
        # Initialize the sums to zero
        self.reward_sums = {key: 0 for key in self.reward_keys}
        self.episode_rewards = {key: [] for key in self.reward_keys} # Store rewards per episode
        self.num_episodes = 0  # Counter for number of episodes
        self.num_steps = 0

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        # Check if episode is done and a reward is provided in 'info'
        #print(info["Reward6"])
        #print("weights is ",self.reward_weights["Reward6"] )
        if 'done' in info and info['done'] == False:
            self.num_steps += 1 
            for key in self.reward_keys:
                self.reward_sums[key] += info[key]*self.reward_weights[key]  
            
        if 'done' in info and info['done']:
            self.num_episodes += 1 
            self.num_steps += 1 
            for key in self.reward_keys:
                self.reward_sums[key] += info[key]*self.reward_weights[key] 
            for key in self.reward_keys:
                self.episode_rewards[key].append(self.reward_sums[key])
                self.reward_sums[key] = 0
        if self.num_steps % self.model.n_steps == 0:
                per_attacks = {'View-Blocking' : 0 ,"Distraction" :0,"User-Harassment" : 0}
                for key in ['Reward1', 'Reward2', 'Reward3']:
                    average_reward = np.mean(self.episode_rewards[key]) 
                    per_attacks["View-Blocking"]+= average_reward
                    self.episode_rewards[key] = []
                    self.reward_sums[key] = 0
                self.logger.record("rollout/View-Blocking", per_attacks['View-Blocking'])

                for key in ['Reward4', 'Reward5']:
                    average_reward = np.mean(self.episode_rewards[key]) 
                    per_attacks["Distraction"]+= average_reward
                    self.episode_rewards[key] = []
                    self.reward_sums[key] = 0
                self.logger.record("rollout/Distraction", per_attacks['Distraction'])

                per_attacks["User-Harassment"] = np.mean(self.episode_rewards['Reward6']) 
                self.episode_rewards['Reward6'] = []
                self.reward_sums['Reward6'] = 0
                self.logger.record("rollout/User-Harassment", per_attacks['User-Harassment'])
        return True
    def weights(self):
        return {'Reward1':0.007, 'Reward2':0.0003, 'Reward3':0.04761, 'Reward4':0.00384, 'Reward5':0.00384, 'Reward6':0.1052}

def CustomRewardLogging(Weights):
    class CustomRewardTensorBoard(CustomRewardLoggingCallback):
        def weights(self):
            
            return Weights
    return CustomRewardTensorBoard

def CustomTensorBoard_config(config):
    environemnt = config["environment"]
    Weights = dict()
    for i in range(1,7):
        Weights['Reward'+str(i)] = environemnt["alpha" + str(i)]
    
    customRewardLogging = CustomRewardLogging(Weights)
    return customRewardLogging
def Checkpoint_config(config):
    save = config["model"]
    checkpoint_callback = CheckpointCallback(
    save_freq=save["save_freq"],
    save_path=save["save_path"],
    name_prefix=save["name_prefix"],
    save_replay_buffer=True,
    save_vecnormalize=True)
    return checkpoint_callback

def env_config(config):
    environment = config["environment"]
    path = environment["path"]
    max_episode_length = environment["max_episode_length"]
    seed = environment["seed"]
    hyper_keys = list(environment.keys())[1:-2]
    hyperparameters = []
    for key in hyper_keys:
        hyperparameters.append(environment[key])
    return path,hyperparameters,max_episode_length,seed
def network_architectures(config):
     network_architectures = {
        'SimpleCustomNetwork': SimpleCustomNetwork, 
        'DeepCustomNetwork': DeepCustomNetwork,
        'DeepCustomNetworkWithAdaptiveAvgPool1d': DeepCustomNetworkWithAdaptiveAvgPool1d,
        'DeepCustomNetworkTanh' : DeepCustomNetworkTanh,
        'DeepCustomNetworkWithAttention': DeepCustomNetworkWithAttention,
        'SimpleCustomNetworkTanh':SimpleCustomNetworkTanh,
        "EnhancedCustomNetwork": EnhancedCustomNetwork,
        "SimpleDeepFusionCustomNetwork":SimpleDeepFusionCustomNetwork,
        "DeepCustomNetworkLV":DeepCustomNetworkLV,
        "DeepCustomNetworkRR":DeepCustomNetworkRR,
        "AblationNetwork":AblationNetwork
    }
     return network_architectures[config["network"]["architecture"]]

def custom_policy (network,device,optim):
    if optim == "Adam": 
        class CustomNetworkPolicy(CustomActorCriticPolicy):
            def _build_mlp_extractor(self):
                self.mlp_extractor = network().to(self.device)
    else : 
        class CustomNetworkPolicy(CustomActorCriticPolicySGD):
            def _build_mlp_extractor(self):
                self.mlp_extractor = network().to(self.device)

    return CustomNetworkPolicy

def policy_config(config):
    network = network_architectures(config)
    device = config["ppo"]["device"]
    optim = config["optimizer"]["optimizer"]
    policy = custom_policy(network,device,optim)
    return policy
def lr_config(config):
    schedules = {"linear": linear_schedule}
    schedular = schedules[config['optimizer']["schedular"]]
    lr = config["optimizer"]["learning_rate"]
    return schedular(lr)