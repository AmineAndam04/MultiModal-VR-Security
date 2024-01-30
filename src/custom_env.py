from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel, EnvironmentStats
from mlagents_envs.environment import ActionTuple
import numpy as np
from envutils import HyperParametersSideChannel
import gymnasium as gym
from gymnasium import spaces
import torch


class CustomUnityEnvironment(gym.Env):
    
    def __init__(self, unity_environment, behavior_name, max_episode_length,stats_side_channel, max_state=151, max_actions=124):
        super().__init__()
        self.unity_env = unity_environment
        self.behavior_name = behavior_name
        self.max_episode_length = max_episode_length
        self.current_episode_length = 0 
        self.stats_side_channel = stats_side_channel

        # Action space
        self.action_space = spaces.Box(low=np.array([-9999.0] * max_actions), high=np.array([9999.0] * max_actions), dtype=np.float32)

        # Visual observation space
        visual_observation_space = spaces.Box(low=0, high=1, shape=(256, 256, 3), dtype=np.float32)

        # Numerical observation space
        numerical_observation_space = spaces.Box(low=np.array([-9999.0] * max_state,dtype=np.float32), high=np.array([9999.0] * max_state,dtype=np.float32), dtype=np.float32)

        # A dictionary to represent the combined observation space
        self.observation_space = gym.spaces.Dict({
            "numerical_observation": numerical_observation_space,
            "visual_observation": visual_observation_space
        })
    def reset(self,seed = None, options = None):
        # Reset Unity environment
        self.unity_env.reset()
        self.current_episode_length = 0 
        # Get initial observations from Unity environment
        decision_steps, _ = self.unity_env.get_steps(self.behavior_name)
        stats = self.stats_side_channel.get_and_reset_stats()
        self.NbrObsc = int(stats['Nbr Obsc'][0][0])
        self.NbrDistr = int(stats['Nbr distr'][0][0])
        self.NbrMalavatar = int(stats['Nbr malavatar'][0][0])
        # Process and return the initial observations
        return self._process_observation(decision_steps), {}
    def _process_observation(self, decision_steps):
        
        numerical_observation = decision_steps[0].obs[1]
        numerical_observation = np.append(numerical_observation,[self.NbrObsc ,self.NbrDistr,self.NbrMalavatar])
        numerical_observation = numerical_observation.astype(np.float32)
        visual_observation = decision_steps[0].obs[0]
        return { "numerical_observation": numerical_observation,"visual_observation": visual_observation}
    
    def step(self, action):
        # Send the action to the Unity environment
        actions_to_send = torch.zeros((1,124), dtype=torch.float32)
        actions_to_send[:, :action.shape[0]] = torch.tensor(action).reshape(1,-1)
        #actions_to_send = torch.zeros(124, dtype=torch.float32)
        #actions_to_send[:action.shape[0]] = torch.tensor(action)
        action_tuple = ActionTuple()
        action_tuple.add_continuous(actions_to_send.cpu().detach().numpy())
        self.unity_env.set_actions(self.behavior_name, action_tuple)
        # Proceed to the next step in the Unity environment
        self.unity_env.step()
        self.current_episode_length += 1 
        decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)
        stats = self.stats_side_channel.get_and_reset_stats()
        self.NbrObsc = int(stats['Nbr Obsc'][0][0])
        self.NbrDistr = int(stats['Nbr distr'][0][0])
        self.NbrMalavatar = int(stats['Nbr malavatar'][0][0])
        reward = stats['Total Reward'][0][0]
        #Done
        if (self.current_episode_length < self.max_episode_length):
            terminated = False
        else:
            terminated = True
        # info
        info = dict()
        for i in range(1,7):
            info["Reward" + str(i)] = stats["Reward" + str(i)][0][0]
        info["done"] = terminated
        return self._process_observation(decision_steps), reward, terminated,False, info
    def close(self):
        return self.unity_env.close()


def loadCustomEnv(env_path, hyperparameters = [],max_episode_length = 60,seed = 42):
    hyperparameters = [len(hyperparameters)] + hyperparameters
    hyperparametersChannel = HyperParametersSideChannel()
    hyperparametersChannel.send_hyperparameters(hyperparameters)
    stats_side_channel = StatsSideChannel()
    side_channels = [hyperparametersChannel,stats_side_channel]
    # If you run two instances of training, you have to change the worker_id for the second run (worker_id = 0,1,2 ...)
    env = UnityEnvironment(file_name=env_path,side_channels=side_channels,seed = seed, no_graphics=True,worker_id = 8)
    env.reset()
    bhvr_name = behavior_name(env)
    custom_env = CustomUnityEnvironment(env,bhvr_name,max_episode_length = max_episode_length,stats_side_channel=stats_side_channel )
    return custom_env

def behavior_name(env):
    behavior_name = list(env.behavior_specs)[0]
    return behavior_name

    
