from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from gymnasium import spaces
import torch.nn.init as init
import json
from custom_network import SimpleCustomNetwork

class CustomActorCriticPolicy(ActorCriticPolicy):
    """
     Network policy.
     See: https://github.com/DLR-RM/stable-baselines3/blob/d671402c9373391f44d8a2ad11deed615e0f4bae/stable_baselines3/common/policies.py#L414

    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule = None,
        optimizer_class  = torch.optim.Adam, #torch.optim.SGD
        optimizer_kwargs = None,
        *args,
        **kwargs,
        ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs["eps"] = 1e-4
        super(CustomActorCriticPolicy,self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            *args,
        **kwargs,
        )
        self._build(lr_schedule)
        self.iter = 0
        
    

    def _build_mlp_extractor(self) :
        """
         Create the policy and value networks.
        """
        self.mlp_extractor = SimpleCustomNetwork().to(self.device)
    def _build(self,lr_schedule: Schedule= None):
        """
        Create the networks, the optimizer, and the costraints.
        """
        self._build_mlp_extractor()
        self.mlp_extractor.initialize_parameters()
        self.optimizer = self.optimizer_class(self.parameters(),lr =lr_schedule(1), **self.optimizer_kwargs)
        xPos = (-10.2,31)
        yPos = (1,12)
        yPosv = (2.8,4.54)
        zPos = (17.4,45)
        tranp = (0,1)
        fc = (0,10) #20
        fb = (0,10)
        xbb = (0,3)
        ybb = (0,3)
        zbb = (0,3)
        Pos = [xPos,yPos,zPos]
        Posv = [xPos,yPosv,zPos]
        Bb = [xbb,ybb,zbb]
        self.max_obsc = [tup[1] for tup in Pos] + [tup[1] for tup in Bb] +[tranp[1]]
        self.min_obsc = [tup[0] for tup in Pos] + [tup[0] for tup in Bb] +[tranp[0]]
        self.max_distr = [tup[1] for tup in Bb] +[fc[1]]+[fb[1]]
        self.min_distr = [tup[0] for tup in Bb] +[fc[0]]+[fb[0]]
        self.max_malavatar = [tup[1] for tup in Posv]
        self.min_malavatar = [tup[0] for tup in Posv]
    
    def forward(self,obs,deterministic: bool = False):
        """
        Forward pass.
        """
        features = self.extract_features(obs)
        mean, log_std,values = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(mean=mean, log_std=log_std)
        actions = distribution.sample()
        masked_actions = self._mask(features[1], actions)
        masked_actions = torch.tensor(masked_actions).to(actions.device)
        actions = actions * masked_actions
        log_prob = distribution.log_prob(actions)
        actions_to_send = 9999 * torch.ones((1,124), dtype=torch.float32)
        actions_to_send[:, :actions.shape[1]] = actions.reshape(1,-1)
        return actions_to_send, values, log_prob
    
    
    def _mask(self,num_obs, actions):
        """
        Mask function
        """
        max_obsc = self.max_obsc * self.NbrObsc
        min_obsc = self.min_obsc * self.NbrObsc
        max_distr = self.max_distr * self.NbrDistr
        min_distr = self.min_distr * self.NbrDistr
        max_malavatar = self.max_malavatar * self.NbrMalavatar
        min_malavatar = self.min_malavatar * self.NbrMalavatar
        st = self._modifiable_state(num_obs.reshape(1, -1))
        st_1 =st + actions.cpu().numpy()
        max = max_obsc + max_distr + max_malavatar
        min = min_obsc + min_distr + min_malavatar
        mask = np.logical_and(min <= st_1, st_1 <= max).astype(int)
        return mask

    def _modifiable_state(self,num_obs):
        """
        Get mutable states.
        """
        nbrObscActions = self.NbrObsc * 7
        nbrDistr = self.NbrDistr * 5
        nbrMalavatar =self.NbrMalavatar * 3
        if (num_obs.shape[0] > 1):
            print("Inside the _mask")
            print(num_obs)
            print(num_obs.shape)
        obsc = num_obs[:,12: 12 + nbrObscActions]
        distr = num_obs[:,12 + nbrObscActions:12 + nbrObscActions + nbrDistr]
        malavatar = num_obs[:,12 + nbrObscActions + nbrDistr+12 :12 + nbrObscActions + nbrDistr+12 +3]
        return np.concatenate((obsc.cpu(),distr.cpu(),malavatar.cpu()),axis=1)
    
    def extract_features(self,obs):
        """
        Prepare the state space for the forward pass
        """
        vis = obs["visual_observation"]
        num = obs["numerical_observation"]
        num = self._remove_pads(num)
        #Process visual observation
        if isinstance(vis, torch.Tensor) == False:
            vis = torch.tensor(vis).clone().detach().requires_grad_(True)  # Convert to PyTorch tensor

        vis = vis.permute(0, 3, 1, 2)  # Change the order of dimensions
        #vis = vis.unsqueeze(0)
        #Process numerical observaton
        if isinstance(num, torch.Tensor) == False:
            num = torch.tensor(num).clone().detach().requires_grad_(True)  # Convert to PyTorch tensor
        num = num.unsqueeze(0)
        #num = num.unsqueeze(0)
        return vis, num
    def _remove_pads(self,array, pad =9999):
        """
        Remove pads from observations and assign the number of each object.
        Unity requires fixed length states and actions.
        """
        
        nbrs = array[:,-3:]
        try:
            self.NbrObsc ,self.NbrDistr,self.NbrMalavatar = int(nbrs[:,0]),int(nbrs[:,1]),int(nbrs[:,2])
            
        except : 
            self.NbrObsc ,self.NbrDistr,self.NbrMalavatar = nbrs[:,0].to(torch.int),int(nbrs[:,1]).to(torch.int),int(nbrs[:,2]).to(torch.int)
            print(self.NbrObsc)
            print(self.NbrDistr)
            print(self.NbrMalavatar)
        array = array[:,:-3]
        #end_index = np.argmax(array[] == pad)
        end_index = int(np.argmax(array.cpu() == pad, axis=1))
        if end_index == 0:
            return array
        else: 
            return array[:,:end_index]
    def _remove_pads_actions(self,array,pad = 9999):
        """
        Remove pads from actions
        """
        end_index = int(np.argmax(array.cpu() == pad, axis=1))
        if end_index == 0:
            return array
        else: 
            return array[:,:end_index]
    def get_distribution(self, obs):
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = self.extract_features(obs)
        mean, log_std,values = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(mean=mean, log_std=log_std)
        return distribution
    def _predict(self, obs, deterministic = False) :
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        features = self.extract_features(obs)
        distribution = self.get_distribution(obs)
        actions = distribution.sample()
        masked_actions = self._mask(features[1], actions)
        actions = actions * masked_actions
        actions_to_send = torch.ones((1,124), dtype=torch.float32)
        actions_to_send[:, :actions.shape[1]] = actions.reshape(1,-1)
        return actions_to_send
    def predict_values(self, obs):
        """
        Value function v(s)
        """
        features = self.extract_features(obs)
        _, _,values = self.mlp_extractor(features)
        return values
    def evaluate_actions(self, obs, actions) :
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        self.iter += 1
        #print(list(self.mlp_extractor.parameters()))
        """for name, param in self.mlp_extractor.named_parameters():
            if param.grad is not None:
                print(f"Parameter {name}:")
                print(f"Gradients: {param.grad}")"""
        #torch.save(self.mlp_extractor.state_dict(), 'model_weights' + str(self.iter) + '.pth')
        visuals = obs["visual_observation"]
        nums = obs["numerical_observation"]
        filename = "data.json"
        if self.iter < 2 : 
            with open(filename, "w") as json_file:
                json.dump({"num": nums.tolist()}, json_file)
        values = []
        log_probs = []
        entropys=[]
        for i in range(nums.shape[0]):
                vis, num = visuals[i].unsqueeze(0),nums[i].unsqueeze(0)
                ob = { "numerical_observation": num,"visual_observation": vis}
                action = actions[i].unsqueeze(0)
                action = self._remove_pads_actions(action)
                features = self.extract_features(ob)
                mean, log_std,value = self.mlp_extractor(features)
                try: 
                    distribution = self._get_action_dist_from_latent(mean=mean, log_std=log_std)
                except:
                    torch.save(self.mlp_extractor.state_dict(), 'failed_model_weights.pth')
                    print(features)
                    print(self.NbrObsc)
                    print(self.NbrDistr)
                    print(self.NbrMalavatar)
                    print(action.shape)
                    print(mean.shape)
                    print(action)
                    print(mean)
                    print(log_std)
                log_prob = distribution.log_prob(action)
                entropy = distribution.entropy()
                values.append(value)
                log_probs.append(log_prob)
                entropys.append(entropy)
        values = torch.cat(values, dim=0).requires_grad_(True) 
        log_probs = torch.cat(log_probs, dim=0).requires_grad_(True)
        entropys = torch.cat(entropys, dim=0).requires_grad_(True)

        return values, log_probs, entropys
    
    def _get_action_dist_from_latent(self, mean, log_std):
        """Get distribution"""
        std = torch.exp(log_std)
        cov_matrix = torch.diag_embed(std.pow(2)+ 1e-6)
        dist = MultivariateNormal(mean, cov_matrix)
        return dist

class CustomActorCriticPolicySGD(ActorCriticPolicy):
    """
     Network policy.
     See: https://github.com/DLR-RM/stable-baselines3/blob/d671402c9373391f44d8a2ad11deed615e0f4bae/stable_baselines3/common/policies.py#L414

    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule = None,
        optimizer_class  = torch.optim.SGD, 
        optimizer_kwargs = None,
        *args,
        **kwargs,
        ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs["eps"] = 1e-4
        super(CustomActorCriticPolicySGD,self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            *args,
        **kwargs,
        )
        self._build(lr_schedule)
        self.iter = 0
        
    

    def _build_mlp_extractor(self) :
        """
         Create the policy and value networks.
        """
        self.mlp_extractor = SimpleCustomNetwork().to(self.device)
    def _build(self,lr_schedule: Schedule= None):
        """
        Create the networks, the optimizer, and the costraints.
        """
        self._build_mlp_extractor()
        self.mlp_extractor.initialize_parameters()
        self.optimizer = self.optimizer_class(self.parameters(),lr =lr_schedule(1), **self.optimizer_kwargs)
        xPos = (-10.2,31)
        yPos = (1,12)
        yPosv = (2.8,4.54)
        zPos = (17.4,45)
        tranp = (0,1)
        fc = (0,10) #20
        fb = (0,10)
        xbb = (0,3)
        ybb = (0,3)
        zbb = (0,3)
        Pos = [xPos,yPos,zPos]
        Posv = [xPos,yPosv,zPos]
        Bb = [xbb,ybb,zbb]
        self.max_obsc = [tup[1] for tup in Pos] + [tup[1] for tup in Bb] +[tranp[1]]
        self.min_obsc = [tup[0] for tup in Pos] + [tup[0] for tup in Bb] +[tranp[0]]
        self.max_distr = [tup[1] for tup in Bb] +[fc[1]]+[fb[1]]
        self.min_distr = [tup[0] for tup in Bb] +[fc[0]]+[fb[0]]
        self.max_malavatar = [tup[1] for tup in Posv]
        self.min_malavatar = [tup[0] for tup in Posv]
    
    def forward(self,obs,deterministic: bool = False):
        """
        Forward pass.
        """
        features = self.extract_features(obs)
        mean, log_std,values = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(mean=mean, log_std=log_std)
        actions = distribution.sample()
        masked_actions = self._mask(features[1], actions)
        masked_actions = torch.tensor(masked_actions).to(actions.device)
        actions = actions * masked_actions
        log_prob = distribution.log_prob(actions)
        actions_to_send = 9999 * torch.ones((1,124), dtype=torch.float32)
        actions_to_send[:, :actions.shape[1]] = actions.reshape(1,-1)
        return actions_to_send, values, log_prob
    
    
    def _mask(self,num_obs, actions):
        """
        Mask function
        """
        max_obsc = self.max_obsc * self.NbrObsc
        min_obsc = self.min_obsc * self.NbrObsc
        max_distr = self.max_distr * self.NbrDistr
        min_distr = self.min_distr * self.NbrDistr
        max_malavatar = self.max_malavatar * self.NbrMalavatar
        min_malavatar = self.min_malavatar * self.NbrMalavatar
        st = self._modifiable_state(num_obs.reshape(1, -1))
        st_1 =st + actions.cpu().numpy()
        max = max_obsc + max_distr + max_malavatar
        min = min_obsc + min_distr + min_malavatar
        mask = np.logical_and(min <= st_1, st_1 <= max).astype(int)
        return mask

    def _modifiable_state(self,num_obs):
        """
        Get mutable states.
        """
        nbrObscActions = self.NbrObsc * 7
        nbrDistr = self.NbrDistr * 5
        nbrMalavatar =self.NbrMalavatar * 3
        if (num_obs.shape[0] > 1):
            print("Inside the _mask")
            print(num_obs)
            print(num_obs.shape)
        obsc = num_obs[:,12: 12 + nbrObscActions]
        distr = num_obs[:,12 + nbrObscActions:12 + nbrObscActions + nbrDistr]
        malavatar = num_obs[:,12 + nbrObscActions + nbrDistr+12 :12 + nbrObscActions + nbrDistr+12 +3]
        return np.concatenate((obsc.cpu(),distr.cpu(),malavatar.cpu()),axis=1)
    
    def extract_features(self,obs):
        """
        Prepare the state space for the forward pass
        """
        vis = obs["visual_observation"]
        num = obs["numerical_observation"]
        num = self._remove_pads(num)
        #Process visual observation
        if isinstance(vis, torch.Tensor) == False:
            vis = torch.tensor(vis).clone().detach().requires_grad_(True)  # Convert to PyTorch tensor

        vis = vis.permute(0, 3, 1, 2)  # Change the order of dimensions
        #vis = vis.unsqueeze(0)
        #Process numerical observaton
        if isinstance(num, torch.Tensor) == False:
            num = torch.tensor(num).clone().detach().requires_grad_(True)  # Convert to PyTorch tensor
        num = num.unsqueeze(0)
        #num = num.unsqueeze(0)
        return vis, num
    def _remove_pads(self,array, pad =9999):
        """
        Remove pads from observations and assign the number of each object.
        Unity requires fixed length states and actions.
        """
        
        nbrs = array[:,-3:]
        try:
            self.NbrObsc ,self.NbrDistr,self.NbrMalavatar = int(nbrs[:,0]),int(nbrs[:,1]),int(nbrs[:,2])
            
        except : 
            self.NbrObsc ,self.NbrDistr,self.NbrMalavatar = nbrs[:,0].to(torch.int),int(nbrs[:,1]).to(torch.int),int(nbrs[:,2]).to(torch.int)
            print(self.NbrObsc)
            print(self.NbrDistr)
            print(self.NbrMalavatar)
        array = array[:,:-3]
        #end_index = np.argmax(array[] == pad)
        end_index = int(np.argmax(array.cpu() == pad, axis=1))
        if end_index == 0:
            return array
        else: 
            return array[:,:end_index]
    def _remove_pads_actions(self,array,pad = 9999):
        """
        Remove pads from actions
        """
        end_index = int(np.argmax(array.cpu() == pad, axis=1))
        if end_index == 0:
            return array
        else: 
            return array[:,:end_index]
    def get_distribution(self, obs):
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = self.extract_features(obs)
        mean, log_std,values = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(mean=mean, log_std=log_std)
        return distribution
    def _predict(self, obs, deterministic = False) :
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        features = self.extract_features(obs)
        distribution = self.get_distribution(obs)
        actions = distribution.sample()
        masked_actions = self._mask(features[1], actions)
        actions = actions * masked_actions
        actions_to_send = torch.ones((1,124), dtype=torch.float32)
        actions_to_send[:, :actions.shape[1]] = actions.reshape(1,-1)
        return actions_to_send
    def predict_values(self, obs):
        """
        Value function v(s)
        """
        features = self.extract_features(obs)
        _, _,values = self.mlp_extractor(features)
        return values
    def evaluate_actions(self, obs, actions) :
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        self.iter += 1
        #print(list(self.mlp_extractor.parameters()))
        """for name, param in self.mlp_extractor.named_parameters():
            if param.grad is not None:
                print(f"Parameter {name}:")
                print(f"Gradients: {param.grad}")"""
        #torch.save(self.mlp_extractor.state_dict(), 'model_weights' + str(self.iter) + '.pth')
        visuals = obs["visual_observation"]
        nums = obs["numerical_observation"]
        filename = "data.json"
        if self.iter < 2 : 
            with open(filename, "w") as json_file:
                json.dump({"num": nums.tolist()}, json_file)
        values = []
        log_probs = []
        entropys=[]
        for i in range(nums.shape[0]):
                vis, num = visuals[i].unsqueeze(0),nums[i].unsqueeze(0)
                ob = { "numerical_observation": num,"visual_observation": vis}
                action = actions[i].unsqueeze(0)
                action = self._remove_pads_actions(action)
                features = self.extract_features(ob)
                mean, log_std,value = self.mlp_extractor(features)
                try: 
                    distribution = self._get_action_dist_from_latent(mean=mean, log_std=log_std)
                except:
                    torch.save(self.mlp_extractor.state_dict(), 'failed_model_weights.pth')
                    print(features)
                    print(self.NbrObsc)
                    print(self.NbrDistr)
                    print(self.NbrMalavatar)
                    print(action.shape)
                    print(mean.shape)
                    print(action)
                    print(mean)
                    print(log_std)
                log_prob = distribution.log_prob(action)
                entropy = distribution.entropy()
                values.append(value)
                log_probs.append(log_prob)
                entropys.append(entropy)
        values = torch.cat(values, dim=0).requires_grad_(True) 
        log_probs = torch.cat(log_probs, dim=0).requires_grad_(True)
        entropys = torch.cat(entropys, dim=0).requires_grad_(True)

        return values, log_probs, entropys
    
    def _get_action_dist_from_latent(self, mean, log_std):
        """Get distribution"""
        std = torch.exp(log_std)
        cov_matrix = torch.diag_embed(std.pow(2)+ 1e-6)
        dist = MultivariateNormal(mean, cov_matrix)
        return dist