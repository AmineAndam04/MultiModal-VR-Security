import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

class InverseLeakyReLU(nn.Module):
    def __init__(self, positive_slope=0.01):
        super(InverseLeakyReLU, self).__init__()
        self.positive_slope = positive_slope

    def forward(self, x):
        return torch.where(x > 0, self.positive_slope * x, x)
class InverseReLU(nn.Module):
    def forward(self, x):
        return torch.min(x, torch.zeros_like(x))

class SimpleCustomNetwork(nn.Module):
    """
    Network architecture.
    Network for policies and values.
    Parameters are shared.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=4, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.Conv1d(in_channels=4, out_channels=4, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.Conv1d(in_channels=4, out_channels=1, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.01))
        
        self.decoder = nn.Sequential(nn.ConvTranspose1d(in_channels=1, out_channels=4, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.ConvTranspose1d(in_channels=4, out_channels=4, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.ConvTranspose1d(in_channels=4, out_channels=1, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.01))
        image = [torch.nn.Conv2d(in_channels = 3,out_channels= 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(1,2),
                nn.Conv1d(in_channels=64*28,out_channels=1 ,kernel_size= 3)]
        self.image = nn.Sequential(*image)
        self.fusion = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=4, kernel_size= 26),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.Conv1d(in_channels=4, out_channels=1, kernel_size= 26))
        # Heads for mean and standard deviation
        self.mean_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1),
            nn.Flatten()
        )
        
        self.std_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1),
            nn.Flatten()
        )
        # Head for value function
        self.value_head =  nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3),
            InverseReLU(),#InverseLeakyReLU(positive_slope= 0.01),#nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3),
            nn.Flatten(),
            nn.AdaptiveMaxPool1d(1))
        #self.initialize_parameters() we initialize in custom_policy.py
    def initialize_parameters(self):
        for name, module in self.named_children():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d)):
                # Initialize convolutional layers with orthogonal initialization and scaling factor
                init.orthogonal_(module.weight, np.sqrt(2))
                init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Initialize linear layers (fully connected layers) with orthogonal initialization and scaling factor
                init.orthogonal_(module.weight, np.sqrt(2))
                init.constant_(module.bias, 0)
            elif "mean_head" in name:
                
                # Initialize mean_head layer weights with the scale of 0.01
                init.orthogonal_(module[0].weight, 0.01)
                init.constant_(module[0].bias, 0)
            elif "std_head" in name:
                
                # Initialize std_head layer weights with the scale of 0.01
                init.orthogonal_(module[0].weight, 0.01)
                init.constant_(module[0].bias, -2)
            elif "value_head" in name:
                
                # Initialize value_head layer weights with the scale of 1
                init.orthogonal_(module[0].weight, 1)
                init.constant_(module[0].bias, 0)

    def forward(self,features):
        vis, num = features[0],features[1]
        # Visual features extractor
        vis_out = self.image(vis)
        # Numerical features extractor
        enc = self.encoder(num)
        dec = self.decoder(enc)
        # Fusion of num and vis 
        z = torch.cat((vis_out,dec),dim=2)
        output = self.fusion(z)
        # Outputs
        mean = self.mean_head(output)
        log_std = self.std_head(output)
        values = self.value_head(output)
        return mean, log_std,values 
    def check_initialization(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.Linear)):
                weights = module.weight
                if weights is not None:
                    print(f"Layer: {name}")
                    print(f"Weight Statistics:")
                    print(f"  - Mean: {weights.mean().item()}")
                    print(f"  - Standard Deviation: {weights.std().item()}")
                    print(f"  - Min: {weights.min().item()}")
                    print(f"  - Max: {weights.max().item()}")

class DeepCustomNetwork(nn.Module):
    """
    Network architecture.
    Network for policies and values.
    Parameters are shared.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=32, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv1d(in_channels=32, out_channels=64, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv1d(in_channels=64, out_channels=64, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv1d(in_channels=64, out_channels=1, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1))
        
        self.decoder = nn.Sequential(nn.ConvTranspose1d(in_channels=1, out_channels=32, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1))
        image = [torch.nn.Conv2d(in_channels = 3,out_channels= 32, kernel_size=7, stride=1, padding=0),
                nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2 ,stride=1),
                torch.nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=0),
                nn.ReLU(),
                torch.nn.Conv2d(64, 32, kernel_size=5, stride=3, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2 ,stride=1),
                nn.Flatten(1,2),
                nn.Conv1d(in_channels=1216,out_channels=1 ,kernel_size= 1,stride=1)]
        self.image = nn.Sequential(*image)
        self.fusion = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=8, kernel_size= 12),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.Conv1d(in_channels=8, out_channels=8, kernel_size= 12),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.MaxPool1d(kernel_size = 12, stride=1),
                                 nn.Conv1d(in_channels=8, out_channels=4, kernel_size= 12),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.Conv1d(in_channels=4, out_channels=1, kernel_size= 10),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.MaxPool1d(kernel_size = 10, stride=1),
                                 )
        # Heads for mean and standard deviation
        self.mean_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1),
            nn.Flatten()
        )
        
        self.std_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1),
            nn.Flatten()
        )
        # Head for value function
        self.value_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3),
            InverseLeakyReLU(positive_slope= 0.01),#nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3),
            nn.Flatten(),
            nn.AdaptiveMaxPool1d(1))
        
    def initialize_parameters(self):
        for name, module in self.named_children():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d,nn.MaxPool2d)):
                # Initialize convolutional layers with orthogonal initialization and scaling factor
                init.orthogonal_(module.weight, np.sqrt(2))
                init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Initialize linear layers (fully connected layers) with orthogonal initialization and scaling factor
                init.orthogonal_(module.weight, np.sqrt(2))
                init.constant_(module.bias, 0)
            elif "mean_head" in name:
                
                # Initialize mean_head layer weights with the scale of 0.01
                init.orthogonal_(module[0].weight, 0.01)
                init.constant_(module[0].bias, 0)
            elif "std_head" in name:
                
                # Initialize std_head layer weights with the scale of 0.01
                init.orthogonal_(module[0].weight, 0.01)
                init.constant_(module[0].bias, -2)
            elif "value_head" in name:
                
                # Initialize value_head layer weights with the scale of 1
                init.orthogonal_(module[0].weight, 1)
                init.constant_(module[0].bias, 0)

    def forward(self,features):
        vis, num = features[0],features[1]
        # Visual features extractor
        vis_out = self.image(vis)
        # Numerical features extractor
        enc = self.encoder(num)
        dec = self.decoder(enc)
        ######## Weights########
        ##vis_transformed = self.visual_transform(vis_out)
        ##num_transformed = self.num_transform(dec)
        ##scores = vis_transformed*num_transformed
        ##weights = self.softmax(scores)
        ##vis_out = vis_out * weights[0][0][0]
        ##dec = dec * weights[0][0][1]
        #########End of weights #######@
        # Fusion of num and vis 
        z = torch.cat((vis_out,dec),dim=2)
        output = self.fusion(z)
        # Outputs
        mean = self.mean_head(output)
        log_std = self.std_head(output)
        values = self.value_head(output)
        return mean, log_std,values 



    

class DeepCustomNetworkWithAdaptiveAvgPool1d(nn.Module):
    """
    Network architecture.
    Network for policies and values.
    Parameters are shared.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=32, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv1d(in_channels=32, out_channels=64, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv1d(in_channels=64, out_channels=64, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv1d(in_channels=64, out_channels=1, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1))
        
        self.decoder = nn.Sequential(nn.ConvTranspose1d(in_channels=1, out_channels=32, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1))
        image = [torch.nn.Conv2d(in_channels = 3,out_channels= 32, kernel_size=8, stride=4, padding=0),
                nn.LeakyReLU(negative_slope=0.1),
                torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.LeakyReLU(negative_slope=0.1),
                torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Flatten(1,2),
                nn.Conv1d(in_channels=64*28,out_channels=1 ,kernel_size= 3)]
        self.image = nn.Sequential(*image)
        self.fusion = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=4, kernel_size= 26),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.Conv1d(in_channels=4, out_channels=1, kernel_size= 26),
                                 nn.LeakyReLU(negative_slope=0.01))
        # Heads for mean and standard deviation
        self.mean_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1),
            nn.Flatten()
        )
        
        self.std_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1),
            nn.Flatten()
        )
        # Head for value function
        self.value_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1),
            nn.Flatten(),
            nn.AdaptiveAvgPool1d(1))
        #self.initialize_parameters()
    def initialize_parameters(self):
        for name, module in self.named_children():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d)):
                # Initialize convolutional layers with orthogonal initialization and scaling factor
                init.orthogonal_(module.weight, np.sqrt(2))
                init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Initialize linear layers (fully connected layers) with orthogonal initialization and scaling factor
                init.orthogonal_(module.weight, np.sqrt(2))
                init.constant_(module.bias, 0)
            elif "mean_head" in name:
                
                # Initialize mean_head layer weights with the scale of 0.01
                init.orthogonal_(module[0].weight, 0.01)
                init.constant_(module[0].bias, 0)
            elif "std_head" in name:
                
                # Initialize std_head layer weights with the scale of 0.01
                init.orthogonal_(module[0].weight, 0.01)
                init.constant_(module[0].bias,-2)
            elif "value_head" in name:
                
                # Initialize value_head layer weights with the scale of 1
                init.orthogonal_(module[0].weight, 1)
                init.constant_(module[0].bias, 0)

    def forward(self,features):
        vis, num = features[0],features[1]
        # Visual features extractor
        vis_out = self.image(vis)
        # Numerical features extractor
        enc = self.encoder(num)
        dec = self.decoder(enc)
        # Fusion of num and vis 
        z = torch.cat((vis_out,dec),dim=2)
        output = self.fusion(z)
        # Outputs
        mean = self.mean_head(output)
        log_std = self.std_head(output)
        values = self.value_head(output)
        return mean, log_std,values 

class DeepCustomNetworkTanh(nn.Module):
    """
    Network architecture.
    Network for policies and values.
    Parameters are shared.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=32, kernel_size= 3),
                                 nn.Tanh(),
                                 nn.Conv1d(in_channels=32, out_channels=64, kernel_size= 3),
                                 nn.Tanh(),
                                 nn.Conv1d(in_channels=64, out_channels=64, kernel_size= 3),
                                 nn.Tanh(),
                                 nn.Conv1d(in_channels=64, out_channels=1, kernel_size= 3),
                                 nn.Tanh())
        
        self.decoder = nn.Sequential(nn.ConvTranspose1d(in_channels=1, out_channels=32, kernel_size= 3),
                                 nn.Tanh(),
                                 nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size= 3),
                                 nn.Tanh(),
                                 nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size= 3),
                                 nn.Tanh(),
                                 nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size= 3),
                                 nn.Tanh())
        image = [torch.nn.Conv2d(in_channels = 3,out_channels= 32, kernel_size=7, stride=1, padding=0),
                nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2 ,stride=1),
                torch.nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=0),
                nn.ReLU(),
                torch.nn.Conv2d(64, 32, kernel_size=5, stride=3, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2 ,stride=1),
                nn.Flatten(1,2),
                nn.Conv1d(in_channels=1216,out_channels=1 ,kernel_size= 1,stride=1)]
        self.image = nn.Sequential(*image)
        self.fusion = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=8, kernel_size= 12),
                                 nn.Tanh(),
                                 nn.Conv1d(in_channels=8, out_channels=8, kernel_size= 12),
                                 nn.Tanh(),
                                 nn.MaxPool1d(kernel_size = 12, stride=1),
                                 nn.Conv1d(in_channels=8, out_channels=4, kernel_size= 12),
                                 nn.Tanh(),
                                 nn.Conv1d(in_channels=4, out_channels=1, kernel_size= 10),
                                 nn.MaxPool1d(kernel_size = 10, stride=1)
                                 )
        # Heads for mean and standard deviation
        self.mean_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1),
            nn.Flatten()
        )
        
        self.std_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1),
            nn.Flatten()
        )
        # Head for value function
        self.value_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3),
            nn.Flatten(),
            nn.AdaptiveMaxPool1d(1))
        
        
    def initialize_parameters(self):
        for name, module in self.named_children():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d,nn.MaxPool2d)):
                # Initialize convolutional layers with orthogonal initialization and scaling factor
                init.orthogonal_(module.weight, np.sqrt(2))
                init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Initialize linear layers (fully connected layers) with orthogonal initialization and scaling factor
                init.orthogonal_(module.weight, np.sqrt(2))
                init.constant_(module.bias, 0)
            elif "mean_head" in name:
                
                # Initialize mean_head layer weights with the scale of 0.01
                init.orthogonal_(module[0].weight, 0.01)
                init.constant_(module[0].bias, 0)
            elif "std_head" in name:
                
                # Initialize std_head layer weights with the scale of 0.01
                init.orthogonal_(module[0].weight, 0.01)
                init.constant_(module[0].bias, -2)
            elif "value_head" in name:
                
                # Initialize value_head layer weights with the scale of 1
                init.orthogonal_(module[0].weight, 1)
                init.constant_(module[0].bias, 0)

    def forward(self,features):
        vis, num = features[0],features[1]
        # Visual features extractor
        vis_out = self.image(vis)
        # Numerical features extractor
        enc = self.encoder(num)
        dec = self.decoder(enc)
        ######## Weights########
        ##vis_transformed = self.visual_transform(vis_out)
        ##num_transformed = self.num_transform(dec)
        ##scores = vis_transformed*num_transformed
        ##weights = self.softmax(scores)
        ##vis_out = vis_out * weights[0][0][0]
        ##dec = dec * weights[0][0][1]
        #########End of weights #######@
        # Fusion of num and vis 
        z = torch.cat((vis_out,dec),dim=2)
        output = self.fusion(z)
        # Outputs
        mean = self.mean_head(output)
        log_std = self.std_head(output)
        values = self.value_head(output)
        return mean, log_std,values 
    

class DeepCustomNetworkWithAttention(nn.Module):
    """
    Network architecture.
    Network for policies and values.
    Parameters are shared.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=32, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv1d(in_channels=32, out_channels=64, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv1d(in_channels=64, out_channels=64, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv1d(in_channels=64, out_channels=1, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1))
        
        self.decoder = nn.Sequential(nn.ConvTranspose1d(in_channels=1, out_channels=32, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1))
        image = [torch.nn.Conv2d(in_channels = 3,out_channels= 32, kernel_size=7, stride=1, padding=0),
                nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2 ,stride=1),
                torch.nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=0),
                nn.ReLU(),
                torch.nn.Conv2d(64, 32, kernel_size=5, stride=3, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2 ,stride=1),
                nn.Flatten(1,2),
                nn.Conv1d(in_channels=1216,out_channels=1 ,kernel_size= 1,stride=1)]
        self.image = nn.Sequential(*image)
        self.fusion = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=8, kernel_size= 12),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.Conv1d(in_channels=8, out_channels=8, kernel_size= 12),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.MaxPool1d(kernel_size = 12, stride=1),
                                 nn.Conv1d(in_channels=8, out_channels=4, kernel_size= 12),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.Conv1d(in_channels=4, out_channels=1, kernel_size= 10),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.MaxPool1d(kernel_size = 10, stride=1),
                                 )
        # Heads for mean and standard deviation
        self.mean_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1),
            nn.Flatten()
        )
        
        self.std_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1),
            nn.Flatten()
        )
        # Head for value function
        self.value_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3),
            nn.Flatten(),
            nn.AdaptiveMaxPool1d(1))
        self.visual_transform = nn.Sequential(
            nn.Linear(38, 2)
            )
        self.num_transform = nn.Sequential(
           nn.AdaptiveAvgPool1d(38),
           nn.Linear(38, 2)
         )
        self.softmax = nn.Softmax(dim=-1)
        
    def initialize_parameters(self):
        for name, module in self.named_children():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d,nn.MaxPool2d)):
                # Initialize convolutional layers with orthogonal initialization and scaling factor
                init.orthogonal_(module.weight, np.sqrt(2))
                init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Initialize linear layers (fully connected layers) with orthogonal initialization and scaling factor
                init.orthogonal_(module.weight, np.sqrt(2))
                init.constant_(module.bias, 0)
            elif "mean_head" in name:
                
                # Initialize mean_head layer weights with the scale of 0.01
                init.orthogonal_(module[0].weight, 0.01)
                init.constant_(module[0].bias, 0)
            elif "std_head" in name:
                
                # Initialize std_head layer weights with the scale of 0.01
                init.orthogonal_(module[0].weight, 0.01)
                init.constant_(module[0].bias, -2)
            elif "value_head" in name:
                
                # Initialize value_head layer weights with the scale of 1
                init.orthogonal_(module[0].weight, 1)
                init.constant_(module[0].bias, 0)

    def forward(self,features):
        vis, num = features[0],features[1]
        # Visual features extractor
        vis_out = self.image(vis)
        # Numerical features extractor
        enc = self.encoder(num)
        dec = self.decoder(enc)
        ######## Weights########
        vis_transformed = self.visual_transform(vis_out)
        num_transformed = self.num_transform(dec)
        scores = vis_transformed*num_transformed
        weights = self.softmax(scores)
        vis_out = vis_out * weights[0][0][0]
        dec = dec * weights[0][0][1]
        #########End of weights #######@
        # Fusion of num and vis 
        z = torch.cat((vis_out,dec),dim=2)
        output = self.fusion(z)
        # Outputs
        mean = self.mean_head(output)
        log_std = self.std_head(output)
        values = self.value_head(output)
        return mean, log_std,values

class SimpleCustomNetworkTanh(nn.Module):
    """
    Network architecture.
    Network for policies and values.
    Parameters are shared.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=4, kernel_size= 3),
                                 nn.Tanh(),
                                 nn.Conv1d(in_channels=4, out_channels=4, kernel_size= 3),
                                 nn.Tanh(),
                                 nn.Conv1d(in_channels=4, out_channels=1, kernel_size= 3),
                                 nn.Tanh())
        
        self.decoder = nn.Sequential(nn.ConvTranspose1d(in_channels=1, out_channels=4, kernel_size= 3),
                                 nn.Tanh(),
                                 nn.ConvTranspose1d(in_channels=4, out_channels=4, kernel_size= 3),
                                 nn.Tanh(),
                                 nn.ConvTranspose1d(in_channels=4, out_channels=1, kernel_size= 3),
                                 nn.Tanh())
        image = [torch.nn.Conv2d(in_channels = 3,out_channels= 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(1,2),
                nn.Conv1d(in_channels=64*28,out_channels=1 ,kernel_size= 3)]
        self.image = nn.Sequential(*image)
        self.fusion = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=4, kernel_size= 26),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.Conv1d(in_channels=4, out_channels=1, kernel_size= 26))
        # Heads for mean and standard deviation
        self.mean_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1),
            nn.Flatten()
        )
        
        self.std_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1),
            nn.Flatten()
        )
        # Head for value function
        self.value_head =  nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3),
            nn.Flatten(),
            nn.AdaptiveMaxPool1d(1))
        #self.initialize_parameters()
    def initialize_parameters(self):
        for name, module in self.named_children():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d)):
                # Initialize convolutional layers with orthogonal initialization and scaling factor
                init.orthogonal_(module.weight, np.sqrt(2))
                init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Initialize linear layers (fully connected layers) with orthogonal initialization and scaling factor
                init.orthogonal_(module.weight, np.sqrt(2))
                init.constant_(module.bias, 0)
            elif "mean_head" in name:
                
                # Initialize mean_head layer weights with the scale of 0.01
                init.orthogonal_(module[0].weight, 0.01)
                init.constant_(module[0].bias, 0)
            elif "std_head" in name:
                
                # Initialize std_head layer weights with the scale of 0.01
                init.orthogonal_(module[0].weight, 0.01)
                init.constant_(module[0].bias, -2)
            elif "value_head" in name:
                
                # Initialize value_head layer weights with the scale of 1
                init.orthogonal_(module[0].weight, 1)
                init.constant_(module[0].bias, 0)

    def forward(self,features):
        vis, num = features[0],features[1]
        # Visual features extractor
        vis_out = self.image(vis)
        # Numerical features extractor
        enc = self.encoder(num)
        dec = self.decoder(enc)
        # Fusion of num and vis 
        z = torch.cat((vis_out,dec),dim=2)
        output = self.fusion(z)
        # Outputs
        mean = self.mean_head(output)
        log_std = self.std_head(output)
        values = self.value_head(output)
        return mean, log_std,values 
    
class EnhancedCustomNetwork(nn.Module):
    """
    Add a layer to the encoder and decoder.
    Remove the activation function from the last layer.
    Fusion: use a negative slop of 0.1, and remove the activation from the last layer.
    Value-head: use a custom neural network
    Initialize std head to  -6
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=32, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv1d(in_channels=32, out_channels=64, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv1d(in_channels=64, out_channels=64, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv1d(in_channels=64, out_channels=32, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv1d(in_channels=32, out_channels=1, kernel_size= 3))
        
        self.decoder = nn.Sequential(nn.ConvTranspose1d(in_channels=1, out_channels=32, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size= 3))
        image = [torch.nn.Conv2d(in_channels = 3,out_channels= 32, kernel_size=7, stride=1, padding=0),
                nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2 ,stride=1),
                torch.nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=0),
                nn.ReLU(),
                torch.nn.Conv2d(64, 32, kernel_size=5, stride=3, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2 ,stride=1),
                nn.Flatten(1,2),
                nn.Conv1d(in_channels=1216,out_channels=1 ,kernel_size= 1,stride=1)]
        self.image = nn.Sequential(*image)
        self.fusion = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=8, kernel_size= 12),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv1d(in_channels=8, out_channels=8, kernel_size= 12),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.MaxPool1d(kernel_size = 12, stride=1),
                                 nn.Conv1d(in_channels=8, out_channels=4, kernel_size= 12),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv1d(in_channels=4, out_channels=1, kernel_size= 10),
                                 nn.MaxPool1d(kernel_size = 10, stride=1),
                                 )
        # Heads for mean and standard deviation
        self.mean_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1),
            nn.Flatten()
        )
        
        self.std_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1),
            nn.Flatten()
        )
        # Head for value function
        self.value_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3),
            InverseLeakyReLU(positive_slope= 0.01),
            nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3),
            nn.Flatten(),
            nn.AdaptiveMaxPool1d(1))
        
    def initialize_parameters(self):
        for name, module in self.named_children():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d,nn.MaxPool2d)):
                # Initialize convolutional layers with orthogonal initialization and scaling factor
                init.orthogonal_(module.weight, np.sqrt(2))
                init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Initialize linear layers (fully connected layers) with orthogonal initialization and scaling factor
                init.orthogonal_(module.weight, np.sqrt(2))
                init.constant_(module.bias, 0)
            elif "mean_head" in name:
                
                # Initialize mean_head layer weights with the scale of 0.01
                init.orthogonal_(module[0].weight, 0.01)
                init.constant_(module[0].bias, 0)
            elif "std_head" in name:
                
                # Initialize std_head layer weights with the scale of 0.01
                init.orthogonal_(module[0].weight, 0.01)
                init.constant_(module[0].bias, -2)
            elif "value_head" in name:
                
                # Initialize value_head layer weights with the scale of 1
                init.orthogonal_(module[0].weight, 1)
                init.constant_(module[0].bias, 0)

    def forward(self,features):
        vis, num = features[0],features[1]
        # Visual features extractor
        vis_out = self.image(vis)
        # Numerical features extractor
        enc = self.encoder(num)
        dec = self.decoder(enc)
        ######## Weights########
        ##vis_transformed = self.visual_transform(vis_out)
        ##num_transformed = self.num_transform(dec)
        ##scores = vis_transformed*num_transformed
        ##weights = self.softmax(scores)
        ##vis_out = vis_out * weights[0][0][0]
        ##dec = dec * weights[0][0][1]
        #########End of weights #######@
        # Fusion of num and vis 
        z = torch.cat((vis_out,dec),dim=2)
        output = self.fusion(z)
        # Outputs
        mean = self.mean_head(output)
        log_std = self.std_head(output)
        values = self.value_head(output)
        return mean, log_std,values 

class SimpleDeepFusionCustomNetwork(nn.Module):
    """
    Network architecture.
    Network for policies and values.
    Parameters are shared.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=4, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.Conv1d(in_channels=4, out_channels=4, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.Conv1d(in_channels=4, out_channels=1, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.01))
        
        self.decoder = nn.Sequential(nn.ConvTranspose1d(in_channels=1, out_channels=4, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.ConvTranspose1d(in_channels=4, out_channels=4, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.ConvTranspose1d(in_channels=4, out_channels=1, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.01))
        image = [torch.nn.Conv2d(in_channels = 3,out_channels= 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(1,2),
                nn.Conv1d(in_channels=64*28,out_channels=1 ,kernel_size= 3)]
        self.image = nn.Sequential(*image)
        self.fusion = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=8, kernel_size= 12),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv1d(in_channels=8, out_channels=8, kernel_size= 12),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv1d(in_channels=8, out_channels=4, kernel_size= 12),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.MaxPool1d(kernel_size = 12, stride=1),
                                 nn.Conv1d(in_channels=4, out_channels=4, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv1d(in_channels=4, out_channels=1, kernel_size= 3),
                                 nn.MaxPool1d(kernel_size = 3, stride=1))
        # Heads for mean and standard deviation
        self.mean_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1),
            nn.Flatten()
        )
        
        self.std_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1),
            nn.Flatten()
        )
        # Head for value function
        self.value_head =  nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3),
            InverseReLU(), #InverseLeakyReLU(positive_slope= 0.01),#nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3),
            nn.Flatten(),
            nn.AdaptiveMaxPool1d(1))
        
        #self.initialize_parameters()
    def initialize_parameters(self):
        for name, module in self.named_children():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d)):
                # Initialize convolutional layers with orthogonal initialization and scaling factor
                init.orthogonal_(module.weight, np.sqrt(2))
                init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Initialize linear layers (fully connected layers) with orthogonal initialization and scaling factor
                init.orthogonal_(module.weight, np.sqrt(2))
                init.constant_(module.bias, 0)
            elif "mean_head" in name:
                
                # Initialize mean_head layer weights with the scale of 0.01
                init.orthogonal_(module[0].weight, 0.01)
                init.constant_(module[0].bias, 0)
            elif "std_head" in name:
                
                # Initialize std_head layer weights with the scale of 0.01
                init.orthogonal_(module[0].weight, 0.01)
                init.constant_(module[0].bias, -2)
            elif "value_head" in name:
                
                # Initialize value_head layer weights with the scale of 1
                init.orthogonal_(module[0].weight, 1)
                init.constant_(module[0].bias, 0)

    def forward(self,features):
        vis, num = features[0],features[1]
        # Visual features extractor
        vis_out = self.image(vis)
        # Numerical features extractor
        enc = self.encoder(num)
        dec = self.decoder(enc)
        # Fusion of num and vis 
        z = torch.cat((vis_out,dec),dim=2)
        output = self.fusion(z)
        # Outputs
        mean = self.mean_head(output)
        log_std = self.std_head(output)
        values = self.value_head(output)
        return mean, log_std,values 


class DeepCustomNetworkLV(nn.Module):
    """
    Network architecture.
    Network for policies and values.
    Parameters are shared.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=32, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv1d(in_channels=32, out_channels=64, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv1d(in_channels=64, out_channels=64, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv1d(in_channels=64, out_channels=1, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1))
        
        self.decoder = nn.Sequential(nn.ConvTranspose1d(in_channels=1, out_channels=32, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1))
        image = [torch.nn.Conv2d(in_channels = 3,out_channels= 32, kernel_size=7, stride=1, padding=0),
                nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2 ,stride=1),
                torch.nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=0),
                nn.ReLU(),
                torch.nn.Conv2d(64, 32, kernel_size=5, stride=3, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2 ,stride=1),
                nn.Flatten(1,2),
                nn.Conv1d(in_channels=1216,out_channels=1 ,kernel_size= 1,stride=1)]
        self.image = nn.Sequential(*image)
        self.fusion = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=8, kernel_size= 12),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.Conv1d(in_channels=8, out_channels=8, kernel_size= 12),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.MaxPool1d(kernel_size = 12, stride=1),
                                 nn.Conv1d(in_channels=8, out_channels=4, kernel_size= 12),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.Conv1d(in_channels=4, out_channels=1, kernel_size= 10),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.MaxPool1d(kernel_size = 10, stride=1),
                                 )
        # Heads for mean and standard deviation
        self.mean_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1),
            nn.Flatten()
        )
        
        self.std_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1),
            nn.Flatten()
        )
        # Head for value function
        self.value_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3),
            InverseLeakyReLU(positive_slope= 0.01),#nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3),
            nn.Flatten(),
            nn.AdaptiveMaxPool1d(1))
        
    def initialize_parameters(self):
        for name, module in self.named_children():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d,nn.MaxPool2d)):
                # Initialize convolutional layers with orthogonal initialization and scaling factor
                init.orthogonal_(module.weight, np.sqrt(2))
                init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Initialize linear layers (fully connected layers) with orthogonal initialization and scaling factor
                init.orthogonal_(module.weight, np.sqrt(2))
                init.constant_(module.bias, 0)
            elif "mean_head" in name:
                
                # Initialize mean_head layer weights with the scale of 0.01
                init.orthogonal_(module[0].weight, 0.01)
                init.constant_(module[0].bias, 0)
            elif "std_head" in name:
                
                # Initialize std_head layer weights with the scale of 0.01
                init.orthogonal_(module[0].weight, 0.01)
                init.constant_(module[0].bias, -6)
            elif "value_head" in name:
                
                # Initialize value_head layer weights with the scale of 1
                init.orthogonal_(module[0].weight, 1)
                init.constant_(module[0].bias, 0)

    def forward(self,features):
        vis, num = features[0],features[1]
        # Visual features extractor
        vis_out = self.image(vis)
        # Numerical features extractor
        enc = self.encoder(num)
        dec = self.decoder(enc)
        ######## Weights########
        ##vis_transformed = self.visual_transform(vis_out)
        ##num_transformed = self.num_transform(dec)
        ##scores = vis_transformed*num_transformed
        ##weights = self.softmax(scores)
        ##vis_out = vis_out * weights[0][0][0]
        ##dec = dec * weights[0][0][1]
        #########End of weights #######@
        # Fusion of num and vis 
        z = torch.cat((vis_out,dec),dim=2)
        output = self.fusion(z)
        # Outputs
        mean = self.mean_head(output)
        log_std = self.std_head(output)
        values = self.value_head(output)
        return mean, log_std,values 
    
class DeepCustomNetworkRR(nn.Module):
    """
    Network architecture.
    Network for policies and values.
    Parameters are shared.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=32, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv1d(in_channels=32, out_channels=64, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv1d(in_channels=64, out_channels=64, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv1d(in_channels=64, out_channels=1, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1))
        
        self.decoder = nn.Sequential(nn.ConvTranspose1d(in_channels=1, out_channels=32, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1))
        image = [torch.nn.Conv2d(in_channels = 3,out_channels= 32, kernel_size=7, stride=1, padding=0),
                nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2 ,stride=1),
                torch.nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=0),
                nn.ReLU(),
                torch.nn.Conv2d(64, 32, kernel_size=5, stride=3, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2 ,stride=1),
                nn.Flatten(1,2),
                nn.Conv1d(in_channels=1216,out_channels=1 ,kernel_size= 1,stride=1)]
        self.image = nn.Sequential(*image)
        self.fusion = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=8, kernel_size= 12),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.Conv1d(in_channels=8, out_channels=8, kernel_size= 12),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.MaxPool1d(kernel_size = 12, stride=1),
                                 nn.Conv1d(in_channels=8, out_channels=4, kernel_size= 12),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.Conv1d(in_channels=4, out_channels=1, kernel_size= 10),
                                 #nn.LeakyReLU(negative_slope=0.01),
                                 nn.MaxPool1d(kernel_size = 10, stride=1),
                                 )
        # Heads for mean and standard deviation
        self.mean_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1),
            nn.Flatten()
        )
        
        self.std_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1),
            nn.Flatten()
        )
        # Head for value function
        self.value_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3),
            InverseReLU(), #InverseLeakyReLU(positive_slope= 0.01),#nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3),
            nn.Flatten(),
            nn.AdaptiveMaxPool1d(1))
        
    def initialize_parameters(self):
        for name, module in self.named_children():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d,nn.MaxPool2d)):
                # Initialize convolutional layers with orthogonal initialization and scaling factor
                init.orthogonal_(module.weight, np.sqrt(2))
                init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Initialize linear layers (fully connected layers) with orthogonal initialization and scaling factor
                init.orthogonal_(module.weight, np.sqrt(2))
                init.constant_(module.bias, 0)
            elif "mean_head" in name:
                
                # Initialize mean_head layer weights with the scale of 0.01
                init.orthogonal_(module[0].weight, 0.01)
                init.constant_(module[0].bias, 0)
            elif "std_head" in name:
                
                # Initialize std_head layer weights with the scale of 0.01
                init.orthogonal_(module[0].weight, 0.01)
                init.constant_(module[0].bias, -2)
            elif "value_head" in name:
                
                # Initialize value_head layer weights with the scale of 1
                init.orthogonal_(module[0].weight, 1)
                init.constant_(module[0].bias, 0)

    def forward(self,features):
        vis, num = features[0],features[1]
        # Visual features extractor
        vis_out = self.image(vis)
        # Numerical features extractor
        enc = self.encoder(num)
        dec = self.decoder(enc)
        
        # Fusion of num and vis 
        z = torch.cat((vis_out,dec),dim=2)
        output = self.fusion(z)
        # Outputs
        mean = self.mean_head(output)
        log_std = self.std_head(output)
        values = self.value_head(output)
        return mean, log_std,values 

class AblationNetwork(nn.Module):
    """
    Network architecture.
    Network for policies and values.
    Parameters are shared.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=32, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv1d(in_channels=32, out_channels=64, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv1d(in_channels=64, out_channels=64, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Conv1d(in_channels=64, out_channels=1, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1))
        
        self.decoder = nn.Sequential(nn.ConvTranspose1d(in_channels=1, out_channels=32, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size= 3),
                                 nn.LeakyReLU(negative_slope=0.1))
        self.fusion = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=8, kernel_size= 9),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.Conv1d(in_channels=8, out_channels=1, kernel_size= 9),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.MaxPool1d(kernel_size = 9, stride=1),
                                 )
        # Heads for mean and standard deviation
        self.mean_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1),
            nn.Flatten()
        )
        
        self.std_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1),
            nn.Flatten()
        )
        # Head for value function
        self.value_head = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3),
            InverseLeakyReLU(positive_slope= 0.01),#nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3),
            nn.Flatten(),
            nn.AdaptiveMaxPool1d(1))
        
    def initialize_parameters(self):
        for name, module in self.named_children():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d,nn.MaxPool2d)):
                # Initialize convolutional layers with orthogonal initialization and scaling factor
                init.orthogonal_(module.weight, np.sqrt(2))
                init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Initialize linear layers (fully connected layers) with orthogonal initialization and scaling factor
                init.orthogonal_(module.weight, np.sqrt(2))
                init.constant_(module.bias, 0)
            elif "mean_head" in name:
                
                # Initialize mean_head layer weights with the scale of 0.01
                init.orthogonal_(module[0].weight, 0.01)
                init.constant_(module[0].bias, 0)
            elif "std_head" in name:
                
                # Initialize std_head layer weights with the scale of 0.01
                init.orthogonal_(module[0].weight, 0.01)
                init.constant_(module[0].bias, -2)
            elif "value_head" in name:
                
                # Initialize value_head layer weights with the scale of 1
                init.orthogonal_(module[0].weight, 1)
                init.constant_(module[0].bias, 0)

    def forward(self,features):
        vis, num = features[0],features[1]
        # Visual features extractor
        #vis_out = self.image(vis)
        # Numerical features extractor
        enc = self.encoder(num)
        dec = self.decoder(enc)
        #z = torch.cat((vis_out,dec),dim=2)
        output = self.fusion(dec)
        # Outputs
        mean = self.mean_head(output)
        log_std = self.std_head(output)
        values = self.value_head(output)
        return mean, log_std,values 
