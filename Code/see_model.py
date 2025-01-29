
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from ia.constants import DEVICE
from ia.dqn import DQN

WEIGHTS_FILE = "trained_models/circuit_4goal_newmodel" 
INP_SIZE = (3, 201, 151)
OUT_SIZE = 4
def plot_conv_params(model):
    conv_layers = [model.conv1, model.conv2, model.conv3]
    
    for i, conv in enumerate(conv_layers):
        weights = conv.weight.data.cpu().numpy() 
        num_filters = weights.shape[0] 
        num_channels = weights.shape[1] 
        
        fig, axes = plt.subplots(num_filters, num_channels, figsize=(num_channels * 2, num_filters * 2))
        fig.suptitle(f'Conv Layer {i+1} Weights')
        
        for f in range(num_filters):
            for c in range(num_channels):
                ax = axes[f, c] if num_filters > 1 and num_channels > 1 else axes[max(f, c)]
                ax.imshow(weights[f, c], cmap='gray')
                ax.axis('off')
        
        plt.show()

def generate_max_activation(model, input_size=INP_SIZE, lr=0.1, steps=1000):
    model.eval()
    
    conv_layers = [model.conv1, model.conv2, model.conv3]
    
    for layer_idx, layer in enumerate(conv_layers):
        num_filters = layer.weight.shape[0]
        
        for filter_idx in range(num_filters):
            input_tensor = torch.randn((1, *input_size), requires_grad=True, device=DEVICE)
            optimizer = torch.optim.Adam([input_tensor], lr=lr)
            
            for _ in range(steps):
                optimizer.zero_grad()
                x = input_tensor
                x = F.relu(model.conv1(x))
                if layer_idx >= 1:
                    x = F.relu(model.conv2(x))
                if layer_idx >= 2:
                    x = F.relu(model.conv3(x))
                
                loss = -x[0, filter_idx].mean()
                loss.backward()
                optimizer.step()
            
            img = input_tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())
            plt.imshow(img)
            plt.title(f'Activation Maximale - Layer {layer_idx+1}, Filter {filter_idx}')
            plt.axis('off')
            plt.show()

dqn = DQN(input_size=INP_SIZE, output_features=OUT_SIZE).to(DEVICE)
dqn.load_state_dict(torch.load(WEIGHTS_FILE, map_location=DEVICE))

#plot_conv_params(dqn)
generate_max_activation(dqn)