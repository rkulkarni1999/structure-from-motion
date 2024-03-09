import torch
import torch.nn as nn
import numpy as np

# class NeRFmodel(nn.Module):
#     def __init__(self, embed_pos_L, embed_direction_L):
#         super(NeRFmodel, self).__init__()
#         #############################
#         # network initialization
#         #############################
#         pass 

#     def position_encoding(self, x, L):
#         #############################
#         # Implement position encoding here
#         #############################

#         pass 
#         # return y

#     def forward(self, pos, direction):
#         #############################
#         # network structure
#         #############################
#         pass
#         # return output
class NeRFmodel(nn.Module):
    # def __init__(self, embed_pos_L, embed_direction_L):
    def __init__(self, embedding_dim_pos=10, embedding_dim_direction=4, hidden_dim=128):
        super(NeRFmodel, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Linear(embedding_dim_pos*6+3, hidden_dim), nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), )
        
        # Density Estimation 
        self.block2 = nn.Sequential(
            nn.Linear(embedding_dim_pos*6 + hidden_dim +3, hidden_dim), nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), )
        
        # Color Estimation
        self.block3 = nn.Sequential(nn.Linear(embedding_dim_direction * 6 + hidden_dim + 3, hidden_dim // 2), nn.ReLU(), )
        self.block4 = nn.Sequential(nn.Linear(hidden_dim // 2, 3), nn.Sigmoid(), )

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.relu = nn.ReLU()
    
    @staticmethod
    def positional_encoding(x, L):
        
        out = [x]
        for jj in range(L):
            out.append(torch.sin(2**jj*x))
            out.append(torch.cos(2**jj*x))
        return torch.cat(out, dim=1)
        
    def forward(self, pos, direction):
        
        emb_x = self.positional_encoding(pos, self.embedding_dim_pos) # emb_x: [batch_size, embedding_dim_pos * 6]
        emb_d = self.positional_encoding(direction, self.embedding_dim_direction) # emb_d: [batch_size, embedding_dim_direction * 6]
        h = self.block1(emb_x)
        tmp = self.block2(torch.cat((h, emb_x), dim=1)) # tmp: [batch_size, hidden_dim + 1]
        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1]) # h: [batch_size, hidden_dim], sigma: [batch_size]
        h = self.block3(torch.cat((h, emb_d), dim=1)) # h: [batch_size, hidden_dim // 2]
        c = self.block4(h) # c: [batch_size, 3]
        return c, sigma
