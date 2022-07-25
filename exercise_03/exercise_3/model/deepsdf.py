import torch.nn as nn
import torch


class DeepSDFDecoder(nn.Module):

    def __init__(self, latent_size):
        """
        :param latent_size: latent code vector length
        """
        super().__init__()
        dropout_prob = 0.2
        # TODO: Define model
        input_size = latent_size+3
        output_size = latent_size-3
        self.model_1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(input_size,512), name='weight', dim=0),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.utils.weight_norm(nn.Linear(512,512), name='weight', dim=0),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.utils.weight_norm(nn.Linear(512,512), name='weight', dim=0),
            nn.ReLU(),
            nn.Dropout(dropout_prob),   
            nn.utils.weight_norm(nn.Linear(512,output_size), name='weight', dim=0),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )
        self.model_2 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(512,512), name='weight', dim=0),
            nn.ReLU(),
            nn.Dropout(dropout_prob),   
            nn.utils.weight_norm(nn.Linear(512,512), name='weight', dim=0),
            nn.ReLU(),
            nn.Dropout(dropout_prob),  
            nn.utils.weight_norm(nn.Linear(512,512), name='weight', dim=0),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.utils.weight_norm(nn.Linear(512,512), name='weight', dim=0),
            nn.ReLU(),
            nn.Dropout(dropout_prob),  
            nn.Linear(512,1),                                          
        )

    def forward(self, x_in):
        """
        :param x_in: B x (latent_size + 3) tensor
        :return: B x 1 tensor
        """
        # TODO: implement forward pass
        x = self.model_1(x_in)
        x = self.model_2(torch.cat((x,x_in),dim = 1))
        return x
