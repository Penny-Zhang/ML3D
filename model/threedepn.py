import torch
import torch.nn as nn


class ThreeDEPN(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_features = 80

        # TODO: 4 Encoder layers
        self.encoder1 = nn.Sequential(nn.Conv3d(in_channels=2, out_channels = self.num_features, kernel_size=4, stride=2, padding=1),
                                      nn.LeakyReLU(negative_slope = 0.2))
        self.encoder2 = nn.Sequential(nn.Conv3d(in_channels=self.num_features, out_channels = self.num_features*2, kernel_size=4, stride=2, padding=1),
                                      nn.BatchNorm3d(num_features= self.num_features*2),
                                      nn.LeakyReLU(negative_slope = 0.2))
        self.encoder3 = nn.Sequential(nn.Conv3d(in_channels=self.num_features*2, out_channels = self.num_features*4, kernel_size=4, stride=2, padding=1),
                                      nn.BatchNorm3d(num_features= self.num_features*4),
                                      nn.LeakyReLU(negative_slope = 0.2))
        self.encoder4 = nn.Sequential(nn.Conv3d(in_channels=self.num_features*4, out_channels = self.num_features*8, kernel_size=4, stride=1, padding=0),
                                      nn.BatchNorm3d(num_features= self.num_features*8),
                                      nn.LeakyReLU(negative_slope = 0.2))
        

        # TODO: 2 Bottleneck layers
        self.bottleneck = nn.Sequential(nn.Linear(in_features = self.num_features*8, out_features = self.num_features*8),
                                        nn.ReLU(),
                                        nn.Linear(in_features = self.num_features*8, out_features = self.num_features*8),
                                        nn.ReLU())

        # TODO: 4 Decoder layers
        self.decoder1 = nn.Sequential(nn.ConvTranspose3d(in_channels = self.num_features*16, out_channels = self.num_features*4, kernel_size=4, stride=1, padding=0,),
                                       nn.BatchNorm3d(num_features= self.num_features*4),
                                       nn.ReLU())
        self.decoder2 = nn.Sequential(nn.ConvTranspose3d(in_channels = self.num_features*8, out_channels = self.num_features*2, kernel_size=4, stride=2, padding=1,),
                                       nn.BatchNorm3d(num_features= self.num_features*2),
                                       nn.ReLU())
        self.decoder3 = nn.Sequential(nn.ConvTranspose3d(in_channels = self.num_features*4, out_channels = self.num_features, kernel_size=4, stride=2, padding=1,),
                                       nn.BatchNorm3d(num_features= self.num_features),
                                       nn.ReLU())
        self.decoder4 = nn.Sequential(nn.ConvTranspose3d(in_channels = self.num_features*2, out_channels = 1, kernel_size=4, stride=2, padding=1,))

    def forward(self, x):
        b = x.shape[0]
        # Encode
        # TODO: Pass x though encoder while keeping the intermediate outputs for the skip connections
        # Reshape and apply bottleneck layers
        
#         print(x)
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        
        x_e4 = x4
        x = x_e4.view(b, -1)
        x = self.bottleneck(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1, 1)
        # Decode
        # TODO: Pass x through the decoder, applying the skip connections in the process
        x = self.decoder1(torch.cat([x,x4],axis= 1))
        x = self.decoder2(torch.cat([x,x3],axis= 1))
        x = self.decoder3(torch.cat([x,x2],axis= 1))
        x = self.decoder4(torch.cat([x,x1],axis= 1))

        x = torch.squeeze(x, dim=1)
        # TODO: Log scaling
        x = torch.log(torch.abs(x) + 1)

        return x
