import torch
import torch.nn as nn
import einops


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)
    
class up_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.ReLU(inplace=True), 
        )
        self.up_conv2 = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, kernel_size=3, padding=1, stride=1),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, skip):
        x = self.up_conv1(x)
        # Ensure both tensors are contiguous before concatenation
        x_contiguous = x.contiguous()
        skip_contiguous = skip.contiguous()
        concatenated = torch.cat((x_contiguous, skip_contiguous), dim=1)
        return self.up_conv2(concatenated.contiguous())
    

class down_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.down_conv(x)
    

class FiLMLayer(nn.Module):
    def __init__(self, dim_global_feature, out_channels, dropout=0.0):
        super().__init__()
        self.out_channels = out_channels

        # Build MLP with optional dropout for regularization
        layers = [
            nn.Linear(dim_global_feature, dim_global_feature * 2),
            nn.ReLU()
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dim_global_feature * 2, 2 * out_channels))  # 输出2*C用于scale和bias

        self.model = nn.Sequential(*layers)
    
    def forward(self, global_features, x):
        # x: (B, C, T) for 1D conv or (B, C, H, W) for 2D conv
        # global_features: (B, global_dim)
        film_params = self.model(global_features)  # (B, 2*C)
        scale, bias = torch.chunk(film_params, 2, dim=1)  # 分成两个(B, C)

        # Reshape for broadcasting - handle both 1D and 2D cases
        if x.dim() == 3:  # 1D conv case: (B, C, T)
            scale = scale.contiguous().view(-1, self.out_channels, 1)
            bias = bias.contiguous().view(-1, self.out_channels, 1)
        else:  # 2D conv case: (B, C, H, W)
            scale = scale.contiguous().view(-1, self.out_channels, 1, 1)
            bias = bias.contiguous().view(-1, self.out_channels, 1, 1)

        return scale * x + bias

class Unet(nn.Module):
    def __init__(self, dim_global_features, input_dim=2, dropout=0.0):
        super().__init__()
        channels = [input_dim, 256, 512, 1024]

        self.down_convs = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        self.film_layers_down = nn.ModuleList(
            [FiLMLayer(dim_global_features, channels[i+1], dropout=dropout) for i in range(len(channels)-1)]
        )

        for i in range(len(channels)-1):
            self.down_convs.append(conv_block(channels[i], channels[i+1]))
            if i < len(channels)-2:  # 最后一层不需要下采样
                self.down_samples.append(down_conv(channels[i+1], channels[i+1]))

        self.up_samples = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        self.film_layers_up = nn.ModuleList(
            [FiLMLayer(dim_global_features, channels[i], dropout=dropout) for i in range(len(channels) - 2, 0, -1)]
        )
        
        for i in range(len(channels)-2, 0, -1):  # 从倒数第二层开始
            self.up_samples.append(up_conv(channels[i+1], channels[i]))
            self.up_convs.append(conv_block(channels[i], channels[i]))
        
        self.final_conv = nn.Conv1d(channels[1], input_dim, kernel_size=1)
        
    def forward(self, x, global_features):
        # global_features: (B, dim_imgs_emd + dim_time_emd + dim_pos_emd)
        skip_connections = []

        for i, (conv, down) in enumerate(zip(self.down_convs[:-1], self.down_samples)):
            x = conv(x)
            x = self.film_layers_down[i](global_features, x)
            skip_connections.append(x)  # 保存skip connection
            x = down(x)

        x = self.down_convs[-1](x)

        for i, (up, conv) in enumerate(zip(self.up_samples, self.up_convs)):
            skip = skip_connections[-(i+1)]  # 从后往前取skip connection
            x = up(x, skip)
            x = self.film_layers_up[i](global_features, x)
            x = conv(x)

        x = self.final_conv(x)
        return x

