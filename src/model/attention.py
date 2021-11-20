import torch
import torch.nn as nn


class Self_Attn(nn.Module):
    """
    compute visual feature attention energy and multiple audio feature

    Args:
        nn ([type]): [description]
    """

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()

        self.channel_in = in_dim

        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, y):

        bs, c, w, h = x. size()

        proj_query = self.query_conv(x).view(bs, -1, w*h).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(bs, -1, w*h)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy) # output shape (bs, w*h, w*h)

        proj_value = self.value_conv(y).view(bs, -1, w*h)
        out = torch.bmm(proj_value, attention)
        out = out.view(bs, c, w, h)

        out = self.gamma*out + x
        return out, attention

class SE_Attn(nn.Module):
    """
    channel Attention module 

    """
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        y = self.fc(x)
        return x * y.expand_as(x)

class MMAttn(nn.Module):
    """
    MMAttention module designed by xjw

    """
    def __init__(self, channel1, channel2):
        super().__init__()
        self.fc1 = nn.Linear(channel1, channel1, bias=True)
        self.fc2 = nn.Linear(channel2, channel2, bias=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, y):
        x = x.squeeze(2)
        y = y.squeeze(2)
        x_out = self.fc1(x)
        y_out = self.fc1(y)
        out = x + y
        out = self.tanh(out)

        attn_score = self.softmax(out)

        x = attn_score * x_out
        y = attn_score * y_out
        return x+y
