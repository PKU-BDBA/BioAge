from torch import nn
import torch

class CrossAttentionModule(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.query = nn.Linear(channel_size, channel_size)
        self.key = nn.Linear(channel_size, channel_size)
        self.value = nn.Linear(channel_size, channel_size)

    def forward(self, x1, x2, x3):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x3)

        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        attention_scores = torch.softmax(attention_scores, dim=-1)

        attended_values = torch.matmul(attention_scores, v)
        return attended_values
