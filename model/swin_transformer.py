from torch import nn

class SwinTransformerEncoder(nn.Module):
    def __init__(self, num_blocks, channels, num_classes):
        super().__init__()
        self.blocks = nn.Sequential(*[
            SwinTransformerBlock(channels) for _ in range(num_blocks)
        ])
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.blocks(x)
        x = x.mean(dim=[2, 3])  # Global average pooling
        x = self.classifier(x)
        return x
