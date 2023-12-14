from torch import nn

class MLPHeader(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

class ClassifierModel(nn.Module):
    def __init__(self, num_classes, encoder_dim, hidden_dim, num_encoder_blocks):
        super().__init__()
        # Encoders for each modality
        self.face_encoder = SwinTransformerEncoder(num_encoder_blocks, encoder_dim, num_classes)
        self.tongue_encoder = SwinTransformerEncoder(num_encoder_blocks, encoder_dim, num_classes)
        self.fundus_encoder = SwinTransformerEncoder(num_encoder_blocks, encoder_dim, num_classes)

        # Cross-Attention module
        self.cross_attention = CrossAttentionModule(encoder_dim)

        # MLP headers for each modality
        self.face_mlp = MLPHeader(encoder_dim, hidden_dim, num_classes)
        self.tongue_mlp = MLPHeader(encoder_dim, hidden_dim, num_classes)
        self.fundus_mlp = MLPHeader(encoder_dim, hidden_dim, num_classes)

        # Final classifier
        self.final_classifier = nn.Linear(num_classes * 3, num_classes)

    def forward(self, face, tongue, fundus):
        face_features = self.face_encoder(face)
        tongue_features = self.tongue_encoder(tongue)
        fundus_features = self.fundus_encoder(fundus)

        # Cross-attention
        combined_features = self.cross_attention(face_features, tongue_features, fundus_features)

        # MLP headers
        face_out = self.face_mlp(face_features)
        tongue_out = self.tongue_mlp(tongue_features)
        fundus_out = self.fundus_mlp(fundus_features)

        # Concatenate all features
        total_features = torch.cat((face_out, tongue_out, fundus_out, combined_features), dim=1)

        # Final prediction
        prediction = self.final_classifier(total_features)
        return prediction
