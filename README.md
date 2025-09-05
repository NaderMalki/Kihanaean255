# Artificial intelligence neural network with 3D and dual-hemisphere processing
class FullModel(nn.Module):
    def __init__(self, input_dim, seq_len, input_proj_dim=128, hidden_size=64, 
                 num_layers=2, bidirectional=True, hemisphere_fc_size=64, 
                 lstm_dropout=0.1, fc_dropout=0.3, use_attention=False, 
                 mlp_hidden_sizes=[128, 64], n_classes=2, init_type='xavier'):
        super().__init__()

        # پیش‌پردازش / projection ورودی
        self.projector = InputProjector(input_dim, input_proj_dim, use_layernorm=True) \
            if input_dim != input_proj_dim else nn.Identity()

        # بلوک هماهنگ دو نیمکره
        self.dual = CoordinatedDualHemisphere(
            input_size=input_proj_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            fc_size=hemisphere_fc_size,
            lstm_dropout=lstm_dropout,
            fc_dropout=fc_dropout,
            use_attention=use_attention
        )

        # MLP برای طبقه‌بندی نهایی
        combined_size = hemisphere_fc_size * 2  # برای دو نیمکره
        mlp_layers = []
        in_features = combined_size

        for hidden_size in mlp_hidden_sizes:
            mlp_layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU(),
                nn.Dropout(fc_dropout)
            ])
            in_features = hidden_size

        mlp_layers.append(nn.Linear(in_features, n_classes))
        self.mlp = nn.Sequential(*mlp_layers)

        # مقداردهی اولیه وزن‌ها
        self._init_weights(init_type)

    def _init_weights(self, init_type):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_type == 'xavier':
                    nn.init.xavier_uniform_(module.weight)
                elif init_type == 'kaiming':
                    nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x_left, x_right):
        # Project input
        x_left_proj = self.projector(x_left)
        x_right_proj = self.projector(x_right)

        # Process through dual hemispheres
        features = self.dual(x_left_proj, x_right_proj)

        # Final classification
        output = self.mlp(features)

        return output
