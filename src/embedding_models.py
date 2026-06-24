import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EmbeddingCNN(nn.Module):
    """
    CNN model with embedding layer for sequence processing.
    """

    def __init__(self, vocab_size: int = 21, embedding_dim: int = 128,
                 conv_channels: list = [64, 128, 256], kernel_sizes: list = [3, 5, 7],
                 dropout_rate: float = 0.3, output_dim: int = 1):
        """
        Args:
            vocab_size: Vocabulary size (21: 20 AAs + padding)
            embedding_dim: Embedding dimension
            conv_channels: List of convolutional channel sizes
            kernel_sizes: List of kernel sizes for each conv layer
            dropout_rate: Dropout probability
            output_dim: Number of output dimensions
        """
        super(EmbeddingCNN, self).__init__()


        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)


        self.conv_layers = nn.ModuleList()
        prev_channels = embedding_dim

        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            conv = nn.Sequential(
                nn.Conv1d(prev_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
            self.conv_layers.append(conv)
            prev_channels = out_channels


        self.global_pool = nn.AdaptiveAvgPool1d(1)


        self.fc = nn.Sequential(
            nn.Linear(prev_channels, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights (embedding already initialized in __init__)."""

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length) - token indices
        """

        x = self.embedding(x)


        x = x.transpose(1, 2)


        for conv in self.conv_layers:
            x = conv(x)


        x = self.global_pool(x).squeeze(-1)


        return self.fc(x)


class EmbeddingLSTM(nn.Module):
    """
    LSTM model with embedding layer.
    """

    def __init__(self, vocab_size: int = 21, embedding_dim: int = 128,
                 hidden_dim: int = 128, num_layers: int = 2,
                 dropout_rate: float = 0.3, output_dim: int = 1,
                 bidirectional: bool = True):
        """
        Args:
            vocab_size: Vocabulary size (21)
            embedding_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout_rate: Dropout probability
            output_dim: Number of output dimensions
            bidirectional: Whether to use bidirectional LSTM
        """
        super(EmbeddingLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional


        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)


        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )


        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim


        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights (embedding already initialized in __init__)."""

        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length) - token indices
        """

        x = self.embedding(x)


        lstm_out, (hidden, cell) = self.lstm(x)


        if self.bidirectional:

            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]


        return self.fc(hidden)


class EmbeddingGRU(nn.Module):
    """
    GRU model with embedding layer.
    """

    def __init__(self, vocab_size: int = 21, embedding_dim: int = 128,
                 hidden_dim: int = 128, num_layers: int = 2,
                 dropout_rate: float = 0.3, output_dim: int = 1,
                 bidirectional: bool = True):
        """
        Args:
            vocab_size: Vocabulary size (21)
            embedding_dim: Embedding dimension
            hidden_dim: GRU hidden dimension
            num_layers: Number of GRU layers
            dropout_rate: Dropout probability
            output_dim: Number of output dimensions
            bidirectional: Whether to use bidirectional GRU
        """
        super(EmbeddingGRU, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional


        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)


        self.gru = nn.GRU(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )


        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim


        self.fc = nn.Sequential(
            nn.Linear(gru_output_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights (embedding already initialized in __init__)."""

        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length) - token indices
        """

        x = self.embedding(x)


        gru_out, hidden = self.gru(x)


        if self.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]


        return self.fc(hidden)


class EmbeddingRNN(nn.Module):
    """
    RNN model with embedding layer.
    """

    def __init__(self, vocab_size: int = 21, embedding_dim: int = 128,
                 hidden_dim: int = 128, num_layers: int = 2,
                 dropout_rate: float = 0.3, output_dim: int = 1):
        """
        Args:
            vocab_size: Vocabulary size (21)
            embedding_dim: Embedding dimension
            hidden_dim: RNN hidden dimension
            num_layers: Number of RNN layers
            dropout_rate: Dropout probability
            output_dim: Number of output dimensions
        """
        super(EmbeddingRNN, self).__init__()


        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)


        self.rnn = nn.RNN(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0
        )


        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights (embedding already initialized in __init__)."""

        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length) - token indices
        """

        x = self.embedding(x)


        rnn_out, hidden = self.rnn(x)


        hidden = hidden[-1]


        return self.fc(hidden)


class EmbeddingCNNBiLSTMHybrid(nn.Module):
    """
    Hybrid model combining CNN (sequence patterns) + BiLSTM (sequence dependencies).
    Uses embedding layer instead of one-hot encoding.
    """

    def __init__(self, vocab_size: int = 21, embedding_dim: int = 128,
                 conv_channels: list = [64, 128], kernel_sizes: list = [3, 5],
                 lstm_hidden_dim: int = 128, lstm_num_layers: int = 2,
                 dropout_rate: float = 0.3, output_dim: int = 1):
        """
        Args:
            vocab_size: Vocabulary size (21: 20 AAs + padding)
            embedding_dim: Embedding dimension
            conv_channels: List of CNN channel sizes
            kernel_sizes: List of CNN kernel sizes
            lstm_hidden_dim: BiLSTM hidden dimension
            lstm_num_layers: Number of BiLSTM layers
            dropout_rate: Dropout probability
            output_dim: Number of output dimensions
        """
        super(EmbeddingCNNBiLSTMHybrid, self).__init__()


        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)


        self.cnn_layers = nn.ModuleList()
        prev_channels = embedding_dim
        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            conv = nn.Sequential(
                nn.Conv1d(prev_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
            self.cnn_layers.append(conv)
            prev_channels = out_channels

        self.cnn_pool = nn.AdaptiveAvgPool1d(1)
        self.cnn_fc = nn.Sequential(
            nn.Linear(prev_channels, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )


        self.bilstm = nn.LSTM(
            embedding_dim, lstm_hidden_dim, lstm_num_layers,
            batch_first=True, bidirectional=True, dropout=dropout_rate if lstm_num_layers > 1 else 0
        )
        self.lstm_fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )


        self.feature_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ) for _ in range(2)
        ])


        self.combined_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, output_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        for name, param in self.bilstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length) - token indices
        """

        x_emb = self.embedding(x)


        cnn_input = x_emb.transpose(1, 2)
        for conv in self.cnn_layers:
            cnn_input = conv(cnn_input)
        cnn_out = self.cnn_pool(cnn_input).squeeze(-1)
        cnn_out = self.cnn_fc(cnn_out)


        lstm_out, (hidden, cell) = self.bilstm(x_emb)

        lstm_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        lstm_out = self.lstm_fc(lstm_hidden)


        combined = torch.cat([cnn_out, lstm_out], dim=1)


        fused = combined
        for fusion_layer in self.feature_fusion:
            fused_output = fusion_layer(combined)
            fused = fused + fused_output


        return self.combined_mlp(fused)


class EmbeddingPhysioChemCNNBiLSTMHybrid(nn.Module):
    """
    Hybrid model combining embedding CNN-BiLSTM sequence learning with a
    physicochemical feature branch.
    """

    def __init__(self, vocab_size: int = 21, embedding_dim: int = 128,
                 conv_channels: list = [64, 128], kernel_sizes: list = [3, 5],
                 lstm_hidden_dim: int = 128, lstm_num_layers: int = 2,
                 physchem_input_dim: int = 53, physchem_hidden_dims: list = [128, 64],
                 dropout_rate: float = 0.3, output_dim: int = 1):
        super(EmbeddingPhysioChemCNNBiLSTMHybrid, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

        self.cnn_layers = nn.ModuleList()
        prev_channels = embedding_dim
        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            conv = nn.Sequential(
                nn.Conv1d(prev_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
            self.cnn_layers.append(conv)
            prev_channels = out_channels

        self.cnn_pool = nn.AdaptiveAvgPool1d(1)
        self.cnn_fc = nn.Sequential(
            nn.Linear(prev_channels, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        self.bilstm = nn.LSTM(
            embedding_dim, lstm_hidden_dim, lstm_num_layers,
            batch_first=True, bidirectional=True, dropout=dropout_rate if lstm_num_layers > 1 else 0
        )
        self.lstm_fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        physchem_layers = []
        prev_dim = physchem_input_dim
        for hidden_dim in physchem_hidden_dims:
            physchem_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        self.physchem_mlp = nn.Sequential(*physchem_layers)

        self.feature_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(192, 192),
                nn.LayerNorm(192),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ) for _ in range(2)
        ])

        self.combined_mlp = nn.Sequential(
            nn.Linear(192, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(96, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, output_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        for name, param in self.bilstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, physchem_features):
        x_emb = self.embedding(input_ids)

        cnn_input = x_emb.transpose(1, 2)
        for conv in self.cnn_layers:
            cnn_input = conv(cnn_input)
        cnn_out = self.cnn_pool(cnn_input).squeeze(-1)
        cnn_out = self.cnn_fc(cnn_out)

        _, (hidden, _) = self.bilstm(x_emb)
        lstm_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        lstm_out = self.lstm_fc(lstm_hidden)

        physchem_out = self.physchem_mlp(physchem_features)

        combined = torch.cat([cnn_out, lstm_out, physchem_out], dim=1)

        fused = combined
        for fusion_layer in self.feature_fusion:
            fused_output = fusion_layer(combined)
            fused = fused + fused_output

        return self.combined_mlp(fused)


class EmbeddingSeqPhyschemHybrid(nn.Module):
    """
    Single-branch embedding sequence encoder (CNN, RNN, BiLSTM, or BiGRU) + physchem MLP.
    Fusion uses the same 64+64→128 pattern as PhysioChemHybridModel / cnn_bilstm_physchem head.
    """

    def __init__(
        self,
        seq_type: str,
        vocab_size: int = 21,
        embedding_dim: int = 128,
        conv_channels: list = None,
        kernel_sizes: list = None,
        hidden_dim: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        physchem_input_dim: int = 53,
        physchem_hidden_dims: list = None,
        dropout_rate: float = 0.3,
        output_dim: int = 1,
    ):
        super().__init__()
        seq_type = seq_type.lower()
        self.seq_type = seq_type
        if physchem_hidden_dims is None:
            physchem_hidden_dims = [128, 64]

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

        if seq_type == "cnn":
            if not conv_channels or not kernel_sizes:
                raise ValueError("seq_type cnn requires conv_channels and kernel_sizes")
            self.conv_layers = nn.ModuleList()
            prev_channels = embedding_dim
            for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
                conv = nn.Sequential(
                    nn.Conv1d(prev_channels, out_channels, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                )
                self.conv_layers.append(conv)
                prev_channels = out_channels
            self.cnn_pool = nn.AdaptiveAvgPool1d(1)
            self.seq_head = nn.Sequential(
                nn.Linear(prev_channels, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            )
            self.rnn = None
        elif seq_type == "rnn":
            self.conv_layers = None
            self.rnn = nn.RNN(
                embedding_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0,
            )
            self.seq_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            )
        elif seq_type == "lstm":
            self.conv_layers = None
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )
            lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
            self.seq_head = nn.Sequential(
                nn.Linear(lstm_out_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            )
        elif seq_type == "gru":
            self.conv_layers = None
            self.rnn = nn.GRU(
                embedding_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )
            gru_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
            self.seq_head = nn.Sequential(
                nn.Linear(gru_out_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            )
        else:
            raise ValueError(
                f"Unknown seq_type for EmbeddingSeqPhyschemHybrid: {seq_type}. "
                "Use cnn, rnn, lstm, or gru."
            )

        physchem_layers = []
        prev_dim = physchem_input_dim
        for hidden_dim_pc in physchem_hidden_dims:
            physchem_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim_pc),
                    nn.LayerNorm(hidden_dim_pc),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = hidden_dim_pc
        self.physchem_mlp = nn.Sequential(*physchem_layers)

        self.feature_fusion = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(128, 128),
                    nn.LayerNorm(128),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                )
                for _ in range(2)
            ]
        )

        self.combined_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, output_dim),
        )
        self._init_rnn_weights()

    def _init_rnn_weights(self):
        if self.rnn is not None:
            for name, param in self.rnn.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    param.data.fill_(0)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, physchem_features):
        x = self.embedding(input_ids)
        if self.seq_type == "cnn":
            h = x.transpose(1, 2)
            for conv in self.conv_layers:
                h = conv(h)
            h = self.cnn_pool(h).squeeze(-1)
            seq_out = self.seq_head(h)
        else:
            if isinstance(self.rnn, nn.LSTM):
                _, (hidden, _) = self.rnn(x)
                if self.rnn.bidirectional:
                    hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
                else:
                    hidden = hidden[-1]
            elif isinstance(self.rnn, nn.GRU):
                _, hidden = self.rnn(x)
                if self.rnn.bidirectional:
                    hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
                else:
                    hidden = hidden[-1]
            else:
                _, hidden = self.rnn(x)
                hidden = hidden[-1]
            seq_out = self.seq_head(hidden)

        physchem_out = self.physchem_mlp(physchem_features)
        combined = torch.cat([seq_out, physchem_out], dim=1)
        fused = combined
        for fusion_layer in self.feature_fusion:
            fused = fused + fusion_layer(combined)
        return self.combined_mlp(fused)


def get_embedding_model(model_type: str, vocab_size: int = 21, **kwargs):
    """
    Factory function to get embedding-based model by type.

    Args:
        model_type: Type of model ('cnn', 'rnn', 'lstm', 'gru')
        vocab_size: Vocabulary size (21: 20 AAs + padding)
        **kwargs: Model-specific arguments

    Returns:
        Model instance
    """
    model_type = model_type.lower()

    if model_type == 'cnn':
        return EmbeddingCNN(vocab_size=vocab_size, **kwargs)
    elif model_type == 'rnn':
        return EmbeddingRNN(vocab_size=vocab_size, **kwargs)
    elif model_type == 'lstm' or model_type == 'bilstm':
        if 'bidirectional' not in kwargs:
            kwargs['bidirectional'] = True
        return EmbeddingLSTM(vocab_size=vocab_size, **kwargs)
    elif model_type == 'gru' or model_type == 'bigru':
        if 'bidirectional' not in kwargs:
            kwargs['bidirectional'] = True
        return EmbeddingGRU(vocab_size=vocab_size, **kwargs)
    elif model_type == 'cnn_bilstm':
        return EmbeddingCNNBiLSTMHybrid(vocab_size=vocab_size, **kwargs)
    elif model_type == 'cnn_bilstm_physchem':
        return EmbeddingPhysioChemCNNBiLSTMHybrid(vocab_size=vocab_size, **kwargs)
    elif model_type == 'cnn_physchem':
        return EmbeddingSeqPhyschemHybrid(seq_type='cnn', vocab_size=vocab_size, **kwargs)
    elif model_type == 'rnn_physchem':
        return EmbeddingSeqPhyschemHybrid(
            seq_type='rnn', vocab_size=vocab_size, bidirectional=False, **kwargs
        )
    elif model_type == 'bilstm_physchem':
        return EmbeddingSeqPhyschemHybrid(
            seq_type='lstm', vocab_size=vocab_size, bidirectional=True, **kwargs
        )
    elif model_type == 'bigru_physchem':
        return EmbeddingSeqPhyschemHybrid(
            seq_type='gru', vocab_size=vocab_size, bidirectional=True, **kwargs
        )
    else:
        raise ValueError(
            f"Unknown embedding model type: {model_type}. "
            "Supported: cnn, rnn, lstm, bilstm, gru, bigru, cnn_bilstm, cnn_bilstm_physchem, "
            "cnn_physchem, rnn_physchem, bilstm_physchem, bigru_physchem"
        )

