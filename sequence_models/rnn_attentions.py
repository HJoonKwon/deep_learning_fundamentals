import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class EncoderRNN(nn.Module):

    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # initialize GRU(Gated Recurrent Unit)
        self.gru = nn.GRU(input_size=hidden_size,  # input_size = embedding size = hidden size
                          hidden_size=hidden_size,
                          n_layers=n_layers,
                          dropout=(0 if n_layers == 1 else dropout),
                          bidirectional=True)

    def forward(self, input_seq: torch.Tensor, input_lengths: list[int], hidden: Optional[torch.Tensor] = None):
        """Forward pass of encoder of RNN module

        Args:
            input_seq (torch.Tensor): batch of input sequences, shape=(L, N)
            input_lengths (list[int]): list of sentence lengths corresponding to each sentence in the batch
            hidden (Optional[torch.Tensor], optional): hidden state. shape=(n_layers*D, N, H_in).Defaults to None.
        """

        # 1. convert word indices to embeddings(shape = (L, N, H_in))
        embedded = self.embedding(input_seq)

        # 2. pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)

        # 3. Forward pass through GRU
        output, hidden = self.gru(packed, hidden)

        # 4. unpack padding
        output, _ = nn.utils.rnn.pack_padded_sequence(output)

        # 5. sum bidirectional gru outputs
        output = output[:, :, :self.hidden_size] + \
            output[:, :, self.hidden_size:]

        return output, hidden


class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """ Attention module for Decoder of GRU RNN model

        Args:
            hidden (torch.Tensor): hidden state from the corresponding decoder block,
                                    shape = (1, batch_size, hidden_size)
            encoder_outputs (torch.Tensor): concatenated encoder outputs
                                    shape = (max_length, batch_size, hidden_size)

        Returns:
            torch.Tensor: Attention weights normalized by softmax
        """

        # 1. Dot product of hidden and encoder_outputs on hidden_size dimension
        attention_weights = torch.einsum(
            't b h, l b h -> t l b', hidden, encoder_outputs).squeeze(0)  # (max_length, batch)

        # 2. Transpose
        attention_weights = attention_weights.t()  # (batch, max_length)

        # 3. Apply softmax to normalize the attention weights
        return F.softmax(attention_weights, dim=1).unsqueeze(1)


if __name__ == "__main__":
    batch = 10
    hidden_size = 20
    length = 5
    hidden = torch.randn(1, batch, hidden_size)
    encoder_outputs = torch.randn(length, batch, hidden_size)
    attention_module = Attention('self', hidden_size=hidden_size)
    attention_weights = attention_module(hidden, encoder_outputs)
    print(attention_weights.shape)
