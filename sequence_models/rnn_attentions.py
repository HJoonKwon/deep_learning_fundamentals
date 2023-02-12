import torch
import torch.nn as nn
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
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]

        return output, hidden


if __name__ == "__main__":
    print('')

