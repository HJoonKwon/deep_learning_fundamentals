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


class AttentionDecoderRNN(nn.Module):
    def __init__(self, attention_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(AttentionDecoderRNN, self).__init__()
        self.attention_model = attention_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # 1. Define Layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers==1 else dropout))
        self.concat_layer = nn.Linear(hidden_size*2, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.attention = Attention(self.attention_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        """Forwarod method for the Decoder with Attention module

        Args:
            input_step (torch.Tensor): (1, N, H_in)
            last_hidden (torch.Tensor): (D*n_layers, N, H_in)
            encoder_outputs (torch.Tensor): (L, N, H_out)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: return output and hidden
        """

        # 1. get input word's embedding at the current step
        embedded = self.embedding(input_step)

        # 2. forward through the dropout layer
        embedded = self.embedding_dropout(embedded)

        # 3. forward through the GRU and get gru output and hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # 4. calculate attention weights for hidden state / encoder outputs
        attention_weights = self.attention(hidden, encoder_outputs)

        # 5. calculate context vector using attention weights and encoder outputs
        # (batch, 1, length) x (max_length, batch, hidden) <- matmul in length dimension
        context = torch.einsum('b t l, l b h -> t b h', attention_weights, encoder_outputs)

        # 6. concatenate the gru output and context vector, and forward through the linear layers
        rnn_output = rnn_output.squeeze(0) # (b, h)
        context = context.squeeze(0) # (b, h)

        concat_input = torch.cat((rnn_output, context), dim=1) # (b, 2h)
        concat_output = torch.tanh(self.concat_layer(concat_input)) # (b, h)

        output = self.output_layer(concat_output) # (b, out)

        # 7. apply softmax to calculate probability distribution of vocab
        output = F.softmax(output, dim=1)

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
