import torch as t
import torch.nn as nn
import torch.nn.functional as F
from selfModules.highway import Highway
from utils.functional import parameters_allocation_check

# from torchqrnn import QRNN


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()

        self.params = params

        self.hw1 = Highway(self.params.sum_depth + self.params.word_embed_size, 2, F.relu)
        self.bi = True
        self.rnn = nn.LSTM(input_size=self.params.word_embed_size + self.params.sum_depth,
                           hidden_size=self.params.encoder_rnn_size,
                           num_layers=self.params.encoder_num_layers,
                           batch_first=True,
                           bidirectional=self.bi)

    def forward(self, input, State):
        """
        :param input: [batch_size, seq_len, embed_size] tensor
        :return: context of input sentenses with shape of [batch_size, latent_variable_size]
        """

        # print "Three"
        [batch_size, seq_len, embed_size] = input.size()
        # input shape   32    ,    26     ,    825

        input = input.view(-1, embed_size)
        # input shape   832(=32*26),825

        input = self.hw1(input)
        # input shape 832(=32*26),825

        input = input.view(batch_size, seq_len, embed_size)
        # input shape 32    ,    26     ,    825

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'

        ''' Unfold rnn with zero initial state and get its final state from the last layer
        '''
        encoder_outputs, (h_0, final_state) = self.rnn(input, State)
        """Inputs: input, (h_0, c_0)
        - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
          of the input sequence.
          The input can also be a packed variable length sequence.
          Outputs: output, (h_n, c_n)
        - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
          containing the output features `(h_t)` from the last layer of the LSTM,
          for each `t`. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
          given as the input, the output will also be a packed sequence.
        """
        c_0 = final_state

        if self.bi:
            size = 2
        else:
            size = 1

        final_state = final_state.view(self.params.encoder_num_layers, size, batch_size, self.params.encoder_rnn_size)
        final_state = final_state[-1]
        h_1, h_2 = final_state[0], final_state[1]
        final_state = t.cat([h_1, h_2], 1)

        return encoder_outputs, final_state, h_0, c_0


# https://arxiv.org/pdf/1911.05343.pdf
# https://github.com/ruizheliUOA/HR-VAE/blob/master/HR-VAE.py
class EncoderHR(nn.Module):
    def __init__(self, params):
        super(EncoderHR, self).__init__()

        self.params = params

        self.hw1 = Highway(self.params.sum_depth + self.params.word_embed_size, 2, F.relu)
        self.bi = True
        self.rnn = nn.LSTM(input_size=self.params.word_embed_size + self.params.sum_depth,
                           hidden_size=self.params.encoder_rnn_size,
                           num_layers=self.params.encoder_num_layers,
                           batch_first=True,
                           bidirectional=self.bi)

        self.layer_dim = (self.params.encoder_num_layers*2)*self.params.encoder_rnn_size

        self.linear_mu = nn.Linear(self.layer_dim*2, self.layer_dim*2)
        self.linear_var = nn.Linear(self.layer_dim*2, self.layer_dim*2)

    

    def forward(self, input, State):
        """
        :param input: [batch_size, seq_len, embed_size] tensor
        :return: context of input sentenses with shape of [batch_size, latent_variable_size]
        """

        # print "Three"
        [batch_size, seq_len, embed_size] = input.size()
        # input shape   32    ,    26     ,    825

        input = input.view(-1, embed_size)
        # input shape   832(=32*26),825

        input = self.hw1(input)
        # input shape 832(=32*26),825

        input = input.view(batch_size, seq_len, embed_size)
        # input shape 32    ,    26     ,    825

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'

        ''' Unfold rnn with zero initial state and get its final state from the last layer
        '''

        context_ = []
        
        for word_id in range(seq_len):
            encoder_outputs, (h_0, final_state) = self.rnn(input[:,word_id].unsqueeze(1), State)
            """Inputs: input, (h_0, c_0)
            - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
            of the input sequence.
            The input can also be a packed variable length sequence.
            Outputs: output, (h_n, c_n)
            - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
            containing the output features `(h_t)` from the last layer of the LSTM,
            for each `t`. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
            given as the input, the output will also be a packed sequence.
            """
            State = (h_0, final_state)

            c_0 = final_state
            
            final_state = final_state.view(self.params.encoder_num_layers, 2, batch_size, self.params.encoder_rnn_size)
            final_state = final_state[-1]
            h_1, h_2 = final_state[0], final_state[1]
            final_state = t.cat([h_1, h_2], 1)

            context_.append(final_state)

        return encoder_outputs, final_state, h_0, c_0, context_
