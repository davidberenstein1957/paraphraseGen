import torch as t
import torch.nn as nn
import torch.nn.functional as F
from utils.functional import parameters_allocation_check
from torchqrnn import QRNN
import warnings
warnings.filterwarnings("ignore")

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()

        self.params = params

        self.rnn = nn.LSTM(input_size=self.params.latent_variable_size + self.params.word_embed_size,
                           hidden_size=self.params.decoder_rnn_size,
                           num_layers=self.params.decoder_num_layers,
                           batch_first=True,
                           bidirectional=False)

        self.fc = nn.Linear(self.params.decoder_rnn_size, self.params.word_vocab_size)

    def only_decoder_beam(self, decoder_input, z, drop_prob, encoder_outputs, initial_state=None):

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'
        [beam_batch_size, _, _] = decoder_input.size()
        '''
            decoder rnn is conditioned on context via additional bias = W_cond * z to every input token
        '''
        decoder_input = F.dropout(decoder_input, drop_prob)
        z = z.unsqueeze(0)
        z = t.cat([z] * beam_batch_size, 0)
        decoder_input = t.cat([decoder_input, z], 2)
        decoder_input = t.cat([decoder_input, z], 2)

        rnn_out, final_state = self.rnn(decoder_input, initial_state)

        return rnn_out, final_state

    def forward(self, decoder_input, z, drop_prob, encoder_outputs, initial_state=None):
        """
        :param decoder_input: tensor with shape of [batch_size, seq_len, embed_size]
        :param z: sequence context with shape of [batch_size, latent_variable_size]
        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout
        :param initial_state: initial state of decoder rnn

        :return: unnormalized logits of sentense words distribution probabilities
                    with shape of [batch_size, seq_len, word_vocab_size]
                 final rnn state with shape of [num_layers, batch_size, decoder_rnn_size]
        """

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'

        [batch_size, seq_len, _] = decoder_input.size()

        '''
            decoder rnn is conditioned on context via additional bias = W_cond * z to every input token
        '''
        decoder_input = F.dropout(decoder_input, drop_prob)

        z = t.cat([z] * seq_len, 1).view(batch_size, seq_len, self.params.latent_variable_size)
        decoder_input = t.cat([decoder_input, z], 2)
        
        rnn_out, final_state = self.rnn(decoder_input, initial_state)
        
        rnn_out = rnn_out.contiguous().view(-1, self.params.decoder_rnn_size)

        result = self.fc(rnn_out)
        result = result.view(batch_size, seq_len, self.params.word_vocab_size)

        return result, final_state

class DecoderQRNN(nn.Module):
    def __init__(self, params):
        super(DecoderQRNN, self).__init__()

        self.params = params
        
        self.rnn = QRNN(self.params.latent_variable_size + self.params.word_embed_size,
                        self.params.decoder_rnn_size,
                        num_layers=1,
                        dropout=0.4)

        self.fc = nn.Linear(self.params.decoder_rnn_size, self.params.word_vocab_size)

    def only_decoder_beam(self, decoder_input, z, drop_prob, encoder_outputs, initial_state=None):

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'
        [beam_batch_size, _, _] = decoder_input.size()
        '''
            decoder rnn is conditioned on context via additional bias = W_cond * z to every input token
        '''
        decoder_input = F.dropout(decoder_input, drop_prob)
        z = z.unsqueeze(0)
        z = t.cat([z] * beam_batch_size, 0)
        decoder_input = t.cat([decoder_input, z], 2)
        decoder_input = t.cat([decoder_input, z], 2)

        rnn_out, final_state = self.fix_for_qrnn(decoder_input, initial_state)

        return rnn_out, final_state

    def fix_for_qrnn(self, decoder_input, initial_state):
        decoder_input = decoder_input.transpose(0,1)
        print(initial_state[0].size())
        rnn_out, final_state = self.rnn(decoder_input, initial_state)
        decoder_input = rnn_out.transpose(0,1)
        
        return rnn_out, final_state


    def forward(self, decoder_input, z, drop_prob, encoder_outputs, initial_state=None):
        """
        :param decoder_input: tensor with shape of [batch_size, seq_len, embed_size]
        :param z: sequence context with shape of [batch_size, latent_variable_size]
        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout
        :param initial_state: initial state of decoder rnn

        :return: unnormalized logits of sentense words distribution probabilities
                    with shape of [batch_size, seq_len, word_vocab_size]
                 final rnn state with shape of [num_layers, batch_size, decoder_rnn_size]
        """

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'

        [batch_size, seq_len, _] = decoder_input.size()

        '''
            decoder rnn is conditioned on context via additional bias = W_cond * z to every input token
        '''
        decoder_input = F.dropout(decoder_input, drop_prob)

        z = t.cat([z] * seq_len, 1).view(batch_size, seq_len, self.params.latent_variable_size)
        decoder_input = t.cat([decoder_input, z], 2)
        
        rnn_out, final_state = self.fix_for_qrnn(decoder_input, initial_state)
        
        rnn_out = rnn_out.contiguous().view(-1, self.params.decoder_rnn_size)

        result = self.fc(rnn_out)
        result = result.view(batch_size, seq_len, self.params.word_vocab_size)

        return result, final_state


class ResidualDecoder(nn.Module):
    def __init__(self, params):
        super(ResidualDecoder, self).__init__()

        self.params = params

        self.rnn_1 = nn.LSTM(input_size=self.params.latent_variable_size + self.params.word_embed_size,
                           hidden_size=self.params.decoder_rnn_size,
                           num_layers=self.params.decoder_num_layers,
                           batch_first=True,
                           bidirectional=False)

        self.rnn_2 = nn.LSTM(input_size=self.params.decoder_rnn_size,
                           hidden_size=self.params.decoder_rnn_size,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=False)

        self.fc = nn.Linear(self.params.decoder_rnn_size, self.params.word_vocab_size)

    def only_decoder_beam(self, decoder_input, z, drop_prob, encoder_outputs, initial_state=None):

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'
        [beam_batch_size, _, _] = decoder_input.size()
        '''
            decoder rnn is conditioned on context via additional bias = W_cond * z to every input token
        '''
        decoder_input = F.dropout(decoder_input, drop_prob)
        z = z.unsqueeze(0)
        z = t.cat([z] * beam_batch_size, 0)
        decoder_input = t.cat([decoder_input, z], 2)
        rnn_out, final_state = self.residual_unrolling_in_the_deep(decoder_input,  initial_state, False)

        return rnn_out, final_state

    # https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
    def residual_unrolling_in_the_deep(self, decoder_input,  initial_state, step=True):
        [batch_size, seq_len, _] = decoder_input.size()
        output_words = t.empty(decoder_input.size(), requires_grad=True).cuda()
        h_0_states = t.empty(initial_state[0].size(), requires_grad=True).cuda()
        c_0_states = t.empty(initial_state[1].size(), requires_grad=True).cuda()
        h_0, c_0 = initial_state[0], initial_state[1]
        for sentence_id in range(batch_size):
            state = (h_0[:, sentence_id, :].unsqueeze(1).contiguous(), c_0[:, sentence_id, :].unsqueeze(1).contiguous())
            for word_id in range(seq_len):
                word = decoder_input[sentence_id, word_id, :].view(1, 1, -1)
                rnn_out, (h_n, c_n) = self.rnn_1(word, state)
                h_n_new = t.add(word, h_n[-1,:,:].unsqueeze(1))
                rnn_out, final_state = self.rnn_2(h_n_new.cuda())
                output_words[sentence_id, word_id] = rnn_out.cuda()
        
            h_0_states[0, sentence_id] = h_n_new[-1,:,:].unsqueeze(1).cuda()
            c_0_states[0, sentence_id] = c_n[-1,:,:].unsqueeze(1).cuda()
            h_0_states[1, sentence_id] = final_state[0].cuda()
            c_0_states[1, sentence_id] = final_state[1].cuda()
        rnn_out = output_words
        final_state = (h_0_states, c_0_states)

        return rnn_out, initial_state

    
    def forward(self, decoder_input, z, drop_prob, encoder_outputs, initial_state=None):
        """
        :param decoder_input: tensor with shape of [batch_size, seq_len, embed_size]
        :param z: sequence context with shape of [batch_size, latent_variable_size]
        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout
        :param initial_state: initial state of decoder rnn

        :return: unnormalized logits of sentense words distribution probabilities
                    with shape of [batch_size, seq_len, word_vocab_size]
                 final rnn state with shape of [num_layers, batch_size, decoder_rnn_size]
        """

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'

        [batch_size, seq_len, _] = decoder_input.size()

        '''
            decoder rnn is conditioned on context via additional bias = W_cond * z to every input token
        '''
        decoder_input = F.dropout(decoder_input, drop_prob)

        z = t.cat([z] * seq_len, 1).view(batch_size, seq_len, self.params.latent_variable_size)
        decoder_input = t.cat([decoder_input, z], 2)

        rnn_out, final_state = self.residual_unrolling_in_the_deep(decoder_input, initial_state)
        rnn_out = rnn_out.contiguous().view(-1, self.params.decoder_rnn_size)

        result = self.fc(rnn_out)
        result = result.view(batch_size, seq_len, self.params.word_vocab_size)

        return result, final_state

class AttnDecoder(nn.Module):
    def __init__(self, params):
        super(AttnDecoder, self).__init__()
        # self.attn = nn.Linear(hidden_size, hidden_size)
        self.params = params
        self.attn = nn.Linear(self.params.decoder_rnn_size, self.params.decoder_rnn_size)

        # self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout_p)
        self.rnn = nn.LSTM(input_size=self.params.latent_variable_size + self.params.word_embed_size,
                           hidden_size=self.params.decoder_rnn_size,
                           num_layers=self.params.decoder_num_layers,
                           batch_first=True,
                           bidirectional=False)

        # self.attn_combine = nn.Linear(hidden_size + embedding_size, hidden_size)
        self.attn_combine = nn.Linear(self.params.decoder_rnn_size + self.params.latent_variable_size + self.params.word_embed_size, self.params.decoder_rnn_size)
        # self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.concat = nn.Linear(self.params.decoder_rnn_size * 2, self.params.decoder_rnn_size)
        # self.out = nn.Linear(hidden_size, output_size)
        self.out = nn.Linear(self.params.decoder_rnn_size, self.params.word_vocab_size)        
        
    def score(self, rnn_output, e_outputs):
        energy = self.attn(e_outputs)
        energy = energy.transpose(0,1).transpose(1,2)
        rnn_output = rnn_output.transpose(0,1)
        energy = (rnn_output @ energy).squeeze(1)
        
        return F.softmax(energy, dim=1).unsqueeze(1)
    
    # def forward(self, input_seq, hidden, e_outputs, batch_size):
    def forward(self, decoder_input, z, drop_prob, encoder_outputs, initial_state=None):
        
        [batch_size, seq_len, _] = decoder_input.size()

        '''
            decoder rnn is conditioned on context via additional bias = W_cond * z to every input token
        '''
        decoder_input = F.dropout(decoder_input, drop_prob)

        z = t.cat([z] * seq_len, 1).view(batch_size, seq_len, self.params.latent_variable_size)
        decoder_input = t.cat([decoder_input, z], 2)

        rnn_output, hidden = self.rnn(decoder_input, initial_state)

        attn_weights = self.score(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        concat = torch.cat((rnn_output.squeeze(0), context.squeeze(1)), 1)
        concat = F.tanh(self.concat(concat))
        output = self.out(concat)
        
        return output, hidden

