import torch as t
import torch.nn as nn
import torch.nn.functional as F
from utils.functional import parameters_allocation_check
# from torchqrnn import QRNN
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
        nn.init.xavier_normal(self.fc.weight)

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

class AttnDecoder(nn.Module):
    def __init__(self, params):
        super(AttnDecoder, self).__init__()

        self.params = params

        self.attention = Attention(self.params.decoder_rnn_size, self.params.decoder_rnn_size)

        self.rnn = nn.LSTM(input_size=self.params.decoder_rnn_size*2+self.params.latent_variable_size + self.params.word_embed_size,
                           hidden_size=self.params.decoder_rnn_size,
                           num_layers=self.params.decoder_num_layers,
                           batch_first=False,
                           bidirectional=False)

        self.fc = nn.Linear(self.params.decoder_rnn_size, self.params.word_vocab_size)
        self.fc_out = nn.Linear(self.params.decoder_rnn_size*3+self.params.latent_variable_size + self.params.word_embed_size, self.params.word_vocab_size)

    def only_decoder_beam(self, mask, decoder_input, z, drop_prob, encoder_outputs, initial_state=None):

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'
        [beam_batch_size, _, _] = decoder_input.size()
        '''
            decoder rnn is conditioned on context via additional bias = W_cond * z to every input token
        '''
        print(decoder_input.size())
        decoder_input = F.dropout(decoder_input, drop_prob)
        z = z.unsqueeze(0)
        z = t.cat([z] * beam_batch_size, 0)
        decoder_input = t.cat([decoder_input, z], 2)

        rnn_out, final_state = self.rnn(decoder_input, initial_state)

        return rnn_out, final_state
                    
    def forward(self, mask, decoder_input, z, drop_prob, encoder_outputs, initial_state=None):
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

    def forward(self, mask, input, z, drop_prob, encoder_outputs, hidden):
             
        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        #mask = [batch size, src len]
        encoder_outputs = encoder_outputs.transpose(0, 1)
        temp_h = hidden[0][-1]
        
        embedded = input.unsqueeze(0)
                        
        #embedded = [1, batch size, emb dim]
        
        a = self.attention(temp_h, encoder_outputs, mask)

                
        #a = [batch size, src len]
        
        a = a.unsqueeze(1)
        
        #a = [batch size, 1, src len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        weighted = t.bmm(a, encoder_outputs)
        
        #weighted = [batch size, 1, enc hid dim * 2]
        
        weighted = weighted.permute(1, 0, 2)
        
        #weighted = [1, batch size, enc hid dim * 2]
        
        rnn_input = t.cat((embedded, weighted), dim = 2)
        
        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
            
        output, hidden = self.rnn(rnn_input, hidden)
        
        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        # assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(t.cat((output, weighted, embedded), dim = 1))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, a.squeeze(1)

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs, mask):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
  
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        energy = t.tanh(self.attn(t.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        
        #attention = [batch size, src len]
        
        attention = attention.masked_fill(mask == 0, -1e10)
        
        return F.softmax(attention, dim = 1)

# class DecoderQRNN(nn.Module):
#     def __init__(self, params):
#         super(DecoderQRNN, self).__init__()

#         self.params = params
        
#         self.rnn = QRNN(self.params.latent_variable_size + self.params.word_embed_size,
#                         self.params.decoder_rnn_size,
#                         num_layers=1,
#                         dropout=0.4)

#         self.fc = nn.Linear(self.params.decoder_rnn_size, self.params.word_vocab_size)

#     def only_decoder_beam(self, decoder_input, z, drop_prob, encoder_outputs, initial_state=None):

#         assert parameters_allocation_check(self), \
#             'Invalid CUDA options. Parameters should be allocated in the same memory'
#         [beam_batch_size, _, _] = decoder_input.size()
#         '''
#             decoder rnn is conditioned on context via additional bias = W_cond * z to every input token
#         '''
#         decoder_input = F.dropout(decoder_input, drop_prob)
#         z = z.unsqueeze(0)
#         z = t.cat([z] * beam_batch_size, 0)
#         decoder_input = t.cat([decoder_input, z], 2)
#         decoder_input = t.cat([decoder_input, z], 2)

#         rnn_out, final_state = self.fix_for_qrnn(decoder_input, initial_state)

#         return rnn_out, final_state

#     def fix_for_qrnn(self, decoder_input, initial_state):
#         decoder_input = decoder_input.transpose(0,1)
#         print(initial_state[0].size())
#         rnn_out, final_state = self.rnn(decoder_input, initial_state)
#         decoder_input = rnn_out.transpose(0,1)
        
#         return rnn_out, final_state


#     def forward(self, decoder_input, z, drop_prob, encoder_outputs, initial_state=None):
#         """
#         :param decoder_input: tensor with shape of [batch_size, seq_len, embed_size]
#         :param z: sequence context with shape of [batch_size, latent_variable_size]
#         :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout
#         :param initial_state: initial state of decoder rnn

#         :return: unnormalized logits of sentense words distribution probabilities
#                     with shape of [batch_size, seq_len, word_vocab_size]
#                  final rnn state with shape of [num_layers, batch_size, decoder_rnn_size]
#         """

#         assert parameters_allocation_check(self), \
#             'Invalid CUDA options. Parameters should be allocated in the same memory'

#         [batch_size, seq_len, _] = decoder_input.size()

#         '''
#             decoder rnn is conditioned on context via additional bias = W_cond * z to every input token
#         '''
#         decoder_input = F.dropout(decoder_input, drop_prob)

#         z = t.cat([z] * seq_len, 1).view(batch_size, seq_len, self.params.latent_variable_size)
#         decoder_input = t.cat([decoder_input, z], 2)
        
#         rnn_out, final_state = self.fix_for_qrnn(decoder_input, initial_state)
        
#         rnn_out = rnn_out.contiguous().view(-1, self.params.decoder_rnn_size)

#         result = self.fc(rnn_out)
#         result = result.view(batch_size, seq_len, self.params.word_vocab_size)

#         return result, final_state


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
                           num_layers=self.params.decoder_num_layers,
                           batch_first=True,
                           bidirectional=False)

        self.fc = nn.Linear(self.params.decoder_rnn_size, self.params.word_vocab_size)
        nn.init.xavier_normal(self.fc.weight)

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
        rnn_out, final_state = self.batch_residual_unrolling(decoder_input,  initial_state, False)

        return rnn_out, final_state

    def batch_residual_unrolling(self, decoder_input,  initial_state, x=None):
        [batch_size, seq_len, _] = decoder_input.size()
        output_words = t.empty(decoder_input.size(), requires_grad=True).cuda()
        for word_id in range(seq_len):
            input = decoder_input[:,word_id,:].unsqueeze(1)
            rnn_out, (h_n, c_n) = self.rnn_1(input, initial_state)
            h_n_new = t.add(input, h_n[-1,:,:].unsqueeze(1)).cuda()
            rnn_out, initial_state = self.rnn_2(h_n_new)
            output_words[:, word_id] = rnn_out.squeeze(1)

        return output_words, initial_state

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
                words = decoder_input[sentence_id, word_id, :].view(1, 1, -1)
                rnn_out, (h_n, c_n) = self.rnn_1(word, state)
                h_n_new = t.add(words, h_n[-1,:,:].unsqueeze(1))
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

        rnn_out, final_state = self.batch_residual_unrolling(decoder_input, initial_state)
        rnn_out = rnn_out.contiguous().view(-1, self.params.decoder_rnn_size)

        result = self.fc(rnn_out)
        result = result.view(batch_size, seq_len, self.params.word_vocab_size)

        return result, final_state





