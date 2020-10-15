import torch as t
import torch.nn as nn
import torch.nn.functional as F
from utils.functional import parameters_allocation_check
import warnings
from torch.autograd import Variable

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

class DecoderResidual(nn.Module):
    def __init__(self, params):
        super(DecoderResidual, self).__init__()

        self.params = params

        self.rnn_1 = nn.LSTM(input_size=self.params.latent_variable_size + self.params.word_embed_size,
                           hidden_size=self.params.decoder_rnn_size,
                           num_layers=self.params.decoder_num_layers,
                           batch_first=True,
                           bidirectional=False,
                           dropout=0.5)

        self.rnn_2 = nn.LSTM(input_size=self.params.decoder_rnn_size,
                           hidden_size=self.params.decoder_rnn_size,
                           num_layers=self.params.decoder_num_layers,
                           batch_first=True,
                           bidirectional=False,
                           dropout=0.5)
        
        self.rnn_3 = nn.LSTM(input_size=self.params.decoder_rnn_size,
                           hidden_size=self.params.decoder_rnn_size,
                           num_layers=self.params.decoder_num_layers,
                           batch_first=True,
                           bidirectional=False)

        # self.rnn_4 = nn.LSTM(input_size=self.params.decoder_rnn_size,
        #                    hidden_size=self.params.decoder_rnn_size,
        #                    num_layers=self.params.decoder_num_layers,
        #                    batch_first=True,
        #                    bidirectional=False)


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
        rnn_out, final_state = self.batch_unrolling(decoder_input,  initial_state, False)

        return rnn_out, final_state

    def batch_unrolling(self, decoder_input,  initial_state, x=None):
        rnn_out_1, _ = self.rnn_1(decoder_input, initial_state)
        rnn_out_2, _ = self.rnn_2(rnn_out_1, initial_state)    
        rnn_out = t.add(rnn_out_2, rnn_out_1)
        rnn_out_3, final_state = self.rnn_3(rnn_out, initial_state)    
        # rnn_out = t.add(rnn_out_3, rnn_out)
        # rnn_out_4, final_state = self.rnn_4(rnn_out, initial_state)  
        # rnn_out = t.add(rnn_out_4, rnn_out)

        return rnn_out_3, final_state
    
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

        rnn_out, final_state = self.batch_unrolling(decoder_input, initial_state)
        rnn_out = rnn_out.contiguous().view(-1, self.params.decoder_rnn_size)

        result = self.fc(rnn_out)
        result = result.view(batch_size, seq_len, self.params.word_vocab_size)

        return result, final_state

class DecoderAttention(nn.Module):
    def __init__(self, params):
        super(DecoderAttention, self).__init__()

        self.params = params
        self.input_size = self.params.latent_variable_size + self.params.word_embed_size
        self.rnn = nn.LSTM(input_size=self.input_size,
                           hidden_size=self.params.decoder_rnn_size,
                           num_layers=self.params.decoder_num_layers,
                           batch_first=True,
                           bidirectional=False)

        self.hidden_size = self.params.decoder_rnn_size

        self.attention = Attention(self.params.decoder_rnn_size, 'general')
        self.fc_enc = nn.Linear(self.input_size*2, self.input_size)
        self.fc_out = nn.Linear(self.input_size*2, self.params.decoder_rnn_size)
        self.fc = nn.Linear(self.input_size, self.params.word_vocab_size)
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
        rnn_out, final_state = self.batch_unrolling_beam(decoder_input,  initial_state, encoder_outputs)

        return rnn_out, final_state

    def batch_unrolling_beam(self, decoder_input, initial_state, encoder_outputs):
        [batch_size, seq_len, embeddings] = decoder_input.size()
        output_words = t.empty(decoder_input.size(), requires_grad=True).cuda()
        output_h = t.empty(initial_state[0].size(), requires_grad=True).cuda()
        output_c = t.empty(initial_state[0].size(), requires_grad=True).cuda()
        encoder_outputs = self.fc_enc(encoder_outputs)
        input = decoder_input[sentence_id].unsqueeze(0)
        for sentence_id in range(batch_size):
            state = (initial_state[0][:, sentence_id, :].unsqueeze(1).contiguous(), initial_state[1][:, sentence_id, :].unsqueeze(1).contiguous())
            sent_outputs = encoder_outputs
            
            lstm_out, state = self.rnn(input, state)
            
            # Calculating Alignment Scores - see Attention class for the forward pass function
            alignment_scores = self.attention(lstm_out, sent_outputs)
            # Softmaxing alignment scores to obtain Attention weights
            attn_weights = F.softmax(alignment_scores.view(1,-1), dim=1)
            
            # Multiplying Attention weights with encoder outputs to get context vector
            context_vector = t.bmm(attn_weights.unsqueeze(0), sent_outputs)
            
            # Concatenating output from LSTM with context vector
            output = t.cat((lstm_out, context_vector),-1)
            output = self.fc_out(output)

            output_words[sentence_id] = output
            
            output_h[:, sentence_id] = state[0].squeeze(1)
            output_c[:, sentence_id] = state[1].squeeze(1)
          
        state = (output_h, output_c)

        return output_words, state

    def batch_unrolling(self, decoder_input, initial_state, encoder_outputs):
        [batch_size, seq_len, embeddings] = decoder_input.size()
        output_words = t.empty(decoder_input.size(), requires_grad=True).cuda()
        encoder_outputs = self.fc_enc(encoder_outputs)
        lstm_out, state = self.rnn(decoder_input, initial_state)

        for sentence_id in range(batch_size):
            sent_outputs = encoder_outputs[sentence_id].unsqueeze(0)
            for word_id in range(seq_len):  
                attention_input = lstm_out[sentence_id, word_id, :].unsqueeze(0).unsqueeze(0)

                # Calculating Alignment Scores - see Attention class for the forward pass function
                alignment_scores = self.attention(attention_input, sent_outputs)
                # Softmaxing alignment scores to obtain Attention weights
                attn_weights = F.softmax(alignment_scores.view(1,-1), dim=1)

                # Multiplying Attention weights with encoder outputs to get context vector
                context_vector = t.bmm(attn_weights.unsqueeze(0), sent_outputs)
                
                # Concatenating output from LSTM with context vector
                output = t.cat((attention_input, context_vector),-1)
                output = self.fc_out(output)
                output_words[sentence_id, word_id] = output

        return output_words, state

    
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

        rnn_out, final_state = self.batch_unrolling(decoder_input, initial_state, encoder_outputs)
        
        rnn_out = rnn_out.contiguous().view(-1, self.params.decoder_rnn_size)
    
        result = self.fc(rnn_out)
        result = result.view(batch_size, seq_len, self.params.word_vocab_size)

        return result, final_state

class Attention(nn.Module):
    def __init__(self, hidden_size, method="dot"):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        # Defining the layers/weights required depending on alignment scoring method
        if method == "general":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
            
        elif method == "concat":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
            self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, decoder_hidden, encoder_outputs):
        if self.method == "dot":
            # For the dot scoring method, no weights or linear layers are involved
            
            return encoder_outputs.bmm(decoder_hidden.view(1,-1,1)).squeeze(-1)

        elif self.method == "general":
            # For general scoring, decoder hidden state is passed through linear layers to introduce a weight matrix
            out = self.fc(decoder_hidden)
        
            return encoder_outputs.bmm(out.transpose(1,2)).squeeze(-1) # use for batch unrolling
            # return encoder_outputs.bmm(out.view(1,-1,1)).squeeze(-1) # normalling unrolling

        elif self.method == "concat":
            # For concat scoring, decoder hidden state and encoder outputs are concatenated first
            out = torch.tanh(self.fc(decoder_hidden+encoder_outputs))
            return out.bmm(self.weight.unsqueeze(-1)).squeeze(-1)

class DecoderResidualAttention(nn.Module):
    def __init__(self, params):
        super(DecoderResidual, self).__init__()

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
        
        self.rnn_3 = nn.LSTM(input_size=self.params.decoder_rnn_size,
                           hidden_size=self.params.decoder_rnn_size,
                           num_layers=self.params.decoder_num_layers,
                           batch_first=True,
                           bidirectional=False)

        self.rnn_4 = nn.LSTM(input_size=self.params.decoder_rnn_size,
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
        rnn_out, final_state = self.batch_unrolling(decoder_input,  initial_state, False)

        return rnn_out, final_state

    def batch_unrolling(self, decoder_input,  initial_state, x=None):
        rnn_out_1, _ = self.rnn_1(decoder_input, initial_state)
        rnn_out = t.add(rnn_out_1, decoder_input)
        rnn_out_2, _ = self.rnn_2(rnn_out, initial_state)    
        rnn_out = t.add(rnn_out, rnn_out_2)
        rnn_out_3, _ = self.rnn_3(rnn_out, initial_state)    
        rnn_out = t.add(rnn_out_3, decoder_input)
        rnn_out, final_state = self.rnn_4(rnn_out, initial_state)  

        return rnn_out, final_state

    # # https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
    # def residual_unrolling_in_the_deep(self, decoder_input,  initial_state, step=True):
    #     [batch_size, seq_len, _] = decoder_input.size()
    #     output_words = t.empty(decoder_input.size(), requires_grad=True).cuda()
    #     h_0_states = t.empty(initial_state[0].size(), requires_grad=True).cuda()
    #     c_0_states = t.empty(initial_state[1].size(), requires_grad=True).cuda()
    #     h_0, c_0 = initial_state[0], initial_state[1]
    #     for sentence_id in range(batch_size):
    #         state = (h_0[:, sentence_id, :].unsqueeze(1).contiguous(), c_0[:, sentence_id, :].unsqueeze(1).contiguous())
    #         for word_id in range(seq_len):
    #             words = decoder_input[sentence_id, word_id, :].view(1, 1, -1)
    #             rnn_out, (h_n, c_n) = self.rnn_1(word, state)
    #             h_n_new = t.add(words, h_n[-1,:,:].unsqueeze(1))
    #             rnn_out, final_state = self.rnn_2(h_n_new.cuda())
    #             output_words[sentence_id, word_id] = rnn_out.cuda()
        
    #         h_0_states[0, sentence_id] = h_n_new[-1,:,:].unsqueeze(1).cuda()
    #         c_0_states[0, sentence_id] = c_n[-1,:,:].unsqueeze(1).cuda()
    #         h_0_states[1, sentence_id] = final_state[0].cuda()
    #         c_0_states[1, sentence_id] = final_state[1].cuda()
    #     rnn_out = output_words
    #     final_state = (h_0_states, c_0_states)

    #     return rnn_out, initial_state

    
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

        rnn_out, final_state = self.batch_unrolling(decoder_input, initial_state)
        rnn_out = rnn_out.contiguous().view(-1, self.params.decoder_rnn_size)

        result = self.fc(rnn_out)
        result = result.view(batch_size, seq_len, self.params.word_vocab_size)

        return result, final_state


