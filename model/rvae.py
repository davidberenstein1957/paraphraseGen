import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from beam_search import Beam
from selfModules.embedding import Embedding
from utils.functional import fold, parameters_allocation_check

from .decoder import Decoder, DecoderAttention, DecoderResidual, DecoderResidualAttention
from .encoder import Encoder, EncoderHR


class RVAE(nn.Module):
    def __init__(self, params: object, params_2: object, path:str) -> None:
        """
        [summary] initializes the RVAE  with the correct parameters and data files

        Args:
            params (object): [description] parameters for original encoder
            params_2 (object): [description] parameters for paraphrase encoder
            path (str): [description] a path to the data files
        """
        super(RVAE, self).__init__()

        self.params = params
        self.params_2 = params_2 

        self.embedding = Embedding(self.params, path)
        self.embedding_2 = Embedding(self.params_2, path, True)

        self.encoder_original = Encoder(self.params)
        if self.params.hrvae: 
            self.encoder_paraphrase = EncoderHR(self.params_2)
        else:
            self.encoder_paraphrase = Encoder(self.params_2)

        self.context_to_mu = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)
        self.context_to_logvar = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)

        if self.params.attn_model and self.params.res_model:
            self.decoder = DecoderResidualAttention(self.params_2)
        elif self.params.attn_model:
            self.decoder = DecoderAttention(self.params_2)
        elif self.params.res_model:
            self.decoder = DecoderResidual(self.params_2)
        else:
            self.decoder = Decoder(self.params_2)

    def forward(self, unk_idx: int, drop_prob: float,
                encoder_word_input: object=None, encoder_character_input: object=None,
                encoder_word_input_2: object=None, encoder_character_input_2: object=None,
                decoder_word_input_2: object=None, decoder_character_input_2: object=None,
                z: object=None, initial_state: tuple=None) -> tuple:
        """
        :param encoder_word_input: An tensor with shape of [batch_size, seq_len] of Long type
        :param encoder_character_input: An tensor with shape of [batch_size, seq_len, max_word_len] of Long type
        :param decoder_word_input: An tensor with shape of [batch_size, max_seq_len + 1] of Long type
        :param initial_state: initial state of decoder rnn in order to perform sampling

        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout

        :param z: context if sampling is performing

        :return: unnormalized logits of sentence words distribution probabilities
                    with shape of [batch_size, seq_len, word_vocab_size]
                 final rnn state with shape of [num_layers, batch_size, decoder_rnn_size]
        """

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'
        use_cuda = self.embedding.word_embed.weight.is_cuda

        assert z is None and fold(lambda acc, parameter: acc and parameter is not None,
                                  [encoder_word_input, encoder_character_input, decoder_word_input_2],
                                  True) \
            or (z is not None and decoder_word_input_2 is not None), \
            "Invalid input. If z is None then encoder and decoder inputs should be passed as arguments"

        if z is None:
            ''' Get context from encoder and sample z ~ N(mu, std) '''  
            [batch_size, _] = encoder_word_input.size()
            encoder_input = self.embedding(encoder_word_input, encoder_character_input, unk_idx, drop_prob)

            ''' ===================================================Doing the same for encoder-2=================================================== '''
            [batch_size_2, _] = encoder_word_input_2.size()
            encoder_input_2 = self.embedding_2(encoder_word_input_2, encoder_character_input_2, unk_idx, drop_prob)
            
            ''' ================================================================================================================================== '''
            enc_out_original, context, h_0, c_0, _ = self.encoder_original(encoder_input, None)
            state_original = (h_0, c_0)  # Final state of Encoder-1 
            enc_out_paraphrase, context_2, h_0, c_0, context_ = self.encoder_paraphrase(encoder_input_2, state_original)  # Encoder_2 for Ques_2  
            state_paraphrase = (h_0, c_0)  # Final state of Encoder-2 

            if context_ is not None:

                mu_ = []
                logvar_ = []
                for entry in context_:
                    mu_.append(self.context_to_mu(entry))
                    logvar_.append(self.context_to_logvar(entry))
                
                std = t.exp(0.5 * logvar_[-1])

                z = Variable(t.randn([batch_size, self.params.latent_variable_size]))
                if use_cuda:
                    z = z.cuda()

                z = z * std + mu_[-1]

                mu = t.stack(mu_)
                logvar = t.stack(logvar_)

                kld = -0.5 * t.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kld = kld / mu.shape[0]
            
            else:

                mu = self.context_to_mu(context_2)
                logvar = self.context_to_logvar(context_2)
                std = t.exp(0.5 * logvar)

                z = Variable(t.randn([batch_size, self.params.latent_variable_size]))
                if use_cuda:
                    z = z.cuda()

                z = z * std + mu

                kld = (-0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1)).mean().squeeze()

        else:
            kld = None
            mu = None
            std = None

        decoder_input_2 = self.embedding_2.word_embed(decoder_word_input_2)
        out, final_state = self.decoder(decoder_input_2, z, drop_prob, enc_out_paraphrase, state_original)

        return out, final_state, kld, mu, std

    def learnable_parameters(self) -> list:
        """ creates a gradients for each parameter in the class to be optimized """
        return [p for p in self.parameters() if p.requires_grad]

    def trainer(self, optimizer: object, batch_loader: object, batch_loader_2: object) -> object:
        def train(coef: float, batch_size: int, use_cuda: bool, dropout: float, start_index: int) -> tuple:
            """ train the encoder/decoder step by step via train() """
            input = batch_loader.next_batch(batch_size, 'train', start_index)
            input = [Variable(t.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]
            
            [encoder_word_input, encoder_character_input, decoder_word_input, decoder_character_input, target, _] = input

            ''' =================================================== Input for Encoder-2 ========================================================'''
            input_2 = batch_loader_2.next_batch(batch_size, 'train', start_index)
            input_2 = [Variable(t.from_numpy(var)) for var in input_2]
            input_2 = [var.long() for var in input_2]
            input_2 = [var.cuda() if use_cuda else var for var in input_2]
            [encoder_word_input_2, encoder_character_input_2, decoder_word_input_2, decoder_character_input_2, target, _] = input_2
            unk_idx = None

            ''' ================================================================================================================================ '''
            logits, _, kld, _, _ = self(unk_idx, dropout,
                                        encoder_word_input, encoder_character_input,
                                        encoder_word_input_2, encoder_character_input_2,
                                        decoder_word_input_2, decoder_character_input_2,
                                        z=None)
            
            logits = logits.view(-1, self.params_2.word_vocab_size)
            target = target.view(-1)
            cross_entropy = F.cross_entropy(logits, target)

            loss = 79 * cross_entropy + coef * kld  # 79 as arbitrary loss weight

            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  

            return cross_entropy, kld, coef 
        return train

    def validater(self, batch_loader, batch_loader_2):
        def validate(batch_size, use_cuda, start_index):
            """ validate the encoder/decoder step by step via validate() """
            input = batch_loader.next_batch(batch_size, 'valid', start_index)
            input = [Variable(t.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]

            [encoder_word_input, encoder_character_input, decoder_word_input, decoder_character_input, target] = input

            ''' ==================================================== Input for Encoder-2 ========================================================
            '''

            input_2 = batch_loader_2.next_batch(batch_size, 'valid', start_index)
            input_2 = [Variable(t.from_numpy(var)) for var in input_2]
            input_2 = [var.long() for var in input_2]
            input_2 = [var.cuda() if use_cuda else var for var in input_2]
            [encoder_word_input_2, encoder_character_input_2, decoder_word_input_2, decoder_character_input_2, target] = input_2

            ''' ==================================================================================================================================
            '''
            unk_idx = batch_loader_2.word_to_idx[batch_loader_2.unk_token]
            logits, _, kld, _, _ = self(unk_idx, 0.,
                                        encoder_word_input, encoder_character_input,
                                        encoder_word_input_2, encoder_character_input_2,
                                        decoder_word_input_2, decoder_character_input_2,
                                        z=None)

            # logits = logits.view(-1, self.params.word_vocab_size)
            logits = logits.view(-1, self.params_2.word_vocab_size)
            target = target.view(-1)
            cross_entropy = F.cross_entropy(logits, target)

            return cross_entropy, kld

        return validate

    def sample(self, batch_loader: object, seq_len: int, seed: int, use_cuda: bool, State: object) -> tuple:
        """ unroll the decoder step by step to obtain a sample, based on the input encoded original and a random seed """        
        if use_cuda:
            seed = seed.cuda()

        decoder_word_input_np, decoder_character_input_np = batch_loader.go_input(1)

        decoder_word_input = Variable(t.from_numpy(decoder_word_input_np).long())
        decoder_character_input = Variable(t.from_numpy(decoder_character_input_np).long())

        if use_cuda:
            decoder_word_input, decoder_character_input = decoder_word_input.cuda(), decoder_character_input.cuda()

        result = ''

        initial_state = State

        for i in range(seq_len):
            logits, initial_state, _, _, _ = self(0., None, None,
                                                  None, None,
                                                  decoder_word_input, decoder_character_input,
                                                  seed, initial_state)
            
            logits = logits.view(-1, self.params_2.word_vocab_size)
            prediction = F.softmax(logits)

            word = batch_loader.sample_word_from_distribution(prediction.data.cpu().numpy()[-1])

            if word == batch_loader.end_token:
                break

            result += ' ' + word

            decoder_word_input_np = np.array([[batch_loader.word_to_idx[word]]])
            decoder_character_input_np = np.array([[batch_loader.encode_characters(word)]])

            decoder_word_input = Variable(t.from_numpy(decoder_word_input_np).long())
            decoder_character_input = Variable(t.from_numpy(decoder_character_input_np).long())

            if use_cuda:
                decoder_word_input, decoder_character_input = decoder_word_input.cuda(), decoder_character_input.cuda()

        return result

    def sampler(self, batch_loader, batch_loader_2, seq_len, seed, use_cuda, i, beam_size, n_best):
        """ sample using a encoded sentence and a beam search over the states of the decoder """
        input = batch_loader.next_batch(1, 'valid', i)
        input = [Variable(t.from_numpy(var)) for var in input]
        input = [var.long() for var in input]
        input = [var.cuda() if use_cuda else var for var in input]
        [encoder_word_input, encoder_character_input, decoder_word_input, decoder_character_input, target, _] = input

        encoder_input = self.embedding(encoder_word_input, encoder_character_input)

        encoder_output, _, h0, c0, _ = self.encoder_original(encoder_input, None)
        State = (h0, c0)
        
        results, scores = self.sample_beam(batch_loader_2, seq_len, seed, use_cuda, State, beam_size, n_best, encoder_output)

        return results, scores

    def sample_beam(self, batch_loader, seq_len, seed, use_cuda, State, beam_size, n_best, encoder_output):
        """ sample and beam search for unrolling every step of the decoder based on a encoded original input sentence """
        if use_cuda:
            seed = seed.cuda()

        decoder_word_input_np, decoder_character_input_np = batch_loader.go_input(1)

        decoder_word_input = Variable(t.from_numpy(decoder_word_input_np).long())
        decoder_character_input = Variable(t.from_numpy(decoder_character_input_np).long())

        if use_cuda:
            decoder_word_input, decoder_character_input = decoder_word_input.cuda(), decoder_character_input.cuda()

        dec_states = State
        
        dec_states = [
            dec_states[0].repeat(1, beam_size, 1),
            dec_states[1].repeat(1, beam_size, 1)
        ]

        drop_prob = 0.0
        beam_size = beam_size
        batch_size = 1

        beam = [Beam(beam_size, batch_loader, cuda=True) for k in range(batch_size)]

        batch_idx = list(range(batch_size))
        remaining_sents = batch_size

        for i in range(seq_len):
            input = t.stack(
                [b.get_current_state() for b in beam if not b.done]
            ).t().contiguous().view(1, -1)

            trg_emb = self.embedding_2.word_embed(Variable(input).transpose(1, 0))
            trg_h, dec_states = self.decoder.only_decoder_beam(trg_emb, seed, drop_prob, encoder_output, dec_states)

            dec_out = trg_h.squeeze(1)
            out = F.softmax(self.decoder.fc(dec_out)).unsqueeze(0)

            word_lk = out.view(
                beam_size,
                remaining_sents,
                -1
            ).transpose(0, 1).contiguous()

            active = []
            for b in range(batch_size):
                if beam[b].done:
                    continue

                idx = batch_idx[b]
                if not beam[b].advance(word_lk.data[idx]):
                    active += [b]

                for dec_state in dec_states:  # iterate over h, c
                    # layers x beam*sent x dim
                    sent_states = dec_state.view(
                        -1, beam_size, remaining_sents, dec_state.size(2)
                    )[:, :, idx]
                    sent_states.data.copy_(
                        sent_states.data.index_select(
                            1,
                            beam[b].get_current_origin()
                        )
                    )

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_idx = t.cuda.LongTensor([batch_idx[k] for k in active])
            batch_idx = {beam: idx for idx, beam in enumerate(active)}

            def update_active(t):
                # select only the remaining active sentences
                view = t.data.view(
                    -1, remaining_sents,
                    self.params.decoder_rnn_size
                )
                new_size = list(t.size())
                new_size[-2] = new_size[-2] * len(active_idx) \
                    // remaining_sents
                return Variable(view.index_select(
                    1, active_idx
                ).view(*new_size))

            dec_states = (
                update_active(dec_states[0]),
                update_active(dec_states[1])
            )
            dec_out = update_active(dec_out)
            remaining_sents = len(active)

        allHyp, allScores = [], []

        for b in range(batch_size):
            scores, ks = beam[b].sort_best()
            allScores += [scores[:n_best]]
            hyps = zip(*[beam[b].get_hyp(k) for k in ks[:n_best]])
            allHyp += [hyps]

        # print '==== Complete ========='

        return allHyp, allScores
