import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from numpy import dot
from numpy.linalg import norm
from torch.autograd import Variable

from beam_search import Beam
from selfModules.embedding import Embedding
from utils.functional import fold, parameters_allocation_check

# import tensorflow as tf
# import tensorflow_hub as hub
from .decoder import Decoder, DecoderAttention, DecoderResidual, DecoderResidualAttention
from .encoder import Encoder, EncoderHR


class RVAE(nn.Module):
    def __init__(self, params, params_2, path):
        super(RVAE, self).__init__()

        self.params = params
        self.params_2 = params_2  # Encoder-2 parameters

        self.embedding = Embedding(self.params, path)
        self.embedding_2 = Embedding(self.params_2, path, True)

        self.encoder_original = Encoder(self.params)
        if self.params.hrvae:
            self.encoder_paraphrase = EncoderHR(self.params_2)
        else:
            self.encoder_paraphrase = Encoder(self.params_2)

        # self.embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

        self.context_to_mu = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)
        self.context_to_logvar = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)

        if self.params.res_model:
            self.decoder = DecoderResidual(self.params_2)
        else:
            self.decoder = Decoder(self.params_2)

    def forward(
        self,
        unk_idx,
        drop_prob,
        encoder_word_input=None,
        encoder_character_input=None,
        encoder_word_input_2=None,
        encoder_character_input_2=None,
        decoder_word_input_2=None,
        decoder_character_input_2=None,
        z=None,
        initial_state=None,
    ):

        # Modified the parameters of forward function according to Encoder-2
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

        assert parameters_allocation_check(
            self
        ), "Invalid CUDA options. Parameters should be allocated in the same memory"
        use_cuda = self.embedding.word_embed.weight.is_cuda

        assert (
            z is None
            and fold(
                lambda acc, parameter: acc and parameter is not None,
                [encoder_word_input, encoder_character_input, decoder_word_input_2],
                True,
            )
            or (z is not None and decoder_word_input_2 is not None)
        ), "Invalid input. If z is None then encoder and decoder inputs should be passed as arguments"

        if z is None:
            """Get context from encoder and sample z ~ N(mu, std)"""  # 把word和character拼接成一个向量
            [batch_size, _] = encoder_word_input.size()

            encoder_input = self.embedding(encoder_word_input, encoder_character_input, unk_idx, drop_prob)

            """ ===================================================Doing the same for encoder-2===================================================
            """
            [batch_size_2, _] = encoder_word_input_2.size()

            encoder_input_2 = self.embedding_2(encoder_word_input_2, encoder_character_input_2, unk_idx, drop_prob)

            """ ==================================================================================================================================
            """

            enc_out_original, context, h_0, c_0, _ = self.encoder_original(encoder_input, None)
            state_original = (h_0, c_0)  # Final state of Encoder-1 原始句子编码
            # state_original = context
            enc_out_paraphrase, context_2, h_0, c_0, context_ = self.encoder_paraphrase(
                encoder_input_2, state_original
            )  # Encoder_2 for Ques_2  接下去跟释义句编码
            state_paraphrase = (h_0, c_0)  # Final state of Encoder-2 原始句子编码
            # state_paraphrase = context_2

            if context_ is not None:

                mu_ = []
                logvar_ = []
                for entry in context_:
                    mu_.append(self.context_to_mu(entry))
                    logvar_.append(self.context_to_logvar(entry))

                z_sampled = self.sample_gaussian(batch_size)
                if use_cuda:
                    z_sampled = z_sampled.cuda()

                mu = t.stack(mu_)
                logvar = t.stack(logvar_)

                if self.params.wae:
                    z_tilda = self.sample_z_tilda_from_posterior(z_sampled, logvar_[-1], mu_[-1], 1).cuda()
                    p = t.distributions.Normal(mu, t.exp(logvar))
                    q = t.distributions.Normal(mu, t.ones(logvar.size()).cuda())
                    kld = t.sum(t.distributions.kl_divergence(p, q))
                    kld = kld / mu.shape[0]
                    kld = 0
                    for i in range(len(mu_)):
                        p = t.distributions.Normal(mu_[i], t.exp(logvar_[i]))
                        q = t.distributions.Normal(mu_[i], t.ones(logvar.size()).cuda())
                        kld += t.sum(t.distributions.kl_divergence(p, q))
                    kld = kld / len(mu_)
                    wasserstein_loss = self.imq_kernel(z_sampled, z_tilda, self.params.latent_variable_size)
                    kld = 0.01 * kld + 10 * wasserstein_loss
                else:
                    z_tilda = self.sample_z_tilda_from_posterior(z_sampled, logvar_[-1], mu_[-1], 0.5).cuda()
                    kld = 0
                    for i in range(len(mu_)):
                        kld += (-0.5 * t.sum(logvar_[i] - t.pow(mu_[i], 2) - t.exp(logvar_[i]) + 1, 1)).mean().squeeze()
                    kld = kld / len(mu_)

            else:

                mu = self.context_to_mu(context_2)
                logvar = self.context_to_logvar(context_2)

                z_sampled = self.sample_gaussian(batch_size)
                if use_cuda:
                    z_sampled = z_sampled.cuda()

                if self.params.wae:
                    z_tilda = self.sample_z_tilda_from_posterior(z_sampled, logvar, mu, 1).cuda()
                    p = t.distributions.Normal(mu, t.exp(logvar))
                    q = t.distributions.Normal(mu, t.ones(logvar.size()).cuda())
                    kld = t.sum(t.distributions.kl_divergence(p, q))
                    wasserstein_loss = self.imq_kernel(z_sampled, z_tilda, self.params.latent_variable_size)
                    kld = 0.01 * kld + 10 * wasserstein_loss
                else:
                    z_tilda = self.sample_z_tilda_from_posterior(z_sampled, logvar, mu, 0.5).cuda()
                    kld = (-0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1)).mean().squeeze()
        else:
            kld = None
            mu = None
            std = None

        # What to do with this decoder input ? --> Slightly resolved
        decoder_input_2 = self.embedding_2.word_embed(decoder_word_input_2)
        # if context_ is not None:
        #     decoder_input_2 = t.ones(decoder_input_2.size()).cuda()
        out, final_state = self.decoder(decoder_input_2, z_tilda, drop_prob, enc_out_paraphrase, state_original)

        return out, final_state, kld, mu, None

    def sample_z_tilda_from_posterior(self, z_sampled, z_log_sigma, z_mean, z_temperature=0.5):
        """(Differentiably!) draw sample from Gaussian with given shape, subject to random noise epsilon"""
        return z_sampled * t.exp(z_log_sigma * z_temperature) + z_mean  # N(mu, I * sigma**2)

    def sample_gaussian(self, batch_size):
        """(Differentiably!) draw sample from Gaussian with given shape, subject to random noise epsilon"""
        return Variable(t.randn([batch_size, self.params.latent_variable_size]))  # Dimension [batch_size x latent_dim]

    def imq_kernel(self, sample_qz: t.Tensor, sample_pz: t.Tensor, h_dim: int):
        batch_size = sample_pz.size(0)

        norms_pz = sample_pz.pow(2).sum(1, keepdim=True)  # batch_size x 1
        prods_pz = t.mm(sample_pz, sample_pz.t())  # batch_size x batch_size
        dists_pz = norms_pz + norms_pz.t() - 2 * prods_pz

        norms_qz = sample_qz.pow(2).sum(1, keepdim=True)  # batch_size x 1
        prods_qz = t.mm(sample_qz, sample_qz.t())  # batch_size x batch_size
        dists_qz = norms_qz + norms_qz.t() - 2 * prods_qz

        dotprods = t.mm(sample_qz, sample_pz.t())
        distances = norms_qz + norms_pz.t() - 2 * dotprods

        stats = 0
        Cbase = 2.0 * h_dim * 2.0 * 1.0
        for scale in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
            C = Cbase * scale
            res1 = C / (C + dists_qz)
            res1 += C / (C + dists_pz)

            if t.cuda.is_available():
                res1 = (1 - t.eye(batch_size).cuda()) * res1
            else:
                res1 = (1 - t.eye(batch_size)) * res1

            res1 = res1.sum() / (batch_size * batch_size - batch_size)
            res2 = C / (C + distances)
            res2 = res2.sum() * 2.0 / (batch_size * batch_size)
            stats += res1 - res2

        return stats

    def learnable_parameters(self):
        # wordembedding是固定值，因此必须从优化器的参数列表里移除。
        # word_embedding is constant parameter thus it must be dropped from list of parameters for optimizer
        return [p for p in self.parameters() if p.requires_grad]

    def trainer(self, optimizer, batch_loader, batch_loader_2):
        def train(coef, batch_size, use_cuda, dropout, start_index):
            input = batch_loader.next_batch(batch_size, "train", start_index)
            input = [Variable(t.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]
            # 这里是data/train.txt,转换变成embedding，用pand补齐，
            # 其中encoder_word_input, encoder_character_input是将 xo原始句输入倒过来前面加若干占位符，
            # decoder_word_input, decoder_character_input是 xo原始句加了开始符号末端补齐
            # target，结束句子后面加了结束符，target是xo原始句加结束符后面加若干占位符
            [
                encoder_word_input,
                encoder_character_input,
                decoder_word_input,
                decoder_character_input,
                target,
                _,
            ] = input

            """ =================================================== Input for Encoder-2 ========================================================
            """

            input_2 = batch_loader_2.next_batch(batch_size, "train", start_index)
            input_2 = [Variable(t.from_numpy(var)) for var in input_2]
            input_2 = [var.long() for var in input_2]
            input_2 = [var.cuda() if use_cuda else var for var in input_2]
            # 这里是data/super/train.txt,转换变成embedding，用pand补齐，
            # 其中encoder_word_input, encoder_character_input是将 释义句xp输入倒过来前面加若干占位符，
            # decoder_word_input, decoder_character_input是 释义句xp加了开始符号末端补齐
            # target，结束句子后面加了结束符，target是释义句xp加结束符后面加若干占位符
            [
                encoder_word_input_2,
                encoder_character_input_2,
                decoder_word_input_2,
                decoder_character_input_2,
                target,
                _,
            ] = input_2
            unk_idx = None

            """ ================================================================================================================================
            """
            # 这里encoder-input是原始句子xo的输入（句子翻转），encoder-input2是释义句xp的输入（句子翻转），decoder-input是释义句加加开始符号
            logits, _, kld, _, _ = self(
                unk_idx,
                dropout,
                encoder_word_input,
                encoder_character_input,
                encoder_word_input_2,
                encoder_character_input_2,
                decoder_word_input_2,
                decoder_character_input_2,
                z=None,
            )

            logits = logits.view(-1, self.params_2.word_vocab_size)
            target = target.view(-1)

            # 前面logit 是每一步输出的词汇表所有词的概率， target是每一步对应的词的索引不用变成onehot，函数内部做变换
            cross_entropy = F.cross_entropy(logits, target)

            if self.params.wae:
                loss = 1 * cross_entropy + coef * kld  # 79应该是作者拍脑袋的
            elif self.params.hrvae:
                loss = 79 * cross_entropy + coef * kld  # 79应该是作者拍脑袋的
            else:
                loss = 79 * cross_entropy + coef * kld  # 79应该是作者拍脑袋的

            optimizer.zero_grad()  # 标准用法先计算损失函数值，然后初始化梯度为0，
            loss.backward()  # 然后反向传递
            optimizer.step()  # 反向跟新梯度

            return cross_entropy, kld, coef  # 交叉熵，kl-devergence，kld-coef是为了让他

        return train

    def validater(self, batch_loader, batch_loader_2):
        def validate(batch_size, use_cuda, start_index):
            input = batch_loader.next_batch(batch_size, "valid", start_index)
            input = [Variable(t.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]

            [encoder_word_input, encoder_character_input, decoder_word_input, decoder_character_input, target] = input

            """ ==================================================== Input for Encoder-2 ========================================================
            """

            input_2 = batch_loader_2.next_batch(batch_size, "valid", start_index)
            input_2 = [Variable(t.from_numpy(var)) for var in input_2]
            input_2 = [var.long() for var in input_2]
            input_2 = [var.cuda() if use_cuda else var for var in input_2]
            [
                encoder_word_input_2,
                encoder_character_input_2,
                decoder_word_input_2,
                decoder_character_input_2,
                target,
            ] = input_2

            """ ==================================================================================================================================
            """
            unk_idx = batch_loader_2.word_to_idx[batch_loader_2.unk_token]
            logits, _, kld, _, _ = self(
                unk_idx,
                0.0,
                encoder_word_input,
                encoder_character_input,
                encoder_word_input_2,
                encoder_character_input_2,
                decoder_word_input_2,
                decoder_character_input_2,
                z=None,
            )

            # logits = logits.view(-1, self.params.word_vocab_size)
            logits = logits.view(-1, self.params_2.word_vocab_size)
            target = target.view(-1)
            cross_entropy = F.cross_entropy(logits, target)

            return cross_entropy, kld

        return validate

    def sample(self, batch_loader, seq_len, seed, use_cuda, State):
        # seed = Variable(t.from_numpy(seed).float())
        # seed = Variable(t.randn([1, parameters.latent_variable_size]))
        if use_cuda:
            seed = seed.cuda()

        decoder_word_input_np, decoder_character_input_np = batch_loader.go_input(1)

        decoder_word_input = Variable(t.from_numpy(decoder_word_input_np).long())
        decoder_character_input = Variable(t.from_numpy(decoder_character_input_np).long())

        if use_cuda:
            decoder_word_input, decoder_character_input = decoder_word_input.cuda(), decoder_character_input.cuda()

        result = ""

        initial_state = State

        for i in range(seq_len):
            logits, initial_state, _, _, _ = self(
                0.0, None, None, None, None, decoder_word_input, decoder_character_input, seed, initial_state
            )

            # forward(self, drop_prob,
            #           encoder_word_input=None, encoder_character_input=None,
            #           encoder_word_input_2=None, encoder_character_input_2=None,
            #           decoder_word_input_2=None, decoder_character_input_2=None,
            #           z=None, initial_state=None):

            # logits = logits.view(-1, self.params.word_vocab_size)
            # logits = logits.view(-1, self.params.word_vocab_size)
            logits = logits.view(-1, self.params_2.word_vocab_size)
            # print '---------------------------------------'
            # print 'Printing logits'
            # print logits
            # print '------------------------------------------'

            prediction = F.softmax(logits)

            word = batch_loader.sample_word_from_distribution(prediction.data.cpu().numpy()[-1])

            if word == batch_loader.end_token:
                break

            result += " " + word

            decoder_word_input_np = np.array([[batch_loader.word_to_idx[word]]])
            decoder_character_input_np = np.array([[batch_loader.encode_characters(word)]])

            decoder_word_input = Variable(t.from_numpy(decoder_word_input_np).long())
            decoder_character_input = Variable(t.from_numpy(decoder_character_input_np).long())

            if use_cuda:
                decoder_word_input, decoder_character_input = decoder_word_input.cuda(), decoder_character_input.cuda()

        return result

    def sampler(self, batch_loader, batch_loader_2, seq_len, seed, use_cuda, i, beam_size, n_best):
        input = batch_loader.next_batch(1, "valid", i)
        input = [Variable(t.from_numpy(var)) for var in input]
        input = [var.long() for var in input]
        input = [var.cuda() if use_cuda else var for var in input]
        [encoder_word_input, encoder_character_input, decoder_word_input, decoder_character_input, target, _] = input

        encoder_input = self.embedding(encoder_word_input, encoder_character_input)

        encoder_output, _, h0, c0, _ = self.encoder_original(encoder_input, None)
        State = (h0, c0)

        # print '----------------------'
        # print 'Printing h0 ---------->'
        # print h0
        # print '----------------------'

        # State = None
        results, scores = self.sample_beam(
            batch_loader_2, seq_len, seed, use_cuda, State, beam_size, n_best, encoder_output
        )

        return results, scores

    def sample_beam(self, batch_loader, seq_len, seed, use_cuda, State, beam_size, n_best, encoder_output):
        # seed = Variable(t.from_numpy(seed).float())
        if use_cuda:
            seed = seed.cuda()

        decoder_word_input_np, decoder_character_input_np = batch_loader.go_input(1)

        decoder_word_input = Variable(t.from_numpy(decoder_word_input_np).long())
        decoder_character_input = Variable(t.from_numpy(decoder_character_input_np).long())

        if use_cuda:
            decoder_word_input, decoder_character_input = decoder_word_input.cuda(), decoder_character_input.cuda()

        dec_states = State

        # print '========= Before ================'
        # print "dec_states:", dec_states[0].size()
        # print "dec_states:", dec_states[1].size()
        # print '=================================='

        # dec_states = [
        #     Variable(dec_states[0].repeat(1, beam_size, 1)),
        #     Variable(dec_states[1].repeat(1, beam_size, 1))
        # ]

        dec_states = [dec_states[0].repeat(1, beam_size, 1), dec_states[1].repeat(1, beam_size, 1)]

        # print'========== After =================='
        # print "dec_states:", dec_states[0].size()
        # print "dec_states:", dec_states[1].size()
        # print '=================================='
        # exit()

        drop_prob = 0.0
        beam_size = beam_size
        batch_size = 1

        beam = [Beam(beam_size, batch_loader, cuda=True) for k in range(batch_size)]

        batch_idx = list(range(batch_size))
        remaining_sents = batch_size

        for i in range(seq_len):
            input = t.stack([b.get_current_state() for b in beam if not b.done]).t().contiguous().view(1, -1)

            trg_emb = self.embedding_2.word_embed(Variable(input).transpose(1, 0))

            trg_h, dec_states = self.decoder.only_decoder_beam(trg_emb, seed, drop_prob, encoder_output, dec_states)

            # trg_h, (trg_h_t, trg_c_t) = self.model.decoder(trg_emb, (dec_states[0].squeeze(0), dec_states[1].squeeze(0)), context )

            # print trg_h.size()
            # print trg_h_t.size()
            # print trg_c_t.size()

            # dec_states = (trg_h_t, trg_c_t)

            # print 'State dimension ----------->'
            # print State[0].size()
            # print State[1].size()
            # print '======================================='
            # print "dec_states:", dec_states[0].size()
            # print "dec_states:", dec_states[1].size()
            # print '========== Things successful ==========='

            # exit()

            dec_out = trg_h.squeeze(1)

            # print "dec_out:", dec_out.size()

            out = F.softmax(self.decoder.fc(dec_out)).unsqueeze(0)

            word_lk = out.view(beam_size, remaining_sents, -1).transpose(0, 1).contiguous()

            active = []
            for b in range(batch_size):
                if beam[b].done:
                    continue

                idx = batch_idx[b]
                if not beam[b].advance(word_lk.data[idx]):
                    active += [b]

                for dec_state in dec_states:  # iterate over h, c
                    # layers x beam*sent x dim
                    sent_states = dec_state.view(-1, beam_size, remaining_sents, dec_state.size(2))[:, :, idx]
                    sent_states.data.copy_(sent_states.data.index_select(1, beam[b].get_current_origin()))

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_idx = t.cuda.LongTensor([batch_idx[k] for k in active])
            batch_idx = {beam: idx for idx, beam in enumerate(active)}

            def update_active(t):
                # select only the remaining active sentences
                view = t.data.view(-1, remaining_sents, self.params.decoder_rnn_size)
                new_size = list(t.size())
                new_size[-2] = new_size[-2] * len(active_idx) // remaining_sents
                return Variable(view.index_select(1, active_idx).view(*new_size))

            dec_states = (update_active(dec_states[0]), update_active(dec_states[1]))
            dec_out = update_active(dec_out)
            # context = update_active(context)

            remaining_sents = len(active)

        # (4) package everything up

        allHyp, allScores = [], []

        for b in range(batch_size):
            scores, ks = beam[b].sort_best()
            # print scores
            # print ks
            allScores += [scores[:n_best]]
            hyps = zip(*[beam[b].get_hyp(k) for k in ks[:n_best]])
            # print hyps
            # print "------------------"
            allHyp += [hyps]

        # print '==== Complete ========='

        return allHyp, allScores
