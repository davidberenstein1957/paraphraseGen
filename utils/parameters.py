from .functional import *


class Parameters:
    def __init__(
        self,
        max_word_len: int,
        max_seq_len: int,
        word_vocab_size: int,
        char_vocab_size: int,
        attn_model: bool,
        res_model: bool,
        hrvae: bool,
        wae: bool,
    ):
        """
        [summary] initializes structural/architectural model parameters
        """
        self.max_word_len = int(max_word_len)
        self.max_seq_len = int(max_seq_len) + 1  # go or eos token

        self.word_vocab_size = int(word_vocab_size)
        self.char_vocab_size = int(char_vocab_size)

        self.word_embed_size = 300
        self.char_embed_size = 15

        self.kernels = [(1, 25), (2, 50), (3, 75), (4, 100), (5, 125), (6, 150)]
        self.sum_depth = fold(lambda x, y: x + y, [depth for _, depth in self.kernels], 0)

        self.latent_variable_size = 1100

        if res_model:
            self.encoder_rnn_size = self.latent_variable_size + self.word_embed_size
            self.decoder_rnn_size = self.latent_variable_size + self.word_embed_size
        else:
            self.encoder_rnn_size = 600
            self.decoder_rnn_size = 600

        self.encoder_num_layers = 1
        self.decoder_num_layers = 2

        self.attn_model = attn_model
        self.res_model = res_model
        self.hrvae = hrvae
        self.wae = wae
