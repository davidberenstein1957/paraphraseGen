# -*- coding: utf-8 -*-
import argparse

import numpy as np
import torch as t
from torch.autograd import Variable
from torch.optim import SGD

from selfModules.neg import NEG_loss
from utils.batch_loader import BatchLoader
from utils.parameters import Parameters

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="word2vec")
    parser.add_argument(
        "--num-iterations", type=int, default=2000000, metavar="NI", help="num iterations (default: 1000000)"
    )
    parser.add_argument("--batch-size", type=int, default=10, metavar="BS", help="batch size (default: 10)")
    parser.add_argument("--num-sample", type=int, default=5, metavar="NS", help="num sample (default: 5)")
    parser.add_argument("--use-cuda", type=bool, default=True, metavar="CUDA", help="use cuda (default: True)")
    parser.add_argument("--data-name", type=str, default="both", metavar="CUDA", help="use cuda (default: False)")
    args = parser.parse_args()

    data_name = args.data_name  # quora, mscoco, both

    path = "paraphraseGen/"

    data_files = [path + f"data/super/train_{data_name}_2.txt", path + f"data/super/test_{data_name}_2.txt"]

    idx_files = [
        path + f"data/super/words_vocab_{data_name}_2.pkl",
        path + f"data/super/characters_vocab_{data_name}_2.pkl",
    ]

    tensor_files = [
        [
            path + f"data/super/train_word_tensor_{data_name}_2.npy",
            path + f"data/super/valid_word_tensor_{data_name}_2.npy",
        ],
        [
            path + f"data/super/train_character_tensor_{data_name}_2.npy",
            path + f"data/super/valid_character_tensor_{data_name}_2.npy",
        ],
    ]
    batch_loader_2 = BatchLoader(data_files, idx_files, tensor_files, path)

    # batch_loader_2 = BatchLoader('')
    params = Parameters(
        batch_loader_2.max_word_len,
        batch_loader_2.max_seq_len,
        batch_loader_2.words_vocab_size,
        batch_loader_2.chars_vocab_size,
        data_name,
        False,
        False,
        False
    )

    neg_loss = NEG_loss(params.word_vocab_size, params.word_embed_size)
    if args.use_cuda:
        neg_loss = neg_loss.cuda()

    # NEG_loss is defined over two embedding matrixes with shape of [params.word_vocab_size, params.word_embed_size]
    optimizer = SGD(neg_loss.parameters(), 0.1)

    for iteration in range(args.num_iterations):

        input_idx, target_idx = batch_loader_2.next_embedding_seq(args.batch_size)

        input = Variable(t.from_numpy(input_idx).long())
        target = Variable(t.from_numpy(target_idx).long())
        if args.use_cuda:
            input, target = input.cuda(), target.cuda()

        out = neg_loss(input, target, args.num_sample).mean()

        optimizer.zero_grad()
        out.backward()
        optimizer.step()

        if iteration % 500 == 0:
            out = out.cpu().data.numpy()
            print("iteration = {}, loss = {}".format(iteration, out))

    word_embeddings = neg_loss.input_embeddings()
    # Saves the word embeddings at the end of this programs
    np.save(path + f"data/super/word_embeddings_{data_name}.npy", word_embeddings)
