import argparse
import os
import time
from random import randint

import numpy as np
import torch as t
from torch.autograd import Variable
from torch.optim import Adam

from model.rvae_previous import RVAE
from utils.batch_loader import BatchLoader
from utils.functional import *
from utils.parameters import Parameters

# from model.rvae import RVAE


if __name__ == "__main__":
    path = "paraphraseGen/"
    save_path = "/content/drive/My Drive/thesis/"
    # if not os.path.exists(path + "data/word_embeddings.npy"):
    #     raise FileNotFoundError("word embeddings file was't found")
    # 一次一句，这样容易看，一次两个词
    parser = argparse.ArgumentParser(description="RVAE")
    parser.add_argument("--num-iterations", type=int, default=120000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--use-cuda", type=bool, default=True)
    parser.add_argument("--learning-rate", type=float, default=0.00005)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--hrvae", type=bool, default=False)
    parser.add_argument("--annealing", type=str, default="mono")  # none, mono, cyc
    parser.add_argument("--use-trained", type=bool, default=False)
    parser.add_argument("--attn-model", type=bool, default=False)
    parser.add_argument("--res-model", type=bool, default=False)
    data_name = "coco"  # quora, coco, both
    parser.add_argument("--data_name", type=str, default=data_name)  # quora, coco, both
    embeddings_name = "coco"  # quora, coco, both
    parser.add_argument("--embeddings_name", type=str, default=data_name)  # quora, coco, both

    parser.add_argument("--use-file", type=bool, default=True)
    parser.add_argument("--test-file", type=str, default=path + f"/data/test_{data_name}.txt")
    parser.add_argument("--train-file", type=str, default=path + f"/data/train_{data_name}.txt")

    parser.add_argument("--num-sample", type=int, default=5)
    parser.add_argument("--beam-top", type=int, default=1)
    parser.add_argument("--beam-size", type=int, default=10)

    parser.add_argument("--ce-result", default="")
    parser.add_argument("--kld-result", default="")
    parser.add_argument("--model-result", default="")

    args = parser.parse_args()

    """ =================== Creating batch_loader for encoder-1 =========================================
    """
    data_files = [path + f"data/train_{data_name}.txt", path + f"data/test_{data_name}.txt"]

    idx_files = [
        path + f"data/words_vocab_{embeddings_name}.pkl",
        path + f"data/characters_vocab_{embeddings_name}.pkl",
    ]

    tensor_files = [
        [
            path + f"data/train_word_tensor_{embeddings_name}.npy",
            path + f"data/valid_word_tensor_{embeddings_name}.npy",
        ],
        [
            path + f"data/train_character_tensor_{embeddings_name}.npy",
            path + f"data/valid_character_tensor_{embeddings_name}.npy",
        ],
    ]

    batch_loader = BatchLoader(data_files, idx_files, tensor_files, path)
    parameters = Parameters(
        batch_loader.max_word_len,
        batch_loader.max_seq_len,
        batch_loader.words_vocab_size,
        batch_loader.chars_vocab_size,
        embeddings_name,
        args.res_model,
        args.hrvae,
    )

    """ =================== Doing the same for encoder-2 ===============================================
    """
    data_files = [path + f"data/super/train_{data_name}_2.txt", path + f"data/super/test_{data_name}_2.txt"]

    idx_files = [
        path + f"data/super/words_vocab_{embeddings_name}_2.pkl",
        path + f"data/super/characters_vocab_{embeddings_name}_2.pkl",
    ]

    tensor_files = [
        [
            path + f"data/super/train_word_tensor_{embeddings_name}_2.npy",
            path + f"data/super/valid_word_tensor_{embeddings_name}_2.npy",
        ],
        [
            path + f"data/super/train_character_tensor_{embeddings_name}_2.npy",
            path + f"data/super/valid_character_tensor_{embeddings_name}_2.npy",
        ],
    ]
    batch_loader_2 = BatchLoader(data_files, idx_files, tensor_files, path)
    parameters_2 = Parameters(
        batch_loader_2.max_word_len,
        batch_loader_2.max_seq_len,
        batch_loader_2.words_vocab_size,
        batch_loader_2.chars_vocab_size,
        embeddings_name,
        args.res_model,
        args.hrvae,
    )
    """=================================================================================================
    """
    modulo_operator = len(open(args.train_file, "r").readlines())
    text_input = ""
    if not args.use_file:
        text_input = input("Input Question : ")
    else:
        file_1 = open(args.test_file, "r")
        data = file_1.readlines()

    rvae = RVAE(parameters, parameters_2, path)
    if args.use_trained:
        rvae.load_state_dict(t.load("trained_RVAE"))
    if args.use_cuda:
        rvae = rvae.cuda()

    optimizer = Adam(rvae.learnable_parameters(), args.learning_rate)

    train_step = rvae.trainer(optimizer, batch_loader, batch_loader_2)  # batchloader里面是原始句子，batechloader2里面存储的是释义句
    validate = rvae.validater(batch_loader, batch_loader_2)

    ce_result = []
    kld_result = []

    start_index = 0
    start_time = time.time()

    coef_modulo = 10000
    sample_modulo = 1000

    if int(args.num_iterations / coef_modulo) % 2 == 0:
        args.num_iterations = int(args.num_iterations / coef_modulo) * coef_modulo
    else:
        args.num_iterations = (int(args.num_iterations / coef_modulo) + 1) * coef_modulo

    for iteration in range(args.num_iterations):
        start_index = (start_index + args.batch_size) % (modulo_operator)

        if args.annealing == "cyc":
            coef = kld_coef_cyc(iteration, coef_modulo)
        elif args.annealing == "mono":
            coef = kld_coef_mono(iteration)
        else:
            coef = 1

        cross_entropy, kld, _ = train_step(coef, args.batch_size, args.use_cuda, args.dropout, start_index)

        if ((iteration % int(sample_modulo) == 0)) & (iteration != 0):
            print("\n")
            print("------------TRAIN-------------")
            print("-------------ETA--------------")
            percentage = (iteration / args.num_iterations) * 100
            time_spend = time.time() - start_time
            print(time_spend)
            time_required = (time_spend / percentage) * (100 - percentage)
            time_required_min = time_required / 60
            print(time_required_min)
            print("----------ITERATION-----------")
            print(iteration, round(percentage, 2))
            print("--------CROSS-ENTROPY---------")
            print(cross_entropy.data.cpu().numpy())
            print("-------------KLD--------------")
            print(kld.data.cpu().numpy())
            print("-----------KLD-coef-----------")
            print(coef)
            print("------------------------------")

        if (iteration % coef_modulo == 0) & (iteration != 0):
            t.save(rvae.state_dict(), save_path + f"/trained_RVAE_{iteration}")
            # np.save(save_path + f"/ce_result_{iteration}.npy".format(args.ce_result), np.array(ce_result))
            # np.save(save_path + f"/kld_result_npy_{iteration}.npy".format(args.kld_result), np.array(kld_result))
            print("MODEL SAVED")

        if iteration % int(sample_modulo) == 0:
            index = randint(0, len(data) - 1)

            ref = data[index]
            hyp_ = []

            if args.use_file:
                print("original sentence:     " + ref)
            else:
                print("original sentence:     " + text_input + "\n")

            for _ in range(args.num_sample):

                seed = Variable(t.randn([1, parameters.latent_variable_size]))
                seed = seed.cuda()

                results, scores = rvae.sampler(
                    batch_loader, batch_loader_2, 50, seed, args.use_cuda, index, args.beam_size, args.beam_top
                )

                for tt in results:

                    for k in range(args.beam_top):
                        sen = " ".join([batch_loader_2.decode_word(x[k]) for x in tt])
                        if batch_loader.end_token in sen:
                            hyp = sen[: sen.index(batch_loader.end_token)]
                            print("generate sentence:     " + hyp)
                        else:
                            hyp = sen
                            print("generate sentence:     " + hyp)

                    hyp_.append(hyp)

            ce_result.append(cross_entropy.data.cpu().numpy())
            kld_result.append(kld.data.cpu().numpy() * coef)

    t.save(rvae.state_dict(), save_path + f"trained_RVAE{iteration}")
    np.save(save_path + f"ce_result.npy".format(args.ce_result), np.array(ce_result))
    np.save(save_path + f"kld_result_npy".format(args.kld_result), np.array(kld_result))
