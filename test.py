import argparse
import os
import statistics
import time

import numpy as np
import torch as t
from six.moves import cPickle
from torch.autograd import Variable

from evaluation.paraphrase_evaluation import get_evaluation_scores
from model.rvae import RVAE
from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from utils.tensor import preprocess_data

if __name__ == "__main__":

    # assert os.path.exists("./trained_RVAE"), "trained model not found"
    save_path = "/content/drive/My Drive/thesis/"
    path = "paraphraseGen/"
    parser = argparse.ArgumentParser(description="Sampler")
    parser.add_argument("--use-cuda", type=bool, default=True, metavar="CUDA", help="use cuda (default: True)")
    parser.add_argument("--num-sample", type=int, default=5, metavar="NS", help="num samplings (default: 5)")
    parser.add_argument("--num-sentence", type=int, default=10, metavar="NS", help="num samplings (default: 10)")
    parser.add_argument("--beam-top", type=int, default=3, metavar="NS", help="beam top (default: 1)")
    parser.add_argument("--beam-size", type=int, default=10, metavar="NS", help="beam size (default: 10)")
    parser.add_argument("--use-file", type=bool, default=True, metavar="NS", help="use file (default: False)")
    # Path to test file ---
    parser.add_argument(
        "--test-file", type=str, default=path + "data/test.txt", metavar="NS", help="test file path (default: data/test.txt)"
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default="./trained_RVAE",
        metavar="NS",
        help="trained model save path (default: ./trained_RVAE)",
    )
    args = parser.parse_args()

    # Removing, is already some previous files exist from last execution of program
    if os.path.exists("data/test_word_tensor.npy"):
        os.remove("data/test_word_tensor.npy")
    if os.path.exists("data/test_character_tensor.npy"):
        os.remove("data/test_character_tensor.npy")

    str = ""
    if not args.use_file:
        str = raw_input("Input Question : ")
    else:
        file_1 = open(args.test_file, "r")
        data = file_1.readlines()

    """ ================================= BatchLoader loading ===============================================
    """
    data_files = [args.test_file]

    idx_files = [path + "data/words_vocab.pkl", path + "data/characters_vocab.pkl"]

    tensor_files = [[path + "data/test_word_tensor.npy"], [path + "data/test_character_tensor.npy"]]

    preprocess_data(data_files, idx_files, tensor_files, args.use_file, str)

    batch_loader = BatchLoader(data_files, idx_files, tensor_files)
    parameters = Parameters(
        batch_loader.max_word_len,
        batch_loader.max_seq_len,
        batch_loader.words_vocab_size,
        batch_loader.chars_vocab_size,
    )

    """ ============================ BatchLoader for Question-2 ===============================================
    """
    data_files = [path + "data/super/train_2.txt"]

    idx_files = [path + "data/super/words_vocab_2.pkl", path + "data/super/characters_vocab_2.pkl"]

    tensor_files = [[path + "data/super/train_word_tensor_2.npy"], [path + "data/super/train_character_tensor_2.npy"]]
    batch_loader_2 = BatchLoader(data_files, idx_files, tensor_files)
    parameters_2 = Parameters(
        batch_loader_2.max_word_len,
        batch_loader_2.max_seq_len,
        batch_loader_2.words_vocab_size,
        batch_loader_2.chars_vocab_size,
    )

    """ ==================================== Parameters Initialising ===========================================
        """
    n_best = args.beam_top
    beam_size = args.beam_size

    assert n_best <= beam_size
    use_cuda = args.use_cuda

    if args.use_file:
        num_sentence = args.num_sentence
    else:
        num_sentence = 1

    """ ======================================================================================================= """

    """======================================== MODEL loading ================================================= """
    print("Started loading")
    start_time = time.time()
    rvae = RVAE(parameters, parameters_2)
    num_training_iters = 120000
    coef_modulo = 10000

    meteor_result = []
    blue_result = []
    rouge_result = []
    ter_result = []
    muse_result = []

    for i in range(1, int(120000 / 10000)):
        model_state = i * coef_modulo

        rvae.load_state_dict(t.load(save_path + f"/trained_RVAE_{model_state}"))
        if args.use_cuda:
            rvae = rvae.cuda()
        loading_time = time.time() - start_time
        print(f"Time elapsed in loading model {model_state} =", loading_time)
        print("Finished loading")

        hyp__ = []
        ref_ = []
        for i in range(len(data)):
            if args.use_file:
                print("original sentence:     " + data[i])
            else:
                print("original sentence:     " + str + "\n")
            ref_.append(data[i])
            hyp_ = []
            for iteration in range(args.num_sample):
                seed = Variable(t.randn([1, parameters.latent_variable_size]))
                seed = seed.cuda()

                results, scores = rvae.sampler(
                    batch_loader, batch_loader_2, 50, seed, args.use_cuda, i, beam_size, n_best
                )

                for tt in results:
                    for k in range(n_best):
                        sen = " ".join([batch_loader_2.decode_word(x[k]) for x in tt])
                        if batch_loader.end_token in sen:
                            print("generate sentence:     " + sen[: sen.index(batch_loader.end_token)])
                            hyp_.append(sen[: sen.index(batch_loader.end_token)])
                        else:
                            print("generate sentence:     " + sen)
                            hyp_.append(sen)
            hyp__.append(hyp_)
            print("\n")

        scores.get_evaluation_scores(hyp__, ref_)
        meteor_result.append(statistics.mean(scores["METEOR"]))
        blue_result.append(statistics.mean(scores["BLUE"]))
        rouge_result.append(statistics.mean(scores["ROUGE"]))
        ter_result.append(statistics.mean(scores["TER"]))
        muse_result.append(statistics.mean(scores["MUSE"]))
