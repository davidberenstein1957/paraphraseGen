import argparse
import os
import random
import statistics
import time

import numpy as np
import torch as t
from six.moves import cPickle
from torch.autograd import Variable

from evaluation.paraphrase_evaluation import get_evaluation_scores
from model.rvae_previous import RVAE
from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from utils.tensor import PreProcessor

if __name__ == "__main__":

    # assert os.path.exists("./trained_RVAE"), "trained model not found"
    save_path = "/content/drive/My Drive/thesis/results/"
    path = "paraphraseGen/"
    parser = argparse.ArgumentParser(description="Sampler")
    parser.add_argument("--use-cuda", type=bool, default=True, metavar="CUDA", help="use cuda (default: True)")
    parser.add_argument("--num-sample", type=int, default=5, metavar="NS", help="num samplings (default: 5)")
    parser.add_argument("--num-sentence", type=int, default=10, metavar="NS", help="num samplings (default: 10)")
    parser.add_argument("--beam-top", type=int, default=1, metavar="NS", help="beam top (default: 1)")
    parser.add_argument("--beam-size", type=int, default=100, metavar="NS", help="beam size (default: 10)")
    parser.add_argument("--use-file", type=bool, default=True, metavar="NS", help="use file (default: False)")
    parser.add_argument("--adam", type=bool, default=False)
    parser.add_argument("--hrvae", type=bool, default=False)
    parser.add_argument("--annealing", type=str, default="mono")  # none, mono, cyc
    parser.add_argument("--use-trained", type=bool, default=False)
    parser.add_argument("--attn-model", type=bool, default=False)
    parser.add_argument("--res-model", type=bool, default=False)
    parser.add_argument("--wae", type=bool, default=False)
    parser.add_argument("--data-name", type=str, default="quora")  # quora, coco, both
    parser.add_argument("--embeddings-name", type=str, default="quora")  # quora, coco, both
    parser.add_argument("--test-type", type=str, default="final")  # quora, coco, both

    if parser.parse_args().test_type == "final":
        parser.add_argument(
            "--test-file",
            type=str,
            default=path + f"data/test_{parser.parse_args().data_name}.txt",
            metavar="NS",
            help="test file path (default: data/test.txt)",
        )
    else:
        parser.add_argument(
            "--test-file",
            type=str,
            default=path + f"data/test_final_{parser.parse_args().data_name}.txt",
            metavar="NS",
            help="test file path (default: data/test.txt)",
        )
    # parser.add_argument(
    #     "--test-file",
    #     type=str,
    #     default=path + f"data/test_final_{parser.parse_args().data_name}.txt",
    #     metavar="NS",
    #     help="test file path (default: data/test.txt)"
    # )

    args = parser.parse_args()

    if args.res_model:
        save_path = os.path.join(save_path, "stacked")
    elif args.embeddings_name == "both":
        save_path = os.path.join(save_path, "word2vec")
    elif args.wae:
        save_path = os.path.join(save_path, "wae")
    elif args.hrvae:
        save_path = os.path.join(save_path, "hrvae")
    elif args.annealing == "cyc":
        save_path = os.path.join(save_path, "cyclical")
    elif args.adam:
        save_path = os.path.join(save_path, "adam")
    else:
        save_path = os.path.join(save_path, "base")

    if args.data_name == "quora":
        save_path = os.path.join(save_path, "quora")
    elif args.data_name == "coco":
        save_path = os.path.join(save_path, "coco")
    else:
        save_path = os.path.join(save_path, "both")

    str = ""
    if not args.use_file:
        str = raw_input("Input Question : ")
    else:
        file_1 = open(args.test_file, "r")
        data = file_1.readlines()

    """ ================================= BatchLoader loading ===============================================
    """
    data_files = [args.test_file]

    idx_files = [
        path + f"data/words_vocab_{args.embeddings_name}.pkl",
        path + f"data/characters_vocab_{args.embeddings_name}.pkl",
    ]

    tensor_files = [
        [path + f"data/test_word_tensor_{args.embeddings_name}.npy"],
        [path + f"data/test_character_tensor_{args.embeddings_name}.npy"],
    ]

    preprocessor = PreProcessor(idx_files)
    preprocessor.preprocess_data(data_files, idx_files, tensor_files, args.use_file, str)

    batch_loader = BatchLoader(data_files, idx_files, tensor_files)
    parameters = Parameters(
        batch_loader.max_word_len,
        batch_loader.max_seq_len,
        batch_loader.words_vocab_size,
        batch_loader.chars_vocab_size,
        args.embeddings_name,
        args.res_model,
        args.hrvae,
        args.wae,
    )

    """ ============================ BatchLoader for Question-2 ===============================================
    """
    data_files = [path + f"data/super/train_{args.data_name}_2.txt"]

    idx_files = [
        path + f"data/super/words_vocab_{args.embeddings_name}_2.pkl",
        path + f"data/super/characters_vocab_{args.embeddings_name}_2.pkl",
    ]

    tensor_files = [
        [path + f"data/super/train_word_tensor_{args.embeddings_name}_2.npy"],
        [path + f"data/super/train_character_tensor_{args.embeddings_name}_2.npy"],
    ]
    batch_loader_2 = BatchLoader(data_files, idx_files, tensor_files)
    parameters_2 = Parameters(
        batch_loader_2.max_word_len,
        batch_loader_2.max_seq_len,
        batch_loader_2.words_vocab_size,
        batch_loader_2.chars_vocab_size,
        args.embeddings_name,
        args.res_model,
        args.hrvae,
        args.wae,
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
    print("Started loading", save_path)
    start_time = time.time()
    rvae = RVAE(parameters, parameters_2, path)
    num_training_iters = 120000
    coef_modulo = 10000

    meteor_result = []
    blue_result = []
    rouge_result = []
    ter_result = []
    muse_result = []
    meteor_result_std = []
    blue_result_std = []
    rouge_result_std = []
    ter_result_std = []
    muse_result_std = []
    # for i in range(1, int(120000 / 10000)+1):

    for i in range(1):
        # model_state = i * coef_modulo
        model_state = 120000
        rvae.load_state_dict(t.load(save_path + f"/trained_RVAE_{model_state}"))
        if args.use_cuda:
            rvae = rvae.cuda()
        loading_time = time.time() - start_time
        print(f"Time elapsed in loading model {model_state} =", loading_time)
        print("Finished loading")

        hyp__ = []
        ref_ = []
        for j in range(2):
            # j = random.randint(0, len(data))
            ref_.append(data[j].replace("\n", ""))
            print(data[j])
            hyp_ = []

            for iteration in range(args.num_sample):
                seed = Variable(t.randn([1, parameters.latent_variable_size]))
                seed = seed.cuda()

                results, scores = rvae.sampler(
                    batch_loader, batch_loader_2, 50, seed, args.use_cuda, j, beam_size, n_best
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

        # scores = get_evaluation_scores(hyp__, ref_)
        # meteor_result.append(statistics.mean(scores["METEOR"]))
        # blue_result.append(statistics.mean(scores["BLUE"]))
        # rouge_result.append(statistics.mean(scores["ROUGE"]))
        # ter_result.append(statistics.mean(scores["TER"]))
        # muse_result.append(statistics.mean(scores["MUSE"]))

        # meteor_result_std.append(statistics.variance(scores["METEOR"]))
        # blue_result_std.append(statistics.variance(scores["BLUE"]))
        # rouge_result_std.append(statistics.variance(scores["ROUGE"]))
        # ter_result_std.append(statistics.variance(scores["TER"]))
        # muse_result_std.append(statistics.variance(scores["MUSE"]))

    # np.save(save_path + f"/meteor_result.npy", np.array(meteor_result))
    # np.save(save_path + f"/blue_result.npy", np.array(blue_result))
    # np.save(save_path + f"/rouge_result.npy", np.array(rouge_result))
    # np.save(save_path + f"/ter_result.npy", np.array(ter_result))
    # np.save(save_path + f"/muse_result.npy", np.array(muse_result))

    # np.save(save_path + f"/meteor_result_std.npy", np.array(meteor_result_std))
    # np.save(save_path + f"/blue_result_std.npy", np.array(blue_result_std))
    # np.save(save_path + f"/rouge_result_std.npy", np.array(rouge_result_std))
    # np.save(save_path + f"/ter_result_std.npy", np.array(ter_result_std))
    # np.save(save_path + f"/muse_result_std.npy", np.array(muse_result_std))
