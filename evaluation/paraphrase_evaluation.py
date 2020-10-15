import pyter
import tensorflow as tf
import tensorflow_hub as hub
from nltk.translate import bleu_score, meteor_score
from rouge import Rouge


def get_evaluation_scores(hypothesis: list[list[str]], reference: list[str]) -> dict:
    score_overview = {}
    score_overview['METEOR'] = get_blue_score(hypothesis, reference)
    score_overview['BLUE'] = get_meteor_score(hypothesis, reference)
    score_overview['ROUGUE'] = get_rougue_score(hypothesis, reference)
    score_overview['TER'] = get_ter_score(hypothesis, reference)
    score_overview['MUSE'] = get_muse_score(hypothesis, reference)

    return score_overview


def get_blue_score(hypothesis: list[list[str]], reference: list[str]) -> list:
    blue_score_list = []
    for (hyp, ref) in zip(hypothesis, reference):
        blue_score = bleu_score(hyp, ref)
        blue_score_list.append(blue_score)

    return blue_score_list


def get_meteor_score(hypothesis: list[list[str]], reference: list[str]) -> list:
    meteor_score_list = []
    for (hyp, ref) in zip(hypothesis, reference):
        meteor_score = meteor_score(hyp, ref)
        meteor_score_list.append(meteor_score)

    return meteor_score_list


def get_rougue_score(hypothesis: list[list[str]], reference: list[str]) -> list:
    rouge = Rouge()
    rouge_score_list = []
    for (hyps, ref) in zip(hypothesis, reference):
        rouge_score = 0
        for hyp_n in hyps:
            rouge_score += rouge.get_scores(hyp_n, ref)['rouge-l']['f']
        rouge_score = rouge_score / len(hyps)

    return rouge_score_list


def get_ter_score(hypothesis: list[list[str]], reference: list[str]) -> list:
    ter_score_list = []
    for (hyps, ref) in zip(hypothesis, reference):
        ter_score = 0
        for hyp_n in hyps:
            ter_score += pyter.ter(hyp_n, ref)
        ter_score = ter_score / len(hyps)
        ter_score_list.append(ter_score)

    return ter_score_list


def get_muse_score(hypothesis: list[list[str]], reference: list[str]) -> list:
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
    reference = embed(reference)
    sim_score_list = []
    for (hyps, ref) in zip(hypothesis, reference):
        hyps = embed(hyps)
        sim_score = 0
        for hyp_n in hyps:
            sim_score += cosine_loss(hyp_n, ref).numpy()
        sim_score = sim_score / len(hyps)
        sim_score_list.append(sim_score)

    return sim_score_list
