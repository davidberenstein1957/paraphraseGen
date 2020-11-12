from typing import List

import pyter
import tensorflow as tf
import tensorflow_hub as hub
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from numpy import dot
from numpy.linalg import norm
from rouge import Rouge
tf.enable_eager_execution() 


def get_evaluation_scores(hypothesis: List[List[str]], reference: List[str]) -> dict:
    try:
        nltk.download('wordnet')
    except:
        pass
    score_overview = dict.fromkeys(["METEOR", "BLUE", "ROUGE", "TER", "MUSE"], [])
    score_overview["BLUE"] = get_blue_score(hypothesis, reference)
    score_overview["METEOR"] = get_meteor_score(hypothesis, reference)
    score_overview["ROUGE"] = get_rougue_score(hypothesis, reference)
    score_overview["TER"] = get_ter_score(hypothesis, reference)
    score_overview["MUSE"] = get_muse_score(hypothesis, reference)

    return score_overview


def get_blue_score(hypothesis: List[List[str]], reference: List[str]) -> list:
    blue_score_list = []
    for hyp, ref in list(zip(hypothesis, reference)):
        b_score = sentence_bleu(hyp, ref)
        blue_score_list.append(b_score)

    return blue_score_list


def get_meteor_score(hypothesis: List[List[str]], reference: List[str]) -> list:
    meteor_score_list = []
    for (hyp, ref) in list(zip(hypothesis, reference)):
        m_score = meteor_score(hyp, ref)
        meteor_score_list.append(m_score)

    return meteor_score_list


def get_rougue_score(hypothesis: List[List[str]], reference: List[str]) -> list:
    rouge = Rouge()
    rouge_score_list = []
    for (hyps, ref) in list(zip(hypothesis, reference)):
        rouge_score = 0
        for hyp_n in hyps:
            rouge_score += rouge.get_scores(hyp_n, ref)[0]["rouge-l"]["f"]
        rouge_score = rouge_score / len(hyps)
        rouge_score_list.append(rouge_score)

    return rouge_score_list


def get_ter_score(hypothesis: List[List[str]], reference: List[str]) -> list:
    ter_score_list = []
    for (hyps, ref) in zip(hypothesis, reference):
        ter_score = 0
        for hyp_n in hyps:
            ter_score += pyter.ter(hyp_n, ref)
        ter_score = ter_score / len(hyps)
        ter_score_list.append(ter_score)

    return ter_score_list


def get_muse_score(hypothesis: List[List[str]], reference: List[str]) -> list:
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    reference_embedding = list(embed(reference).numpy())
    sim_score_list = []
    for (hyps, ref) in zip(hypothesis, reference_embedding):
        hyps_embedding = list(embed(hyps).numpy())
        sim_score = 0
        for hyp_n in hyps_embedding:
            sim_score += dot(hyp_n, ref) / (norm(hyp_n) * norm(ref))
        sim_score = sim_score / len(hyps)
        sim_score_list.append(sim_score)

    return sim_score_list
