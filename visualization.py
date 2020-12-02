import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")

import os


def run():
    for model_type in ["base", "adam", "cyclical", "hrvae", "wae", "stacked"]:
        for data_type in ["quora", "coco"]:
            path = os.path.join("evaluation", "data", model_type, data_type)
            make_loss_plots(path)
            make_metric_plot(path, "")


def make_loss_plots(path):
    partial = path.split("/")
    data_kld = np.load(os.path.join(path, "kld_result.npy"))
    data_ce = np.load(os.path.join(path, "ce_result.npy"))
    df_loss = pd.DataFrame(list(zip(data_ce, data_kld)), columns=["ce_loss", "kld_loss"])
    df_loss["iteration"] = df_loss.index * 1000 + 1000
    df_loss = df_loss[df_loss["iteration"] <= 120000]

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    x = ax1.plot(df_loss["iteration"], df_loss["ce_loss"], "--", color="black", label="CE loss")
    y = ax2.plot(df_loss["iteration"], df_loss["kld_loss"], "-", color="black", label="KL loss")

    ax1.set_xlabel("iteration")
    ax1.set_ylabel("Cross Entropy (CE) loss")
    ax2.set_ylabel("Kullback–Leibler (KL) loss")

    plt.title(f"Model Loss Evaluation Metrics: {partial[-2]} {partial[-1]}")
    lns = x + y
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.savefig(os.path.join("/".join(partial[:-1]), f"loss_values_{partial[-2]}_{partial[-1]}"), bbox_inches="tight")
    plt.clf()


def make_metric_plot(path, metric_type):
    partial = path.split("/")
    eval_types = ["blue", "ter", "muse", "meteor", "rouge"]
    data_blue = np.load(os.path.join(path, f"blue_result{metric_type}.npy"))
    data_ter = np.load(os.path.join(path, f"ter_result{metric_type}.npy"))
    data_muse = np.load(os.path.join(path, f"muse_result{metric_type}.npy"))
    data_meteor = np.load(os.path.join(path, f"meteor_result{metric_type}.npy"))
    data_rouge = np.load(os.path.join(path, f"rouge_result{metric_type}.npy"))
    df_eval = pd.DataFrame(list(zip(data_blue, data_ter, data_muse, data_meteor, data_rouge)), columns=eval_types)
    df_eval_start = pd.DataFrame([[0, 1, 0, 0, 0]], columns=eval_types)
    df_eval["iteration"] = df_eval.index * 10000 + 10000
    df_eval_start["iteration"] = df_eval_start.index
    df_eval_start = df_eval_start.append(df_eval)
    df_eval = df_eval_start
    eval_types = ["ter", "blue", "muse", "meteor", "rouge"]
    optim_type = ["\u2193", "\u2191", "\u2191", "\u2191", "\u2191"]  # up and down
    for idx in range(len(eval_types)):
        plt.plot(df_eval_start["iteration"], df_eval[eval_types[idx]], label=eval_types[idx] + " " + optim_type[idx])
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.yticks(np.arange(0, 1.01, step=0.1))
    # plt.xticks(np.arange(0, 120000, step=20000), rotation=20)
    plt.xlabel("iteration")
    plt.ylabel("metric score")
    plt.title(f"Paraphrase Evaluation Metrics: {partial[-2]} {partial[-1]}")
    plt.savefig(
        os.path.join("/".join(partial[:-1]), f"parapharase_evaluation_{partial[-2]}_{partial[-1]}"), bbox_inches="tight"
    )
    plt.clf()
    print(partial)
    print(df_eval[["blue", "muse", "meteor", "rouge"]].max(), df_eval[["ter"]].min())


if __name__ == "main":
    run()
# make_metric_plot(path, '_std')
