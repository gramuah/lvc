import json
import matplotlib.pyplot as plt
import os
import numpy as np

def single_video_compare(cwd, data, metrics, video_id, frame):
    video_data = data[video_id]
    results = video_data["results"]
    results_h_no_w = video_data["results_h_no_w"]
    results_w = video_data["results_w"]
    results_h = video_data["results_h"]
    fig, axs = plt.subplots(3, 1, figsize=(10, 6))
    axs = axs.flatten()
    for i, metric in enumerate(metrics):
        values = results[metric]
        hnw_values = results_h_no_w[metric]
        w_values = results_w[metric]
        h_values = results_h[metric]
        x = np.arange(1, len(w_values) + 1)
        axs[i].plot(x, values, label="Normal")
        axs[i].plot(x, w_values, label="Weighted + no History window")
        axs[i].plot(x, hnw_values, label="No weighted + History window")
        axs[i].plot(x, h_values, label="History window + weighted")
        axs[i].set_title(metric)
        axs[i].set_ylabel("Value")
        if len(w_values) < 15:
            axs[i].set_xticks(x)
        else:
            axs[i].set_xticks(np.arange(0, len(w_values) + 1, 5))
        axs[i].legend()
        axs[i].set_xlabel("Number of " + r"$\Delta t$")
    plt.suptitle(f"Video {video_id}")
    results_dir_svg = os.path.join(cwd, f"results/images/{frame}", "svg/w_vs_h")
    results_dir_png = os.path.join(cwd, f"results/images/{frame}", "png/w_vs_h")
    os.makedirs(results_dir_svg, exist_ok=True)
    os.makedirs(results_dir_png, exist_ok=True)
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir_png, f"{video_id}.png"))
    fig.savefig(os.path.join(results_dir_svg, f"{video_id}.svg"))


def all_video_plot(cwd, data, metrics, frame):
    for video_id in data.keys():
        if not data[video_id]:
            continue
        single_video_compare(cwd, data, metrics, video_id, frame)

cwd = os.path.abspath(os.path.dirname(__file__))
json_dir = "results/json"
images_dir = "results/images"
if not os.path.exists(os.path.join(cwd, images_dir)):
    os.makedirs(os.path.join(cwd, images_dir))
if not os.path.exists(os.path.join(cwd, json_dir)):
    os.makedirs(os.path.join(cwd, json_dir))
    
    
json_list = os.listdir(os.path.join(cwd, json_dir))
metrics = ["Bleu_4", "METEOR", "ROUGE_L"]

frames_values = [150]
for frame in frames_values:
    json_path = os.path.join(cwd, f"results/json/results_{frame}.json")
    with open(json_path, "r") as f:
        data = json.load(f)
    all_video_plot(cwd, data, metrics, frame)
