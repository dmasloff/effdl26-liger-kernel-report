import os
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np


SEQ_LEN = 1024
BATCH_SIZE = [1, 2, 4, 8, 12, 16, 24, 32]
CE_LOSS = ["torch", "liger", "ligerfusedlinear"]
MODEL_TYPE = ["hf", "liger", "liger__tiled"]
SHRINK_LOWEST = 5


colors = {
    "loss=torch/model=hf": "lightsteelblue",
    "loss=liger/model=hf": "cornflowerblue",
    "loss=ligerfusedlinear/model=hf": "royalblue",
    "loss=torch/model=liger": "lightcoral",
    "loss=liger/model=liger": "firebrick",
    "loss=ligerfusedlinear/model=liger": "darkred",
    "loss=ligerfusedlinear/model=liger__tiled_mlp": "green",
}


if __name__ == "__main__":
    os.makedirs("./imgs", exist_ok=True)

    memory_stats_dict = defaultdict(list)
    loss_stats_dict = defaultdict(list)
    timing_stats_dict = defaultdict(list)


    for bs in BATCH_SIZE:
        for loss in CE_LOSS:
            for model in MODEL_TYPE:
                try:
                    stats = torch.load(f"logs/LlamaForCausalLM__bs_{bs}__ce_{loss}__model_{model}.pt")
                    memory_stats_dict[f"loss={loss}/model={model}"].append(stats["max_memory_allocated"])
                    loss_stats_dict[f"loss={loss}/model={model}"].append(stats["losses"])
                    timing_stats_dict[f"loss={loss}/model={model}"].append(stats["timings"])
                except Exception as e:
                    pass


    plt.subplots(1, 3, figsize=(24, 7))

    plt.subplot(1, 3, 1)
    for loss in CE_LOSS:
        for model in MODEL_TYPE:
            label = f"loss={loss}/model={model}"
            if not label in timing_stats_dict:
                continue
        
            values = timing_stats_dict[label]
            values = np.array(values)
            values = np.sort(values, axis=1)
            values = np.mean(values[:, SHRINK_LOWEST:], axis=1)
            keys = BATCH_SIZE[:len(values)]
            if label.endswith("tiled"):
                label += "_mlp"

            plt.plot(values, color=colors[label])
            plt.scatter(np.arange(len(values)), values, label=label, color=colors[label])

    plt.grid()
    plt.legend()
    plt.xlabel("Batch size")
    plt.xticks(np.arange(len(BATCH_SIZE)), BATCH_SIZE)
    plt.title("AVG Iter time (ms)")

    plt.subplot(1, 3, 2)
    for loss in CE_LOSS:
        for model in MODEL_TYPE:
            label = f"loss={loss}/model={model}"
            if not label in timing_stats_dict:
                continue

            values = timing_stats_dict[label]
            values = np.array(values)
            iters = values.shape[1]
            values = np.sort(values, axis=1)[:, SHRINK_LOWEST:]
            values = np.sum(values, axis=1)
            keys = np.array(BATCH_SIZE[:len(values)])
            throughput = iters * keys * SEQ_LEN * 1000 / values
            if label.endswith("tiled"):
                label += "_mlp"
            
            plt.plot(throughput, color=colors[label])
            plt.scatter(np.arange(len(values)), throughput, label=label, color=colors[label])

    plt.grid()
    plt.legend()
    plt.xlabel("Batch size")
    plt.xticks(np.arange(len(BATCH_SIZE)), BATCH_SIZE)
    plt.title("Throughput (toks/s)")

    plt.subplot(1, 3, 3)
    for loss in CE_LOSS:
        for model in MODEL_TYPE:
            label = f"loss={loss}/model={model}"
            if not label in memory_stats_dict:
                continue
        
            values = np.array(memory_stats_dict[label]) / (1000**3)
            keys = BATCH_SIZE[:len(values)]
            if label.endswith("tiled"):
                label += "_mlp"

            plt.plot(values, color=colors[label])
            plt.scatter(np.arange(len(values)), values, label=label, color=colors[label])
        
    plt.grid()
    plt.legend()
    plt.xlabel("Batch size")
    plt.xticks(np.arange(len(BATCH_SIZE)), BATCH_SIZE)
    plt.title("torch.cuda.max_memory_allocated (GiB)")

    plt.suptitle(f"unsloth/Llama-3-8.1B FP16 Train with GradScaling, SEQ_LEN={SEQ_LEN}, GPU H200")

    plt.show()
    plt.savefig("imgs/perf_uniform_x_axis.png")


    plt.subplots(1, 3, figsize=(24, 7))

    plt.subplot(1, 3, 1)
    for loss in CE_LOSS:
        for model in MODEL_TYPE:
            label = f"loss={loss}/model={model}"
            if not label in timing_stats_dict:
                continue
        
            values = timing_stats_dict[label]
            values = np.array(values)
            values = np.sort(values, axis=1)
            values = np.mean(values[:, SHRINK_LOWEST:], axis=1)
            keys = BATCH_SIZE[:len(values)]
            if label.endswith("tiled"):
                label += "_mlp"

            plt.plot(keys, values, color=colors[label])
            plt.scatter(keys, values, label=label, color=colors[label])

    plt.grid()
    plt.legend()
    plt.xlabel("Batch size")
    plt.title("AVG Iter time (ms)")

    plt.subplot(1, 3, 2)
    for loss in CE_LOSS:
        for model in MODEL_TYPE:
            label = f"loss={loss}/model={model}"
            if not label in timing_stats_dict:
                continue

            values = timing_stats_dict[label]
            values = np.array(values)
            iters = values.shape[1]
            values = np.sort(values, axis=1)[:, SHRINK_LOWEST:]
            values = np.sum(values, axis=1)
            keys = np.array(BATCH_SIZE[:len(values)])
            throughput = iters * keys * SEQ_LEN * 1000 / values
            if label.endswith("tiled"):
                label += "_mlp"
            
            plt.plot(keys, throughput, color=colors[label])
            plt.scatter(keys, throughput, label=label, color=colors[label])

    plt.grid()
    plt.legend()
    plt.xlabel("Batch size")
    plt.title("Throughput (toks/s)")

    plt.subplot(1, 3, 3)
    for loss in CE_LOSS:
        for model in MODEL_TYPE:
            label = f"loss={loss}/model={model}"
            if not label in memory_stats_dict:
                continue
        
            values = np.array(memory_stats_dict[label]) / (2**30)
            keys = BATCH_SIZE[:len(values)]
            if label.endswith("tiled"):
                label += "_mlp"

            plt.plot(keys, values, color=colors[label])
            plt.scatter(keys, values, label=label, color=colors[label])
        
    plt.grid()
    plt.legend()
    plt.xlabel("Batch size")
    plt.title("torch.cuda.max_memory_allocated (GB)")

    plt.suptitle(f"unsloth/Llama-3-8.1B FP16 Train with GradScaling, SEQ_LEN={SEQ_LEN}, GPU H200")

    plt.show()

    plt.savefig("imgs/perf_real_x_axis.png")
