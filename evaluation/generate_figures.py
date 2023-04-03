""""
This file is used to generate the figures in the report.
"""

import json

import matplotlib.pyplot as plt

files = [
    {
        "file": "evaluation/mnist_conv_evaluation_0.json",
        "model": "mnist_conv",
        "specific_output": 0,
    },
    {
        "file": "evaluation/fashionmnist_evaluation_0.json",
        "model": "fashionmnist",
        "specific_output": 0,
    },
    {
        "file": "evaluation/imdb_evaluation_0.json",
        "model": "imdb",
        "specific_output": 0,
    },
]


for file in files:
    with open(file["file"], "r") as f:
        data = json.load(f)

    fig, ax = plt.subplots()

    for i in range(len(data["original_model_accuracy"])):
        ax.plot(
            [i, i],
            [data["original_model_accuracy"][i], data["patched_model_accuracy"][i]],
            color="black",
            linewidth=1,
        )

    ax.scatter(
        range(len(data["original_model_accuracy"])),
        data["original_model_accuracy"],
        label="Original Accuracy",
        facecolors="none",
        edgecolors="blue",
    )
    ax.scatter(
        range(len(data["patched_model_accuracy"])),
        data["patched_model_accuracy"],
        label="Patched Accuracy",
        facecolors="none",
        edgecolors="red",
    )

    ax.set_title(
        f'Mutant Accuracy Comparison: {file["model"]} on outputs {file["specific_output"]}'
    )
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Accuracy")
    ax.legend()

    ax.set_ylim(0, 1)

    plt.savefig(f"evaluation/{file['model']}_evaluation_{file['specific_output']}.pdf")

    # Plot trivial mutants

    fig, ax = plt.subplots()

    for i in range(len(data["original_model_accuracy_trivial"])):
        ax.plot(
            [i, i],
            [
                data["original_model_accuracy_trivial"][i],
                data["patched_model_accuracy_trivial"][i],
            ],
            color="black",
            linewidth=1,
        )

    ax.scatter(
        range(len(data["original_model_accuracy_trivial"])),
        data["original_model_accuracy_trivial"],
        label="Original Accuracy",
        facecolors="none",
        edgecolors="blue",
    )
    ax.scatter(
        range(len(data["patched_model_accuracy_trivial"])),
        data["patched_model_accuracy_trivial"],
        label="Patched Accuracy",
        facecolors="none",
        edgecolors="red",
    )

    ax.set_title(
        f'Mutant Accuracy Comparison: {file["model"]} on outputs other than {file["specific_output"]}'
    )
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Accuracy")

    ax.legend()

    plt.savefig(
        f"evaluation/{file['model']}_evaluation_{file['specific_output']}_trivial.pdf"
    )

# Plot time to generate mutants

fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

for i, file in enumerate(files):
    with open(file["file"], "r") as f:
        data = json.load(f)

    axs[i].boxplot(data["time_to_generate"])
    axs[i].set_title(f"{file['model']} on output {file['specific_output']}")
    axs[i].set_xlabel("Sample Index")
    axs[i].set_ylabel("Time (s)")

fig.suptitle("Time to generate mutants")

plt.savefig("evaluation/time_to_generate_boxplots.pdf")
