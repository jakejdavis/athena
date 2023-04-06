# Athena

Athena is a novel post-training mutation operator for deep learning models. It aims to inject relevant faults into trained models by leveraging [Arachne](https://arxiv.org/abs/1912.12463), search based repair technique for neural networks, to find the most impactful weights given a set of inputs/outputs to degrade.

To accompany the mutation operator, a command-line interface is provided to users as a comprehensive tool for running mutation testing operations on deep learning models. This tool provides commands to generate mutants, run a mutation test on a test set, and evaluate a given mutation operator. The tool is designed to be easily extensible to support new test sets, mutation operators, and models.

To get started, see the [User Guide](user_guide.pdf).