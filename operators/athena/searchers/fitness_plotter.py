import logging
from multiprocessing import Process, Queue
from typing import List


class FitnessPlotter(Process):
    """
    Plot fitness over time in a separate process.
    """

    def __init__(self, queue: Queue) -> None:
        super().__init__()
        self.queue = queue
        self.fitnesses = []

    def run(self) -> None:
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        import matplotlib.pyplot as plt

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fitness")
        ax.set_title("Fitness over time")
        plt.show()

        while True:
            fitness = self.queue.get()
            if fitness is None:
                break
            self.fitnesses.append(fitness)
            ax.plot(self.fitnesses, "b-")
            fig.canvas.draw()
            fig.canvas.flush_events()

    def update(self, fitness: List[float]) -> None:
        self.queue.put(fitness)
