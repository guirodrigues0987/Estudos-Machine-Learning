"""Neuroevolution of a Super Mario Bros agent with NEAT-Python.

Runs a population of feed-forward networks on level 1-1, keeps training
checkpoints so a run can be resumed, and saves the best genome at the end.
"""

import os
import pickle

import cv2
import gym_super_mario_bros
import neat
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

CONFIG_PATH = "config-feedforward"
GENERATION_ZERO_CHECKPOINT = "checkpoint_generation_zero"
LATEST_CHECKPOINT = "checkpoint_latest"
BEST_GENOME_PATH = "best_mario.pkl"
CHECKPOINT_INTERVAL = 5  # generations between rolling checkpoints
STALL_LIMIT = 250  # frames without fitness progress (~4 seconds) before giving up


# --- EVALUATION FUNCTION (where Mario plays) ---
def eval_genomes(genomes, config):
    for _genome_id, genome in genomes:
        # Create the environment (compatible with the newer gym API)
        env = gym_super_mario_bros.make(
            "SuperMarioBros-1-1-v0", render_mode="human", apply_api_compatibility=True
        )
        env = JoypadSpace(env, SIMPLE_MOVEMENT)

        ob, info = env.reset()
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        current_max_fitness = 0
        fitness_current = 0
        counter = 0  # counts frames without progress to detect a stuck Mario

        done = False
        while not done:
            # 1. Vision processing (downscale to 784 inputs: 28x28 grayscale)
            ob = cv2.resize(ob, (28, 28))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            inputs = ob.flatten()

            # 2. The network picks the action
            output = net.activate(inputs)
            action = output.index(max(output))

            # 3. Execute the action in the game
            ob, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 4. Update fitness (distance traveled)
            fitness_current += reward

            # End the run if Mario stays still (avoids infinite loops)
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            # Mario died, or stalled for STALL_LIMIT frames
            if done or counter == STALL_LIMIT:
                done = True
                genome.fitness = fitness_current

        env.close()


# --- MANUAL CHECKPOINTING ---
class SmartCheckpointer(neat.reporting.BaseReporter):
    """Saves generation 0 and a rolling checkpoint every CHECKPOINT_INTERVAL generations."""

    def __init__(self):
        self.current_generation = 0

    def start_generation(self, generation):
        # Keep the current generation in sync with NEAT
        self.current_generation = generation

    def end_generation(self, config, population, species_set):
        local_dir = os.path.dirname(os.path.abspath(__file__))

        checkpoint_path = None
        # Save generation 0 (starting point)
        if self.current_generation == 0:
            checkpoint_path = os.path.join(local_dir, GENERATION_ZERO_CHECKPOINT)
            print("\n[SYSTEM] Saving generation zero checkpoint...")

        # Rolling backup every CHECKPOINT_INTERVAL generations
        elif self.current_generation % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(local_dir, LATEST_CHECKPOINT)
            print(f"\n[SYSTEM] Updating backup (generation {self.current_generation})...")

        if checkpoint_path:
            try:
                # Manual pickle dump (what NEAT does internally)
                data = (config, population, species_set, self.current_generation)
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(data, f)
                print(f"[SUCCESS] File created: {checkpoint_path}")
            except Exception as e:
                print(f"[ERROR] Failed to save checkpoint: {e}")
        self.current_generation += 1

    # Override the original save_checkpoint so it does not write extra files
    # beyond the ones defined in end_generation above
    def save_checkpoint(self, config, population, species_set, generation, filename=None):
        # Without an explicit filename the default behavior would kick in (not wanted)
        if filename:
            super().write_checkpoint(config, population, species_set, generation, filename)


def build_population(config):
    """Resume from the latest checkpoint if one exists, otherwise start a new population."""
    checkpoint = None
    if os.path.exists(LATEST_CHECKPOINT):
        checkpoint = LATEST_CHECKPOINT
    elif os.path.exists(GENERATION_ZERO_CHECKPOINT):
        checkpoint = GENERATION_ZERO_CHECKPOINT

    if checkpoint:
        print(f"--- Resuming from file: {checkpoint} ---")
        with open(checkpoint, "rb") as f:
            # Raw data saved by SmartCheckpointer (only load checkpoints you created yourself)
            saved_config, population, species_set, generation = pickle.load(f)

        # Rebuild the NEAT structure without the default restore_checkpoint
        p = neat.Population(saved_config)
        p.population = population
        p.species = species_set
        p.generation = generation
        return p

    print("--- No progress found. Starting generation 0 ---")
    return neat.Population(config)


def main():
    # Load settings from the INI file
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH,
    )

    p = build_population(config)

    # Terminal reports (statistics table)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())

    # Enable our checkpointing
    p.add_reporter(SmartCheckpointer())

    try:
        # p.run calls eval_genomes repeatedly
        winner = p.run(eval_genomes)

        # Save the final champion
        with open(BEST_GENOME_PATH, "wb") as output:
            pickle.dump(winner, output)
        print(f"\n--- Training complete! Best Mario saved to '{BEST_GENOME_PATH}' ---")

    except KeyboardInterrupt:
        print("\n--- Training interrupted. Progress is safe in the checkpoint files. ---")


if __name__ == "__main__":
    main()
