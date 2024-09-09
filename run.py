import gymnasium as gym
import neat
import numpy as np
import os
import pickle

# Custom logger to display only best fitness, standard deviation, and generation time
class CustomLogger(neat.reporting.BaseReporter):
    def post_evaluate(self, config, population, species, best_genome):
        fitness_scores = [genome.fitness for genome_id, genome in population.items()]
        stdev_fitness = np.std(fitness_scores)
        best_fitness = best_genome.fitness
        print(f"Best fitness: {best_fitness:.4f} | Std Dev: {stdev_fitness:.4f}")

    def end_generation(self, config, population, species):
        pass

    def start_generation(self, generation):
        print(f"Running Generation {generation}...")

    def complete_extinction(self):
        pass

def normalize_observation(observation):
    return (observation - np.mean(observation)) / (np.std(observation) + 1e-8)

def evaluate_creature(genome, config):
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    scores = []

    for trial in range(2):
        environment = gym.make("LunarLander-v2")
        state, _ = environment.reset()
        total_points = 0
        done = False
        truncated = False
        steps = 0
        successfully_landed = False

        while not (done or truncated):
            normalized_state = normalize_observation(state)
            action_taken = np.argmax(network.activate(normalized_state))
            state, reward, done, truncated, _ = environment.step(action_taken)

            if not successfully_landed and state[6] == 1 and state[7] == 1:
                successfully_landed = True

            if successfully_landed and action_taken != 0:
                reward -= 1

            total_points += reward
            steps += 1

            if steps > 1000:
                break

        if total_points > 200:
            total_points += 100
        elif total_points < 0:
            total_points -= 50

        scores.append(total_points)

    return sum(scores) / len(scores)

def assess_population(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = evaluate_creature(genome, config)

def execute_neat():
    current_directory = os.path.dirname(__file__)
    settings_path = os.path.join(current_directory, "config.txt")
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        settings_path
    )

    species_pool = neat.Population(neat_config)
    species_pool.add_reporter(CustomLogger())
    stat_tracker = neat.StatisticsReporter()
    species_pool.add_reporter(stat_tracker)

    champion = species_pool.run(assess_population, 500)
    print('\nBest Genome:\n{!s}'.format(champion))
    return champion, neat_config


def visualize_champion(champion, config, trials=25):
    neural_net = neat.nn.FeedForwardNetwork.create(champion, config)
    environment = gym.make("LunarLander-v2", render_mode="human")
    all_rewards = []
    for trial in range(trials):
        state, _ = environment.reset()
        cumulative_reward = 0
        done = False
        truncated = False
        while not (done or truncated):
            normalized_state = normalize_observation(state)
            action_taken = np.argmax(neural_net.activate(normalized_state))
            state, reward, done, truncated, _ = environment.step(action_taken)
            cumulative_reward += reward
        all_rewards.append(cumulative_reward)
        print(f"Trial {trial + 1}: Reward = {cumulative_reward}")
    environment.close()
    average_reward = sum(all_rewards) / trials
    print(f"Average reward over {trials} trials: {average_reward}")

def save_champion(champion, config, filename="champion.pkl"):
    with open(filename, "wb") as f:
        pickle.dump((champion, config), f)
    print(f"Champion saved to {filename}")

if __name__ == "__main__":
    champion, neat_settings = execute_neat()
    save_champion(champion, neat_settings)

    #Visualize the champion playing Lunar Lander
    visualize_champion(champion, neat_settings)