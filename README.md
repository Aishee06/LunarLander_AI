🚀 Lunar Lander AI with NEAT-Python 🧠
This project leverages the NEAT (NeuroEvolution of Augmenting Topologies) algorithm to train a neural network to play the Lunar Lander game from the Gymnasium toolkit. The goal is to evolve neural networks that can land the lunar module softly and safely. 🌕🛬

🌟 Key Features
Neuroevolution with NEAT: Evolve neural networks over multiple generations to control the lunar lander.
Custom Logger: Track fitness, performance, and evolution over time.
Visualization of Neural Networks: After training, visualize the structure and connections of the evolved networks.
Champion Network: Save and watch the best-performing network, the champion, in action.

🛠️ Setup & Installation
Prerequisites
Install Miniconda (or Anaconda) to easily manage packages and environments.
Create a new environment in VS Code with Python 3.11.9.

🚀 How It Works
1. Setting Up the Environment
We start by setting up the environment using Gymnasium's Lunar Lander game. The lander tries to land softly on the moon's surface, with the neural networks evolving to achieve a perfect landing.

2. Configuring NEAT
The configuration file, config.txt, contains crucial parameters for evolving the networks:

Population Size: 150 networks evolve in each generation.
Fitness Threshold: 300 is the target score for a perfect soft landing.
Activation Functions: Networks use activation functions like tanh, relu, and sigmoid.
Mutation Rates: Parameters define how likely networks are to mutate by adding new nodes or connections.

3. Training the Networks 🧠
In run.py, the NEAT algorithm runs through several generations, evolving the networks to optimize their lunar landing skills:

Custom Logger: Displays the best fitness and performance after each generation.
normalize_observation: Ensures input data (lander’s position, speed, etc.) is normalized for better performance.
evaluate_creature: Networks control the lander, and their fitness is determined by how well they perform.
save_champion: Once trained, the best network (the champion) is saved for later use and analysis.

4. Visualizing the Neural Network 🔍
In visualize.py, we visualize the neural network of the champion:

load_champion: Loads the saved champion network.
visualize_network: Displays the neural network, showing how the input nodes, hidden nodes, and output nodes are connected.

📊 Key Highlights
NEAT Evolution: The networks start simple and evolve to become more complex, gradually learning to control the lunar lander.
Fitness Score: Networks are scored based on their performance. The better the landing, the higher the fitness score.
Champion Visualization: The champion network, after 500 generations, is saved and can be visualized and watched.

🤖 How NEAT Works
Initial Population: Starts with 150 simple neural networks.
Mutation & Crossover: The best-performing networks are kept, mutated, and crossed over to create new networks.
Complexity Growth: Over time, networks grow more complex as they add new nodes and connections.
