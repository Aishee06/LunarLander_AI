import pickle
import neat
import matplotlib.pyplot as plt
import networkx as nx

def load_champion(filename="champion.pkl"):
    with open(filename, "rb") as f:
        champion, config = pickle.load(f)
    return champion, config

def visualize_network(genome, config):
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    input_nodes = config.genome_config.input_keys
    output_nodes = config.genome_config.output_keys
    hidden_nodes = [n for n in genome.nodes.keys() if n not in input_nodes and n not in output_nodes]
    
    for n in input_nodes:
        G.add_node(n, color='lightblue', layer=0)
    for n in hidden_nodes:
        G.add_node(n, color='lightgreen', layer=1)
    for n in output_nodes:
        G.add_node(n, color='salmon', layer=2)
    
    # Add connections
    for key, cg in genome.connections.items():
        if cg.enabled:
            input_node, output_node = key
            G.add_edge(input_node, output_node, weight=cg.weight)
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    pos = nx.multipartite_layout(G, subset_key="layer")
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=[G.nodes[n]['color'] for n in G.nodes])
    
    # Draw edges
    edges = nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
    
    # Add labels
    nx.draw_networkx_labels(G, pos)
    
    plt.title("Neural Network Visualization")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("neural_network.png")
    plt.show()

if __name__ == "__main__":
    champion, config = load_champion()
    visualize_network(champion, config)