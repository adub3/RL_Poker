import networkx as nx
import matplotlib.pyplot as plt

# Create the directed graph
G = nx.DiGraph()

# Add edges with weights (costs)
edges = [
    ("Boston", "Chicago", 14),
    ("Boston", "Austin", 22),
    ("Boston", "Los Angeles", 28),
    ("Raleigh", "Chicago", 16),
    ("Raleigh", "Austin", 18),
    ("Raleigh", "Los Angeles", 17),
    ("Chicago", "Austin", 8),
    ("Chicago", "Los Angeles", 10),
]

for u, v, cost in edges:
    G.add_edge(u, v, weight=cost)

# Node positions for better visualization
pos = {
    "Boston": (0, 2),
    "Raleigh": (0, 0),
    "Chicago": (2, 1),
    "Austin": (4, 0),
    "Los Angeles": (4, 2),
}

# Draw the network
plt.figure(figsize=(10, 6))

# Draw nodes and labels
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

# Draw edges
nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)

# Add edge labels for costs
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

# Set plot title
plt.title("Minimum Cost Network Flow Problem (MCNFP)", fontsize=14)

# Show plot
plt.axis('off')
plt.show()
