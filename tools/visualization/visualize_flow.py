import matplotlib.pyplot as plt
import networkx as nx

def draw_poker_decision_tree():
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes (States)
    G.add_node("Start", label="Start\n(Deal Cards)", color='#ADD8E6') # Light Blue
    
    # Pre-flop decision
    G.add_node("HandStrength", label="Eval Hand\n(High/Pair?)", color='#90EE90') # Light Green
    G.add_node("PreFlopAction", label="Action?\n(Call/Raise)", color='#FFB6C1') # Light Salmon
    
    # Flop
    G.add_node("FlopDeal", label="Flop\n(3 Cards)", color='#ADD8E6')
    G.add_node("HandUpdate1", label="Eval Hand\n(Flush Draw?)", color='#90EE90')
    G.add_node("FlopAction", label="Action?\n(Check/Bet)", color='#FFB6C1')

    # Edges
    G.add_edge("Start", "HandStrength")
    G.add_edge("HandStrength", "PreFlopAction")
    G.add_edge("PreFlopAction", "FlopDeal", label="Call/Raise")
    G.add_edge("FlopDeal", "HandUpdate1")
    G.add_edge("HandUpdate1", "FlopAction")

    # Define positions for a tree-like structure
    pos = {
        "Start": (0, 10),
        "HandStrength": (0, 8),
        "PreFlopAction": (0, 6),
        "FlopDeal": (0, 4),
        "HandUpdate1": (0, 2),
        "FlopAction": (0, 0)
    }

    # Draw the graph
    plt.figure(figsize=(6, 8))
    
    # Draw nodes
    colors = [G.nodes[n]['color'] for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=colors, edgecolors='black', linewidths=1.5)
    
    # Draw labels
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, width=2)
    
    # Edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

    plt.title("Poker Bot Decision Flow (Simplified Round)", fontsize=14, fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('poker_decision_flow.png', dpi=300)
    print("Decision flow diagram saved to poker_decision_flow.png")

if __name__ == "__main__":
    draw_poker_decision_tree()
