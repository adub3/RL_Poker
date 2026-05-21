import matplotlib.pyplot as plt
import networkx as nx

def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
    '''
    If there is a cycle that is reachable from root, then result will not be a hierarchy.
    G: the graph (must be a tree)
    root: the root node of current branch 
    width: horizontal space allocated for this branch - avoids overlap with other branches
    vert_gap: gap between levels of hierarchy
    vert_loc: vertical location of root
    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G))) #allows back compatibility with nx version 1.11
        else:
            # Need to import random for this line to work
            import random
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

def draw_game_tree():
    G = nx.DiGraph()
    
    # Define the nodes and structure
    # Root
    G.add_node("Root", label="Pre-Flop\nP1 Act", color='#87CEEB') # Blue
    
    # Level 1 (P1 Actions)
    G.add_node("P1_Fold", label="P2 Wins", color='#FF6347') # Red (Terminal)
    G.add_node("P1_Call", label="P2 Act", color='#98FB98') # Green
    G.add_node("P1_Raise", label="P2 Act", color='#98FB98') # Green
    
    G.add_edge("Root", "P1_Fold", action="Fold")
    G.add_edge("Root", "P1_Call", action="Call")
    G.add_edge("Root", "P1_Raise", action="Raise")
    
    # Level 2 (From P1_Call -> P2 Act)
    G.add_node("P2_Check_Flop", label="FLOP", color='#FFD700') # Gold
    G.add_node("P2_Raise_After_Call", label="P1 Act", color='#87CEEB') # Blue
    
    G.add_edge("P1_Call", "P2_Check_Flop", action="Check")
    G.add_edge("P1_Call", "P2_Raise_After_Call", action="Raise")
    
    # Level 2 (From P1_Raise -> P2 Act)
    G.add_node("P2_Fold_After_Raise", label="P1 Wins", color='#FF6347') # Red
    G.add_node("P2_Call_After_Raise", label="FLOP", color='#FFD700') # Gold
    G.add_node("P2_Reraise", label="P1 Act", color='#87CEEB') # Blue
    
    G.add_edge("P1_Raise", "P2_Fold_After_Raise", action="Fold")
    G.add_edge("P1_Raise", "P2_Call_After_Raise", action="Call")
    G.add_edge("P1_Raise", "P2_Reraise", action="Re-Raise")

    # Level 3 (From P2_Raise_After_Call -> P1 Act)
    G.add_node("P1_Fold_L3", label="P2 Wins", color='#FF6347')
    G.add_node("P1_Call_L3", label="FLOP", color='#FFD700')
    
    G.add_edge("P2_Raise_After_Call", "P1_Fold_L3", action="Fold")
    G.add_edge("P2_Raise_After_Call", "P1_Call_L3", action="Call")

    # Draw
    pos = hierarchy_pos(G, "Root")
    
    plt.figure(figsize=(12, 8))
    
    colors = [G.nodes[n]['color'] for n in G.nodes]
    labels = nx.get_node_attributes(G, 'label')
    edge_labels = nx.get_edge_attributes(G, 'action')
    
    nx.draw(G, pos, with_labels=False, node_size=2500, node_color=colors, edgecolors='black', linewidths=1.5, arrows=True)
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5)
    
    plt.title("Partial Game Tree (Pre-Flop)", fontsize=16)
    plt.axis('off')
    
    plt.savefig('poker_game_tree.png', dpi=300)
    print("Game tree saved to poker_game_tree.png")

if __name__ == "__main__":
    draw_game_tree()
