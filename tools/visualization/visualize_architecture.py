import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, x, y, width, height, text, color='#E0E0E0', edge_color='black'):
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", 
                                  linewidth=2, edgecolor=edge_color, facecolor=color)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=10, fontweight='bold')
    return x + width/2, y + height, x + width/2, y

def draw_arrow(ax, x1, y1, x2, y2, text=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))
    if text:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(mid_x, mid_y, text, ha='center', va='center', fontsize=8, backgroundcolor='white')

def main():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, "Poker Bot Architecture (New Code)", ha='center', fontsize=14, fontweight='bold')

    # Nodes
    # Game Engine
    ge_top_x, ge_top_y, ge_bot_x, ge_bot_y = draw_box(ax, 3.5, 7.5, 3, 1, "Game Engine\n(OpenSpiel / pyspiel)", color='#FFD700')
    
    # Bot Main Loop
    bot_top_x, bot_top_y, bot_bot_x, bot_bot_y = draw_box(ax, 3.5, 4.5, 3, 1.5, "Bot Logic\n(skeleton.py)", color='#87CEEB')
    
    # Abstraction
    abs_top_x, abs_top_y, abs_bot_x, abs_bot_y = draw_box(ax, 0.5, 4.5, 2, 1.5, "Abstraction\n(abstraction.py)", color='#98FB98')
    
    # Strategy File
    strat_top_x, strat_top_y, strat_bot_x, strat_bot_y = draw_box(ax, 7.5, 4.5, 2, 1.5, "Strategy File\n(JSON / .txt)", color='#FFA07A')

    # Action Space
    act_top_x, act_top_y, act_bot_x, act_bot_y = draw_box(ax, 3.5, 1.5, 3, 1, "Action Selection", color='#D3D3D3')

    # Edges
    
    # Engine -> Bot
    draw_arrow(ax, ge_bot_x, ge_bot_y, bot_top_x, bot_top_y, "Raw Game State\n(Cards, History)")
    
    # Bot <-> Abstraction
    draw_arrow(ax, bot_top_x - 1.5, bot_top_y - 0.75, abs_top_x + 2, abs_top_y - 0.75, "Raw Data")
    draw_arrow(ax, abs_top_x + 2, abs_bot_y + 0.75, bot_top_x - 1.5, bot_bot_y + 0.75, "Abstract Key\n(e.g., [1342][rc])")
    
    # Bot <-> Strategy
    draw_arrow(ax, bot_top_x + 1.5, bot_top_y - 0.75, strat_top_x, strat_top_y - 0.75, "Lookup Key")
    draw_arrow(ax, strat_top_x, strat_bot_y + 0.75, bot_top_x + 1.5, bot_bot_y + 0.75, "Probabilities")
    
    # Bot -> Action
    draw_arrow(ax, bot_bot_x, bot_bot_y, act_top_x, act_top_y, "Policy")
    
    # Action -> Engine
    # Curved arrow for feedback
    ax.annotate("", xy=(ge_top_x + 3, ge_top_y - 0.5), xytext=(act_bot_x, act_bot_y),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.4", lw=1.5, color='black', linestyle='--'))
    ax.text(8.5, 3, "Selected Action\n(Fold, Call, Raise)", ha='center', fontsize=8, backgroundcolor='white')

    plt.tight_layout()
    plt.savefig('bot_architecture.png', dpi=300)
    print("Architecture diagram saved to bot_architecture.png")

if __name__ == "__main__":
    main()
