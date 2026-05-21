import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, x, y, width, height, text, color='#E0E0E0', edge_color='black', fontsize=9):
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", 
                                  linewidth=2, edgecolor=edge_color, facecolor=color)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold')
    return x + width/2, y + height, x + width/2, y

def draw_arrow(ax, x1, y1, x2, y2, text="", connectionstyle="arc3,rad=0", color='black', dashed=False):
    style = "->"
    ls = '--' if dashed else '-'
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, connectionstyle=connectionstyle, lw=1.5, color=color, linestyle=ls))
    if text:
        # Simple midpoint calculation for straight lines
        if connectionstyle == "arc3,rad=0":
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ax.text(mid_x, mid_y, text, ha='center', va='center', fontsize=8, backgroundcolor='white')

def main():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # --- Training Phase (Left) ---
    ax.text(3.5, 9.5, "PHASE 1: TRAINING (CFR)", ha='center', fontsize=12, fontweight='bold', color='#555')
    
    # 1. Regret Table (The "Brain" being trained)
    tr_regret_x, tr_regret_y_top, tr_regret_bot_x, tr_regret_bot_y = draw_box(ax, 2, 7, 3, 1.5, "Regret Table\n(Cumulative Regrets)", color='#FFB6C1')
    
    # 2. Regret Matching
    tr_match_x, tr_match_y_top, tr_match_bot_x, tr_match_bot_y = draw_box(ax, 2, 5, 3, 1, "Regret Matching\n(Calculate Strategy)", color='#D8BFD8')
    
    # 3. Game Simulation
    tr_sim_x, tr_sim_y_top, tr_sim_bot_x, tr_sim_bot_y = draw_box(ax, 2, 3, 3, 1, "Traverse Game Tree\n(Simulate Actions)", color='#D8BFD8')
    
    # 4. Update
    tr_upd_x, tr_upd_y_top, tr_upd_bot_x, tr_upd_bot_y = draw_box(ax, 0.5, 5, 1, 3, "Update\nRegrets", color='#FF6347')

    # Edges Training
    draw_arrow(ax, tr_regret_bot_x, tr_regret_bot_y, tr_match_x, tr_match_y_top)
    draw_arrow(ax, tr_match_bot_x, tr_match_bot_y, tr_sim_x, tr_sim_y_top, "Current Policy")
    
    # Cycle back (Update)
    draw_arrow(ax, tr_sim_x - 1.5, tr_sim_bot_y + 0.5, tr_upd_x + 0.5, tr_upd_y_top - 3, "Counterfactual\nValues")
    draw_arrow(ax, tr_upd_x + 0.5, tr_upd_y_top, tr_regret_x - 0.1, tr_regret_y_top - 0.75, "Add Regret")


    # --- The Bridge (Center) ---
    # Final Strategy File
    strat_x, strat_y_top, strat_bot_x, strat_bot_y = draw_box(ax, 6, 4.5, 2, 2, "FINAL STRATEGY\n(Average Policy)\n[Bucket] -> [Probs]", color='#FFD700', edge_color='red')
    
    # Link Training to Strategy
    draw_arrow(ax, tr_match_x + 1.5, tr_match_bot_y + 0.5, strat_x, strat_y_top - 1, "Save converged\nstrategy", dashed=True)


    # --- Runtime Phase (Right) ---
    ax.text(10.5, 9.5, "PHASE 2: RUNTIME (THE BOT)", ha='center', fontsize=12, fontweight='bold', color='#555')

    # 1. Game Engine
    rt_eng_x, rt_eng_y_top, rt_eng_bot_x, rt_eng_bot_y = draw_box(ax, 9, 7.5, 3, 1, "Game Engine\n(OpenSpiel)", color='#87CEEB')
    
    # 2. Abstraction
    rt_abs_x, rt_abs_y_top, rt_abs_bot_x, rt_abs_bot_y = draw_box(ax, 9, 5.5, 3, 1, "Abstraction\n(abstraction.py)", color='#98FB98')
    
    # 3. Lookup
    rt_look_x, rt_look_y_top, rt_look_bot_x, rt_look_bot_y = draw_box(ax, 9, 3.5, 3, 1, "Lookup Action", color='#98FB98')
    
    # 4. Action
    rt_act_x, rt_act_y_top, rt_act_bot_x, rt_act_bot_y = draw_box(ax, 9, 1.5, 3, 1, "Execute Action", color='#D3D3D3')

    # Edges Runtime
    draw_arrow(ax, rt_eng_bot_x, rt_eng_bot_y, rt_abs_x, rt_abs_y_top, "Raw State")
    draw_arrow(ax, rt_abs_bot_x, rt_abs_bot_y, rt_look_x, rt_look_y_top, "Bucket Key\ne.g. [132][rc]")
    
    # Lookup using Strategy
    draw_arrow(ax, strat_bot_x + 1, strat_bot_y - 1, rt_look_x - 1.5, rt_look_y_top - 0.5, "Get Probs", connectionstyle="arc3,rad=0.2")
    
    draw_arrow(ax, rt_look_bot_x, rt_look_bot_y, rt_act_x, rt_act_y_top, "Sample Policy")
    
    # Loop back to Engine
    ax.annotate("", xy=(rt_eng_x + 3, rt_eng_y_top - 0.5), xytext=(rt_act_bot_x, rt_act_bot_y),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.4", lw=1.5, color='black', linestyle='--'))
    ax.text(13, 4.5, "Next State", ha='center', va='center', fontsize=8, rotation=270, backgroundcolor='white')

    plt.tight_layout()
    plt.savefig('full_system_flow.png', dpi=300)
    print("Full system diagram saved to full_system_flow.png")

if __name__ == "__main__":
    main()
