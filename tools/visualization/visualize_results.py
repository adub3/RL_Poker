import csv
import matplotlib.pyplot as plt

def plot_poker_stats(filename='poker_stats.csv'):
    rounds = []
    player_data = {}

    try:
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)
            player_names = headers[1:]
            
            for name in player_names:
                player_data[name] = []

            for row in reader:
                rounds.append(int(row[0]))
                for i, name in enumerate(player_names):
                    player_data[name].append(int(row[i+1]))

        plt.figure(figsize=(10, 6))
        for name in player_names:
            plt.plot(rounds, player_data[name], label=name)

        plt.title('Poker Simulation - Chip Counts Over Time')
        plt.xlabel('Round')
        plt.ylabel('Chip Count')
        plt.legend()
        plt.grid(True)
        
        output_file = 'poker_simulation_results.png'
        plt.savefig(output_file)
        print(f"Graph saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: {filename} not found. Run the simulation first.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    plot_poker_stats()
