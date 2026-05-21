import pyspiel


names = pyspiel.registered_names()
print(f"Total games: {len(names)}")
print("universal_poker", "universal_poker" in names)
print("hanabi", "hanabi" in names)
print("poker games", [name for name in names if "poker" in name])
