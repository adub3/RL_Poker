import math

def process_sequences(sequence, initial_pot):
    result = ""
    current_number = ""
    current_pot = int(initial_pot)
    future_bets = []
    
    # First pass: collect all bets
    temp_num = ""
    for char in sequence:
        if char.isdigit():
            temp_num += char
        elif temp_num:
            future_bets.append(int(temp_num))
            temp_num = ""
    if temp_num:
        future_bets.append(int(temp_num))
    
    # Second pass: process sequence
    bet_index = 0
    for char in sequence:
        if char == 'c':
            result += 'c'
        elif char == 'r':
            if current_number:
                number = int(current_number)
                # Calculate remaining future bets
                remaining_bets = sum(future_bets[bet_index + 1:])
                # Adjust pot for current calculation
                adjusted_pot = current_pot - remaining_bets
                processed_number = math.floor(math.log(number/adjusted_pot + 1) * 4)
                result += str(processed_number)
                current_pot += number  # Update pot for next calculations
                current_number = ""
                bet_index += 1
            result += 'r'
        elif char.isdigit():
            current_number += char
    
    # Process any remaining number at the end
    if current_number:
        number = int(current_number)
        processed_number = math.floor(math.log(number/current_pot + 1) * 4)
        result += str(processed_number)
    
    return result

# Example usage
data = {'Round': 0, 'Player': 0, 'Pot': 60000, 'Money': [19900, 0], 'Private': ['8c', '8h'], 'Public': ['3s', 'Jc', '5d', '8s', 'Js'], 'Sequences': 'cr20000r40000'}

processed_sequence = process_sequences(data['Sequences'],data['Pot'])
print(processed_sequence)
