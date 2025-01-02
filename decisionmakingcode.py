from treys import Card, Evaluator, Deck
import pandas
evaluator = Evaluator()
round = False
data = {'preflop': 500, 'flop': 200}
print(data)
for bet in data:
    if round:
        print(data[bet])
    else:
        print('nope')

round = 'preflop'

for bet in data:
    if round:
        if round == bet:
            print(data[round])
    else:
        print('nope')