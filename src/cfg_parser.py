import nltk
from nltk import CFG

# Define grammar
grammar = CFG.fromstring("""
S -> NP VP
NP -> Det N | N
VP -> V NP | V
Det -> 'the' | 'a'
N -> 'india' | 'team' | 'match' | 'player'
V -> 'wins' | 'plays'
""")

parser = nltk.ChartParser(grammar)

sentence = "india wins match".split()

print("Sentence:", sentence)

for tree in parser.parse(sentence):
    print(tree)
    tree.pretty_print()
