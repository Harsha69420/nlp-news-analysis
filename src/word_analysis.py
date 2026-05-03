import re

# Sample sentences (use some real ones later)
sentences = [
    "Breaking: India wins cricket match 2024",
    "Election results will be announced on 15th May",
    "AI technology is growing rapidly in 2025"
]

# -------------------------
# REGEX PATTERNS
# -------------------------

# 1. Extract numbers
pattern_numbers = r'\d+'

# 2. Extract dates (like 15th May)
pattern_dates = r'\b\d{1,2}(?:st|nd|rd|th)?\s[A-Z][a-z]+\b'

# 3. Extract capital words (possible names/entities)
pattern_caps = r'\b[A-Z]{2,}|\b[A-Z][a-z]+\b'


for sent in sentences:
    print("\nSentence:", sent)

    print("Numbers:", re.findall(pattern_numbers, sent))
    print("Dates:", re.findall(pattern_dates, sent))
    print("Capital Words:", re.findall(pattern_caps, sent))
