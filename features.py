import re
import numpy as np

def clean_text(text):
    if not isinstance(text, str): return ""
    # Keep mathematical symbols as they are key indicators of difficulty
    text = re.sub(r'[^a-z0-9\s\+\-\*\/\%\=\<\>]', '', text.lower())
    return text

def get_extra_features(text):
    text_l = text.lower()
    
    # Specific keywords for high-precision classification
    easy_keywords = ["sum", "print", "input", "simple", "even", "odd", "basic", "integer", "numbers", "add", "swap", "reverse", "min", "max", "average"]
    medium_keywords = ["sorting", "binary search", "greedy", "hashing", "stack", "queue", "sliding window", "two pointers", "prefix sum", "bfs", "dfs", "recursion"]
    hard_keywords = ["dp", "dynamic programming", "graph", "dijkstra", "segment tree", "bitmask", "complexity", "backtracking", "flow", "geometry", "combinatorics", "strongly connected", "modular inverse"]
    
    return np.array([
        len(text),                                # Length of full text
        len(text.split()),                        # Total word count
        sum(text_l.count(c) for c in "+-*/%<>="), # Math complexity count
        sum(1 for word in easy_keywords if word in text_l),   # Easy markers
        sum(1 for word in medium_keywords if word in text_l), # Medium markers
        sum(1 for word in hard_keywords if word in text_l),   # Hard markers
        text_l.count("log"),                      # Logarithmic complexity
        text_l.count("n^2"),                      # Quadratic complexity
        text_l.count("2^n")                       # Exponential complexity
    ])