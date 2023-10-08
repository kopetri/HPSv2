class PairedComparison:
    def __init__(self, a, b, score):
        self.a = a
        self.b = b
        self.score = score

def rank_elements(comparisons):
    # Create a dictionary to store Elo ratings for each element
    elo_ratings = {}
    
    # Initialize Elo ratings for all elements to a common starting value (e.g., 1000)
    starting_rating = 1000
    for comparison in comparisons:
        elo_ratings[comparison.a] = starting_rating
        elo_ratings[comparison.b] = starting_rating

    # Define the K-factor (controls how quickly Elo ratings change)
    k_factor = 32

    # Perform Elo updates based on paired comparisons
    for comparison in comparisons:
        a_rating = elo_ratings[comparison.a]
        b_rating = elo_ratings[comparison.b]
        
        # Calculate expected scores
        expected_a = 1 / (1 + 10 ** ((b_rating - a_rating) / 400))
        expected_b = 1 / (1 + 10 ** ((a_rating - b_rating) / 400))
        
        # Update ratings based on the actual outcome
        new_a_rating = a_rating + k_factor * (comparison.score - expected_a)
        new_b_rating = b_rating + k_factor * ((1 - comparison.score) - expected_b)
        
        # Update Elo ratings in the dictionary
        elo_ratings[comparison.a] = new_a_rating
        elo_ratings[comparison.b] = new_b_rating

    # Sort elements based on their final Elo ratings (from best to worst)
    ranked_elements = sorted(elo_ratings.keys(), key=lambda x: elo_ratings[x], reverse=True)
    
    return ranked_elements

# Example usage:
if __name__ == "__main__":
    import numpy as np
    from itertools import combinations
    comparisons = [PairedComparison(a, b, np.random.randint(2)) for a,b in combinations(["A", "B", "C", "D", "E"], 2)]
    print([(c.a, c.b, c.score) for c in comparisons])

    ranked_elements = rank_elements(comparisons)
    print("Ranking:", ranked_elements)