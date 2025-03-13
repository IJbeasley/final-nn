# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    np.random.seed(42)

    # Separate sequence list by class
    pos_seqs = [seq for seq, label in zip(seqs, labels) if label]
    neg_seqs = [seq for seq, label in zip(seqs, labels) if not label]
    
    # Find the number of samples needed for balancing 
    max_samples = max(len(pos_seqs), len(neg_seqs))
    
    # Sample with replacement to balance classes
    sampled_pos = np.random.choice(pos_seqs, max_samples, replace=True).tolist()
    sampled_neg = np.random.choice(neg_seqs, max_samples, replace=True).tolist()
    
    # Combine sampled sequences and labels
    sampled_seqs = sampled_pos + sampled_neg
    sampled_labels = [True] * max_samples + [False] * max_samples
    
    return list(sampled_seqs), list(sampled_labels)

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    # Define a mapping for nucleotide bases to one-hot encoding (as provided above)
    encoding_map = {
        'A': [1, 0, 0, 0],
        'T': [0, 1, 0, 0],
        'C': [0, 0, 1, 0],
        'G': [0, 0, 0, 1]
    }
    
    # Use this map to encode sequences
    # This creates an 2D array of encoded sequences, each row is a single nucleotide in the sequence
    encoded_seqs = [np.concatenate([encoding_map[base] for base in seq]) for seq in seq_arr]

    # Convert 2D array to 1D array
    encoded_seqs = np.array(encoded_seqs)
    encoded_seqs = encoded_seqs.flatten()
    
    return encoded_seqs

