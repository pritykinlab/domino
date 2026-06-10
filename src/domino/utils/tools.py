import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
    

####################################################################################################
# Data Procesessing Tools
####################################################################################################

def seq2one_hot(
    sequence: str,
    alphabet: str = "ACGT",
    neutral_alphabet: str = "N",
    neutral_value=0.25,
    dtype=np.float32,
) -> np.ndarray:
    """One-hot encode sequence."""

    def to_uint8(string):
        return np.frombuffer(string.encode("ascii"), dtype=np.uint8)

    hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
    hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
    hash_table[to_uint8(neutral_alphabet)] = neutral_value
    hash_table = hash_table.astype(dtype)
    return hash_table[to_uint8(sequence)]


def one_hot2RC_one_hot(
    one_hot_seq: np.ndarray,
) -> np.ndarray:
    reverse = np.flip(one_hot_seq, axis=0)
    reverse_complement = np.flip(reverse, axis=1)
    return np.concatenate((one_hot_seq, reverse_complement), axis=0)

def region_df(
        df: pd.DataFrame, 
        ival: tuple
) -> pd.DataFrame:
    chrom, start, end = ival
    output = df.query('chrom == @chrom and start > @start and end < @end')
    return output

def reverse_complement(one_hot_seq):
    """
    Returns reverse complement of one-hot sequence (numpy array).
    """
    reverse = np.flip(one_hot_seq, axis=0).copy()
    rc = np.flip(reverse, axis=1).copy()
    return rc

def seq_from_df(genome_fasta, df, DNA_context_length):
    """
    Generates all sequence examples from a dataframe containing
    coordinates. 
    """

    seqs = []

    for i in range(len(df)):
        chrom = df.loc[i, 'chrom']
        start = int(df.loc[i, 'start'])
        end = int(df.loc[i, 'end'])
        start = start - DNA_context_length
        end = end + DNA_context_length + 1   # exclusive
        seq = genome_fasta.fetch(chrom, start, end).upper()
        
        seq = seq2one_hot(seq)
        if 'direction' in df.columns and df.loc[i, 'direction'] == "-":
            seq = reverse_complement(seq)
        seq_tensor = torch.permute(torch.tensor(seq).float(), (1, 0))
        seqs.append(seq_tensor)

    X = torch.stack(seqs)

    return X