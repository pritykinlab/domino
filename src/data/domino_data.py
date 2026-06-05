#####################################################################
# domino_data.py
# Data functions for Domino family models
#####################################################################

import numpy as np
import pysam
import pandas as pd
from domino.utils.tools import seq2one_hot
import bioframe
from torch.utils.data import Dataset
from typing import Literal, Optional, List

def process_bed_file_df(
    bed_fn: str, 
    target_col: Literal['ins_score', 'bound_score']
) -> pd.DataFrame:
    df = bioframe.read_table(bed_fn)
    df.columns = ['chrom', 'start', 'end', 'dot', target_col, 'direction']
    df.drop(columns=['dot', 'direction'], inplace=True)
    return df

def drop_blacklist(
    X: pd.DataFrame,
    blacklist_path: str 
) -> tuple[pd.DataFrame, np.ndarray]:
    # '/scratch/gpfs/PRITYKIN/brendan/microc_preprocessing/annotations/dm6-blacklist.v2.bed
    """Drop blacklisted regions from current dataframe."""
    blacklisted_regions_df = bioframe.read_table(blacklist_path)
    blacklisted_regions_df.columns = ['chrom', 'start', 'end', 'description']
    overlap_df = bioframe.overlap(X, blacklisted_regions_df, how='left').copy().query('description_.isnull()')
    overlap_df.drop(columns=['chrom_', 'start_', 'end_', 'description_'], inplace=True)
    overlap_df.copy_index_1 = overlap_df.copy_index_1.astype(np.int64)    
    overlap_df.copy_index_2 = overlap_df.copy_index_2.astype(np.int64)
    overlap_df.start = overlap_df.start.astype(np.int64)    
    overlap_df.end = overlap_df.end.astype(np.int64)
    overlap_df.boundary = overlap_df.boundary.astype(np.int64)
    
    new_X = overlap_df.reset_index()
    new_X.drop('index', axis=1, inplace=True)
    
    return new_X

def preprocess_data(
    ins1_path: str,
    ins2_path: str,
    bound_path: str,
    demean_context_bin_width: int,
    blacklist_path: str = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Returns combined dataframe and demeaned insulation score values.
    """
    ins_score1 = process_bed_file_df(ins1_path, target_col='ins_score')
    ins_score2 = process_bed_file_df(ins2_path, target_col='ins_score')
    ins_score1['copy_index'] = ins_score1.index
    ins_score2['copy_index'] = ins_score2.index

    bound1 = process_bed_file_df(bound_path, target_col='bound_score')

    # subset insulation df into boundaries and non-boundaries
    ins_score_w_bound1 = ins_score1.merge(bound1, on=['chrom', 'start', 'end']) 
    ins_score_wo_bound1 = ins_score1[~ins_score1['copy_index'].isin(ins_score_w_bound1['copy_index'])] 

    # positive examples where context is within bounds and context contains no NaNs 
    idxs_pos = [np.arange(i-demean_context_bin_width, i+demean_context_bin_width+1) for i in ins_score_w_bound1['copy_index']]
    idxs_pos = np.array(idxs_pos)
    condition = np.any(idxs_pos < 0, axis=1) | np.any(idxs_pos >= len(ins_score1), axis=1)
    idxs_pos = idxs_pos[~condition]
    mid_pos = idxs_pos[:,demean_context_bin_width]
    y_pos = np.array([np.array(ins_score1['ins_score'][idx_arr]) for idx_arr in idxs_pos]) # context with middle bin
    X_pos = ins_score1.iloc[mid_pos] 
    condition = np.isnan(y_pos).any(axis=1) | np.isinf(y_pos).any(axis=1)
    X_pos = X_pos[~condition]
    y_pos = y_pos[~condition]

    # sample negative examples where context is within bounds and context contains no NaNs 
    # idxs_neg = [np.arange(i+20000-10, i+20000+11) for i in ins_score_w_bound1['copy_index']]
    idxs_neg = [np.arange(i-demean_context_bin_width, i+demean_context_bin_width+1) for i in ins_score_wo_bound1['copy_index']]
    idxs_neg = np.array(idxs_neg)
    
    np.random.seed(42)
    i = np.random.choice(idxs_neg.shape[0], y_pos.shape[0], replace=False)
    idxs_neg = idxs_neg[i]
    condition = np.any(idxs_neg < 0, axis=1) | np.any(idxs_neg >= len(ins_score1), axis=1)
    idxs_neg = idxs_neg[~condition]
    mid_neg = idxs_neg[:,demean_context_bin_width]
    y_neg = np.array([np.array(ins_score1['ins_score'][idx_arr]) for idx_arr in idxs_neg])
    X_neg = ins_score1.iloc[mid_neg]
    condition = np.isnan(y_neg).any(axis=1) | np.isinf(y_neg).any(axis=1)
    X_neg = X_neg[~condition]
    y_neg = y_neg[~condition]

    # negative and positive insulations
    X12_pos = X_pos.merge(ins_score2, on=['chrom', 'start', 'end'], suffixes=('_1', '_2'))
    X12_neg = X_neg.merge(ins_score2, on=['chrom', 'start', 'end'], suffixes=('_1', '_2'))

    # insulation context for positive and negative examples 
    idxs2_pos = [np.arange(i-demean_context_bin_width, i+demean_context_bin_width+1) for i in X12_pos['copy_index_2']]
    idxs2_pos = np.array(idxs2_pos)
    y2_pos = np.array([np.array(ins_score2['ins_score'][idx_arr]) for idx_arr in idxs2_pos])

    idxs2_neg = [np.arange(i-demean_context_bin_width, i+demean_context_bin_width+1) for i in X12_neg['copy_index_2']]
    idxs2_neg = np.array(idxs2_neg)
    y2_neg = np.array([np.array(ins_score2['ins_score'][idx_arr]) for idx_arr in idxs2_neg])

    # IMPORTANT: filters for all examples where both rep1 and rep2 don't have an NaN (sanity check)
    condition = np.isnan(y2_pos).any(axis=1) | np.isinf(y2_pos).any(axis=1)
    X12_pos = X12_pos[~condition]
    y1_pos = y_pos[~condition]
    y2_pos = y2_pos[~condition]

    condition = np.isnan(y2_neg).any(axis=1) | np.isinf(y2_neg).any(axis=1)
    X12_neg = X12_neg[~condition]
    y1_neg = y_neg[~condition]
    y2_neg = y2_neg[~condition]

    demeaned_y1_pos = y1_pos - np.mean(y1_pos, axis=1)[:, np.newaxis]
    middle_points1_pos = demeaned_y1_pos[:, demean_context_bin_width]
    demeaned_y1_neg = y1_neg - np.mean(y1_neg, axis=1)[:, np.newaxis]
    middle_points1_neg = demeaned_y1_neg[:, demean_context_bin_width]
    y1 = np.concatenate([middle_points1_pos, middle_points1_neg])

    demeaned_y2_pos = y2_pos - np.mean(y2_pos, axis=1)[:, np.newaxis] # np.newaxis reshapes into column vector
    middle_points2_pos = demeaned_y2_pos[:, demean_context_bin_width]
    demeaned_y2_neg = y2_neg - np.mean(y2_neg, axis=1)[:, np.newaxis]
    middle_points2_neg = demeaned_y2_neg[:, demean_context_bin_width]
    y2 = np.concatenate([middle_points2_pos, middle_points2_neg])

    X12_pos['boundary'] = 1
    X12_neg['boundary'] = 0

    X = pd.concat([X12_pos, X12_neg]).reset_index(drop=True)

    X['demeaned_ins_score_1'] = y1
    X['demeaned_ins_score_2'] = y2

    # additional filtering (if needed)
    if blacklist_path:
        X = drop_blacklist(X, blacklist_path)

    # if chroms_to_drop:
        # e.g. for dm6, chroms_to_drop = ['chrM'] => drop mitochrondrial chromosome 
        # X = X.query('~ chrom.isin(@chroms_to_drop)')

    return X


class GenomeDataset(Dataset):
    """
    Takes in assembly and outputs dataset.
    Adapted from https://github.com/tanjimin/C.Origami/blob/main/src/corigami/data/genome_dataset.py
    """
    def __init__(
        self, 
        ins1_path: str,
        ins2_path: str,
        bound_path: str,
        genome_path: str,
        demean_context_bin_width: int,
        DNA_context_bin_width: int,
        boundary_length: int,
        fold_coords_dict: dict,
        blacklist_path: str = None
        # mode: str = 'train'
    ):
        df = preprocess_data(ins1_path, ins2_path, bound_path, demean_context_bin_width, blacklist_path)
        df.chrom = df.chrom.astype(str) # ensure chrom is of type str
        self.genome_path = genome_path
        self.DNA_context_bin_width = DNA_context_bin_width
        self.boundary_length = boundary_length
        # self.chr_names = self.get_chr_names()
        self.fold_coords_dict = fold_coords_dict
        
        # only splitting one chrom
        if 'full_chr_names' not in self.fold_coords_dict:
            chrom_to_subset = self.fold_coords_dict.get('chrom_to_subset')
            if chrom_to_subset:
                min_start = self.fold_coords_dict.get('min_start')
                max_end = self.fold_coords_dict.get('max_end')
                self.df = self._get_chrom_subset(df, chrom_to_subset, min_start, max_end)
        # case: multiple chromosomes + optional subset
        else:
            full_chr_names = self.fold_coords_dict['full_chr_names']
            self.df = df[df['chrom'].isin(full_chr_names)]

            chrom_to_subset = self.fold_coords_dict.get('chrom_to_subset')
            if chrom_to_subset:
                min_start = self.fold_coords_dict.get('min_start')
                max_end = self.fold_coords_dict.get('max_end')
                subset_df = self._get_chrom_subset(df, chrom_to_subset, min_start, max_end)
                self.df = pd.concat([self.df, subset_df])

        print(self.df.head())
        print(len(self.df))
        # if mode == 'train':
        #     self.chr_names.remove('chr2R')
        #     self.df = df.query(f'chrom.isin({self.chr_names})')
        # elif mode == 'val':
        #     self.chr_names = ['chr2R']
        #     # lower bound for chromosome 2
        #     self.min_val_start = 15000000
        #     self.df = df.query(f'chrom.isin({self.chr_names}) and start >= {self.min_val_start}')
        #     # self.df = df.query(f'chrom.isin({self.chr_names})')
        # elif mode == 'test':
        #     self.chr_names = ['chr2R']
        #     # upper bound (to prevent any overlap between boundary examples in validation set)
        #     self.max_test_end = 15000000 - (200 * DNA_context_bin_width) 
        #     self.df = df.query(f'chrom.isin({self.chr_names}) and end <= {self.max_test_end}')
        # else:
        #     raise ValueError(f'Unknown mode "{mode}"')
            
        self.data = self.load_data()
        
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
    
    def _get_chrom_subset(self, df, chrom, min_start=None, max_end=None):
        query_parts = [f'chrom == "{chrom}"']
        if min_start is not None:
            query_parts.append(f'start >= {min_start}')
        if max_end is not None:
            query_parts.append(f'end <= {max_end}')
        query_str = ' and '.join(query_parts)
        return df.query(query_str)  
    
    
    # def get_chr_names(self):
    #     chr_names = [
    #         'chr2L', 'chr2R', 
    #         'chr3L', 'chr3R', 
    #         'chr4', 'chrX', 'chrY'
    #     ]
    #     return chr_names
    
    def load_data(self):
        """
        Returns list of all examples (forward and reverse for each example).
        Each example contains: (OHE np array, demeaned_insulation, boundary, chrom, start, end, direction).
        """
        DNA_context_length = self.boundary_length * self.DNA_context_bin_width
        genome_fasta = pysam.FastaFile(self.genome_path)
        outputs = []

        # get attributes of current row
        for i, _ in self.df.iterrows():
            chrom = self.df.loc[i, 'chrom']
            start = int(self.df.loc[i, 'start'])
            end = int(self.df.loc[i, 'end'])
            boundary = int(self.df.loc[i, 'boundary'])
            y = self.df.loc[i, 'demeaned_ins_score_1']
            start = start - DNA_context_length
            end = end + DNA_context_length + 1   # exclusive
            seq = genome_fasta.fetch(chrom, start, end).upper()
            
            # forward example
            fw_seq = seq2one_hot(seq)
            outputs.append((fw_seq, y, boundary, chrom, start, end, '+'))
            
            # reverse example
            rc_seq = self.reverse_complement(fw_seq)
            outputs.append((rc_seq, y, boundary, chrom, start, end, '-'))
        
        return outputs
        
    def reverse_complement(self, one_hot_seq):
        """
        Returns reverse complement of one-hot sequence (numpy array).
        """
        reverse = np.flip(one_hot_seq, axis=0).copy()
        rc = np.flip(reverse, axis=1).copy()
        return rc
    