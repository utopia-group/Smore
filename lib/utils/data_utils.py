
from cmath import inf
import random
from typing import Dict, List, Optional


def sample_dataset(data: List[Dict], sample_size: Optional[int] = 0, sample_proportion: Optional[float] = 0.0) -> List[Dict]:
    """
    Given a dataset, sample a finite number of rows
    if sample_size is 0, sample a proportion of the dataset
    """
    
    assert sample_size > 0 or sample_proportion > 0.0

    sample_size = sample_size if sample_size > 0 else int(len(data) * sample_proportion)
    sampled_data = random.sample(data, sample_size)
    
    return sampled_data


def sample_unique_rows(data: List[Dict], column: str, max_size: Optional[int] = inf, one_col_only: bool = True) -> List[Dict]:
    """
    Given a dataset, sample a finite number of unique rows
    """
    
    assert max_size > 0

    if not one_col_only:
        raise NotImplementedError('sample_unique_rows: one_col_only is not implemented')
    
    column_data = [d[column] for d in data]
    unique_column_data = [{column: e} for e in list(set(column_data))]

    if len(unique_column_data) <= max_size:
        return unique_column_data
    else:
        return sample_dataset(unique_column_data, sample_size=max_size)
