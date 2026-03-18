"""Parallel execution wrapper with progress bar."""

import multiprocessing
import threading

from joblib import Parallel, delayed
from tqdm import tqdm


def parallel_map(func, items, n_jobs=-1, desc=None, **kwargs):
    """Apply func to each item in parallel with a progress bar.

    The progress bar tracks actual job completion using a thread-safe counter.

    Parameters
    ----------
    func : callable
        Function to apply to each item.
    items : iterable
        Items to process.
    n_jobs : int
        Number of parallel jobs (-1 = all cores).
    desc : str, optional
        Progress bar description.
    **kwargs
        Additional kwargs passed to joblib.Parallel.

    Returns
    -------
    list
        Results in the same order as items.
    """
    items = list(items)
    n = len(items)

    if n_jobs == 1 or n == 0:
        return [func(item) for item in tqdm(items, desc=desc)]

    # For multi-process parallel execution, use prefer="threads" backend
    # so the tqdm pbar can be updated from worker threads.
    pbar = tqdm(total=n, desc=desc)
    lock = threading.Lock()

    def _tracked(item):
        result = func(item)
        with lock:
            pbar.update(1)
        return result

    try:
        results = Parallel(n_jobs=n_jobs, prefer="threads", **kwargs)(
            delayed(_tracked)(item) for item in items
        )
    finally:
        pbar.close()

    return results
