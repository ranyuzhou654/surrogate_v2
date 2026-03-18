"""Parallel execution wrapper with progress bar."""

import contextlib

import joblib
from joblib import Parallel, delayed
from tqdm import tqdm


@contextlib.contextmanager
def _tqdm_joblib(pbar):
    """Patch joblib to update a tqdm progress bar after each batch.

    Works with any joblib version by monkey-patching the internal
    BatchCompletionCallBack class.
    """
    class _TqdmCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            pbar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_cls = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _TqdmCallback
    try:
        yield pbar
    finally:
        joblib.parallel.BatchCompletionCallBack = old_cls
        pbar.close()


def parallel_map(func, items, n_jobs=-1, desc=None, **kwargs):
    """Apply func to each item in parallel with a progress bar.

    Uses process-based parallelism (loky backend, the joblib default)
    to bypass the GIL for true multi-core utilisation.

    Parameters
    ----------
    func : callable
        Function to apply to each item.  Must be picklable (top-level
        function with picklable arguments).
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

    with _tqdm_joblib(tqdm(total=n, desc=desc)):
        results = Parallel(n_jobs=n_jobs, **kwargs)(
            delayed(func)(item) for item in items
        )

    return results
