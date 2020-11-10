import numpy as np 
from itertools import repeat

def _reshape_2D(X, name):
    """
    Use Fortran ordering to convert ndarrays and lists of iterables to lists of
    1D arrays.
    Lists of iterables are converted by applying `np.asarray` to each of their
    elements.  1D ndarrays are returned in a singleton list containing them.
    2D ndarrays are converted to the list of their *columns*.
    *name* is used to generate the error message for invalid inputs.
    """
    # Iterate over columns for ndarrays, over rows otherwise.
    X = np.atleast_1d(X.T if isinstance(X, np.ndarray) else np.asarray(X))
    if X.ndim == 1 and X.dtype.type != np.object_:
        # 1D array of scalars: directly return it.
        return [X]
    elif X.ndim in [1, 2]:
        # 2D array, or 1D array of iterables: flatten them first.
        return [np.reshape(x, -1) for x in X]
    else:
        raise ValueError("{} must have 2 or fewer dimensions".format(name))

def boxplot_stats(X, whis=1.5, bootstrap=None, labels=None,
                  autorange=False):
    """
    Returns list of dictionaries of statistics used to draw a series
    of box and whisker plots. The `Returns` section enumerates the
    required keys of the dictionary. Users can skip this function and
    pass a user-defined set of dictionaries to the new `axes.bxp` method
    instead of relying on MPL to do the calculations.
    Parameters
    ----------
    X : array-like
        Data that will be represented in the boxplots. Should have 2 or
        fewer dimensions.
    whis : float, string, or sequence (default = 1.5)
        As a float, determines the reach of the whiskers to the beyond the
        first and third quartiles. In other words, where IQR is the
        interquartile range (`Q3-Q1`), the upper whisker will extend to last
        datum less than `Q3 + whis*IQR`). Similarly, the lower whisker will
        extend to the first datum greater than `Q1 - whis*IQR`.
        Beyond the whiskers, data are considered outliers
        and are plotted as individual points. This can be set this to an
        ascending sequence of percentile (e.g., [5, 95]) to set the
        whiskers at specific percentiles of the data. Finally, `whis`
        can be the string ``'range'`` to force the whiskers to the
        minimum and maximum of the data. In the edge case that the 25th
        and 75th percentiles are equivalent, `whis` can be automatically
        set to ``'range'`` via the `autorange` option.
    bootstrap : int, optional
        Number of times the confidence intervals around the median
        should be bootstrapped (percentile method).
    labels : array-like, optional
        Labels for each dataset. Length must be compatible with
        dimensions of `X`.
    autorange : bool, optional (False)
        When `True` and the data are distributed such that the  25th and
        75th percentiles are equal, ``whis`` is set to ``'range'`` such
        that the whisker ends are at the minimum and maximum of the
        data.
    Returns
    -------
    bxpstats : list of dict
        A list of dictionaries containing the results for each column
        of data. Keys of each dictionary are the following:
        ========   ===================================
        Key        Value Description
        ========   ===================================
        label      tick label for the boxplot
        mean       arithemetic mean value
        med        50th percentile
        q1         first quartile (25th percentile)
        q3         third quartile (75th percentile)
        cilo       lower notch around the median
        cihi       upper notch around the median
        whislo     end of the lower whisker
        whishi     end of the upper whisker
        fliers     outliers
        ========   ===================================
    Notes
    -----
    Non-bootstrapping approach to confidence interval uses Gaussian-
    based asymptotic approximation:
    .. math::
        \\mathrm{med} \\pm 1.57 \\times \\frac{\\mathrm{iqr}}{\\sqrt{N}}
    General approach from:
    McGill, R., Tukey, J.W., and Larsen, W.A. (1978) "Variations of
    Boxplots", The American Statistician, 32:12-16.
    """

    def _bootstrap_median(data, N=5000):
        # determine 95% confidence intervals of the median
        M = len(data)
        percentiles = [2.5, 97.5]

        bs_index = np.random.randint(M, size=(N, M))
        bsData = data[bs_index]
        estimate = np.median(bsData, axis=1, overwrite_input=True)

        CI = np.percentile(estimate, percentiles)
        return CI

    def _compute_conf_interval(data, med, iqr, bootstrap):
        if bootstrap is not None:
            # Do a bootstrap estimate of notch locations.
            # get conf. intervals around median
            CI = _bootstrap_median(data, N=bootstrap)
            notch_min = CI[0]
            notch_max = CI[1]
        else:

            N = len(data)
            notch_min = med - 1.57 * iqr / np.sqrt(N)
            notch_max = med + 1.57 * iqr / np.sqrt(N)

        return notch_min, notch_max

    # output is a list of dicts
    bxpstats = []

    # convert X to a list of lists
    X = _reshape_2D(X, "X")

    ncols = len(X)
    if labels is None:
        labels = repeat(None)
    elif len(labels) != ncols:
        raise ValueError("Dimensions of labels and X must be compatible")

    input_whis = whis
    for ii, (x, label) in enumerate(zip(X, labels)):

        # empty dict
        stats = {}
        if label is not None:
            stats['label'] = label

        # restore whis to the input values in case it got changed in the loop
        whis = input_whis

        # note tricksyness, append up here and then mutate below
        bxpstats.append(stats)

        # if empty, bail
        if len(x) == 0:
            stats['fliers'] = np.array([])
            stats['mean'] = np.nan
            stats['med'] = np.nan
            stats['q1'] = np.nan
            stats['q3'] = np.nan
            stats['cilo'] = np.nan
            stats['cihi'] = np.nan
            stats['whislo'] = np.nan
            stats['whishi'] = np.nan
            stats['med'] = np.nan
            continue

        # up-convert to an array, just to be safe
        x = np.asarray(x)

        # arithmetic mean
        stats['mean'] = np.mean(x)

        # medians and quartiles
        q1, med, q3 = np.percentile(x, [25, 50, 75])

        # interquartile range
        stats['iqr'] = q3 - q1
        if stats['iqr'] == 0 and autorange:
            whis = 'range'

        # conf. interval around median
        stats['cilo'], stats['cihi'] = _compute_conf_interval(
            x, med, stats['iqr'], bootstrap
        )

        # lowest/highest non-outliers
        if np.isscalar(whis):
            if np.isreal(whis):
                loval = q1 - whis * stats['iqr']
                hival = q3 + whis * stats['iqr']
            elif whis in ['range', 'limit', 'limits', 'min/max']:
                loval = np.min(x)
                hival = np.max(x)
            else:
                raise ValueError('whis must be a float, valid string, or list '
                                 'of percentiles')
        else:
            loval = np.percentile(x, whis[0])
            hival = np.percentile(x, whis[1])

        # get high extreme
        wiskhi = np.compress(x <= hival, x)
        if len(wiskhi) == 0 or np.max(wiskhi) < q3:
            stats['whishi'] = q3
        else:
            stats['whishi'] = np.max(wiskhi)

        # get low extreme
        wisklo = np.compress(x >= loval, x)
        if len(wisklo) == 0 or np.min(wisklo) > q1:
            stats['whislo'] = q1
        else:
            stats['whislo'] = np.min(wisklo)

        # compute a single array of outliers
        stats['fliers'] = np.hstack([
            np.compress(x < stats['whislo'], x),
            np.compress(x > stats['whishi'], x)
        ])

        # add in the remaining stats
        stats['q1'], stats['med'], stats['q3'] = q1, med, q3

    return bxpstats
