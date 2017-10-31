# signal.py

def decimate(x, *, epochs, prefilter, fs_in=None, fs_out=None, q=None, axis=-1):
    """
    Downsample the signal after applying an anti-aliasing filter.

    Parameters
    ----------
    x : ndarray
        The signal to be downsampled, as an N-dimensional array.
    epochs : , explicit
        If None, then the signal is assumed to be contiguous.
    prefilter : callable filter ???, explicit
        If None, then no filtering is applied before downsampling.
    fs_in : float, optional
    fs_out : float, optional
    q : int, optional
        The downsampling factor. For downsampling factors higher than 13, it is
        recommended to call `decimate` multiple times.
    axis : int, optional
        The axis along which to decimate.
    Returns
    -------
    y : ndarray
        The down-sampled signal.
    """

    if q is None:
        assert fs_out is not None, 'either fs_out or q must be specified!'
    else:
        assert fs_out is None, 'fs_out and q cannot be set simultaneously!'
    if fs_out is not None:
        assert fs_in is not None, 'fs_in and fs_out must both be specified, or use q instead!'

    # valid case 1: (fs_in, q)
    # valid case 2: (fs_in, fs_out)
    # valid case 3: q

    # invalid case 1: q, fs_out
    # invalid case 2: fs_out
