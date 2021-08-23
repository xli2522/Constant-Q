# A Constant Q Transform based on *GWpy qtransform* 
#   The creation of this program was inspired by the need to include a CQT package 
#       with minimal size and dependency for SHARCNET (ComputeCanada) Supercomputer Clusters.
# 
# IMPORTANT: All credits for the original Q transform algorithm go to the authors of *GWpy* and *Omega* pipeline.
# See original algorithms at: [Omega Scan] https://gwdetchar.readthedocs.io/en/stable/omega/
#                             [GWpy] https://gwpy.github.io/docs/stable/
#          particularly       [GWpy qtransform]
#              - https://github.com/gwpy/gwpy/blob/26f63684db17104c5d552c30cdf01248b2ec76c9/gwpy/signal/qtransform.py
# NOTE: Referenced programs are under the GNU license 
# 
__version__ = 'Testing 0.0.1'

from collections import OrderedDict
# registry dict for FFT averaging methods
METHODS = OrderedDict()


def _format_name(name):
    return name.lower().replace("-", "_")

def register_method(func, name=None, deprecated=False):
    """Register a method of calculating an average spectrogram.
    Parameters
    ----------
    func : `callable`
        function to execute
    name : `str`, optional
        name of the method, defaults to ``func.__name__``
    deprecated : `bool`, optional
        whether this method is deprecated (`True`) or not (`False`)
    Returns
    -------
    name : `str`
        the registered name of the function, which may differ
        pedantically from what was given by the user.
    """
    # warn about deprecated functions
    if deprecated:
        '''func = deprecated_function(
            func,
            "the {0!r} PSD methods is deprecated, and will be removed "
            "in a future release, please consider using {1!r} instead".format(
                name, name.split('-', 1)[1],
            ),
        )'''
        Warning.warn('PSD deprecated.')
    if name is None:
        name = func.__name__
    name = _format_name(name)
    METHODS[name] = func
    return name


def get_method(name):
    """Return the PSD method registered with the given name.
    """
    if name is None:
        import warnings
        warnings.warn(
            "the default spectral averaging method is currently 'welch' "
            "(mean averages of overlapping periodograms), but this will "
            "change to 'median' as of gwpy-2.1.0",
            DeprecationWarning,
        )
        name = "welch"

    # find method
    name = _format_name(name)
    try:
        return METHODS[name]
    except KeyError as exc:
        exc.args = ("no PSD method registered with name {0!r}".format(name),)
        raise