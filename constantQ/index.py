# A Constant Q Transform based on *GWpy qtransform* 
#   The creation of this program was inspired by the need to include a CQT package 
#       with minimal size and dependency for SHARCNET (ComputeCanada) Supercomputer Clusters.
#
# Copyright (C) 2021 Xiyuan Li
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# IMPORTANT DISCLAIMER: All credits for the original Q transform algorithm go to the authors of *GWpy* and *Omega* pipeline.
# See original algorithms at: [Omega Scan] https://gwdetchar.readthedocs.io/en/stable/omega/
#                             [GWpy] https://gwpy.github.io/docs/stable/
#          particularly       [GWpy qtransform]
#              - https://github.com/gwpy/gwpy/blob/26f63684db17104c5d552c30cdf01248b2ec76c9/gwpy/signal/qtransform.py
#
# The license information does NOT imply this package (constantQ) as the original q transform/q scan algorithm.
# NOTE: Referenced programs are under the GNU license 
# 
__version__ = 'Testing 0.0.1'

import numpy
from astropy.units import Quantity

################### Index ###################
class Index(Quantity):
    """1-D `~astropy.units.Quantity` array for indexing a `Series`
        https://github.com/gwpy/gwpy/blob/v2.0.4/gwpy/types/index.py
    """
    @classmethod
    def define(cls, start, step, num, dtype=None):
        """Define a new `Index`.
        The output is basically::
            start + numpy.arange(num) * step
        Parameters
        ----------
        start : `Number`
            The starting value of the index.
        step : `Number`
            The step size of the index.
        num : `int`
            The size of the index (number of samples).
        dtype : `numpy.dtype`, `None`, optional
            The desired dtype of the index, if not given, defaults
            to the higher-precision dtype from ``start`` and ``step``.
        Returns
        -------
        index : `Index`
            A new `Index` created from the given parameters.
        """
        if dtype is None:
            dtype = max(
                numpy.array(start, subok=True, copy=False).dtype,
                numpy.array(step, subok=True, copy=False).dtype,
            )
        start = start.astype(dtype, copy=False)
        step = step.astype(dtype, copy=False)
        return cls(start + numpy.arange(num, dtype=dtype) * step, copy=False)

    @property
    def regular(self):
        """`True` if this index is linearly increasing
        """
        try:
            return self.info.meta['regular']
        except (TypeError, KeyError):
            if self.info.meta is None:
                self.info.meta = {}
            self.info.meta['regular'] = self.is_regular()
            return self.info.meta['regular']

    def is_regular(self):
        """Determine whether this `Index` contains linearly increasing samples
        This also works for linear decrease
        """
        if self.size <= 1:
            return False
        return numpy.isclose(numpy.diff(self.value, n=2), 0).all()

    def __getitem__(self, key):
        item = super().__getitem__(key)
        if item.isscalar:
            return item.view(Quantity)
        return item