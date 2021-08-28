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
import warnings

from astropy.units import (Unit, Quantity)

from constantQ.utilityaccess import if_not_none
from constantQ.index import Index
from constantQ.segments import Segment

################### Array ###################
class Array(Quantity):
    """Array holding data with a unit, and other metadata
        Array([ 1., 2., 3., 4., 5.]
              unit: Unit("m / s"),
              name: 'my data',
              epoch: None,
              channel: None)
        https://github.com/gwpy/gwpy/blob/v2.0.4/gwpy/types/array.py
    """

    _metadata_slots = ('name', 'epoch', 'channel')

    def __new__(cls, value, unit=None,  # Quantity attrs
                name=None, epoch=None, channel=None,  # new attrs
                dtype=None, copy=True, subok=True,  # ndarray attrs
                order=None, ndmin=0):
        """Create a new `Array`
        """
        # pick dtype from input array
        if dtype is None and isinstance(value, numpy.ndarray):
            dtype = value.dtype

        # create new array
        new = super().__new__(cls, value, unit=unit, dtype=dtype, copy=False,
                              order=order, subok=subok, ndmin=ndmin)

        # explicitly copy here to get ownership of the data,
        # see (astropy/astropy#7244)
        if copy:
            new = new.copy()

        # set new attributes
        if name is not None:
            new.name = name
        if epoch is not None:
            new.epoch = epoch
        if channel is not None:
            new.channel = channel

        return new

    def __array_finalize__(self, obj):
        # format a new instance of this class starting from `obj`
        if obj is None:
            return

        # call Quantity.__array_finalize__ to handle the units
        super().__array_finalize__(obj)

        # then update metadata
        if isinstance(obj, Quantity):
            self.__metadata_finalize__(obj, force=False)

    def __metadata_finalize__(self, obj, force=False):
        # apply metadata from obj to self if creating a new object
        for attr in self._metadata_slots:
            _attr = '_%s' % attr  # use private attribute (not property)
            # if attribute is unset, default it to None, then update
            # from obj if desired
            try:
                getattr(self, _attr)
            except AttributeError:
                update = True
            else:
                update = force
            if update:
                try:
                    val = getattr(obj, _attr)
                except AttributeError:
                    continue
                else:
                    if isinstance(val, Quantity):  # copy Quantities
                        setattr(self, _attr, type(val)(val))
                    else:
                        setattr(self, _attr, val)

    def _repr_helper(self, print_):
        if print_ is repr:
            opstr = '='
        else:
            opstr = ': '

        # get prefix and suffix
        prefix = '{}('.format(type(self).__name__)
        suffix = ')'
        if print_ is repr:
            prefix = '<{}'.format(prefix)
            suffix += '>'

        indent = ' ' * len(prefix)

        # format value
        arrstr = numpy.array2string(self.view(numpy.ndarray), separator=', ',
                                    prefix=prefix)

        # format unit
        metadata = [('unit', print_(self.unit) or 'dimensionless')]

        # format other metadata
        try:
            attrs = self._print_slots
        except AttributeError:
            attrs = self._metadata_slots
        for key in attrs:
            try:
                val = getattr(self, key)
            except (AttributeError, KeyError):
                val = None
            thisindent = indent + ' ' * (len(key) + len(opstr))
            metadata.append((
                key.lstrip('_'),
                print_(val).replace('\n', '\n{}'.format(thisindent)),
            ))
        metadata = (',\n{}'.format(indent)).join(
            '{0}{1}{2}'.format(key, opstr, value) for key, value in metadata)

        return "{0}{1}\n{2}{3}{4}".format(
            prefix, arrstr, indent, metadata, suffix)

    def __repr__(self):
        """Return a representation of this object
        This just represents each of the metadata objects appropriately
        after the core data array
        """
        return self._repr_helper(repr)

    def __str__(self):
        """Return a printable string format representation of this object
        This just prints each of the metadata objects appropriately
        after the core data array
        """
        return self._repr_helper(str)

    # name 
    @property
    def name(self):
        """Name for this data set
        """
        try:
            return self._name
        except AttributeError:
            self._name = None
            return self._name
        
    @name.setter
    def name(self, val):
        self._name = if_not_none(str, val)           

    @name.deleter
    def name(self):
        try:
            del self._name
        except AttributeError:
            pass

    def abs(self, axis=None, **kwargs):
        return self._wrap_function(numpy.abs, axis, **kwargs)
    abs.__doc__ = numpy.abs.__doc__

    def median(self, axis=None, **kwargs):
        return self._wrap_function(numpy.median, axis, **kwargs)
    median.__doc__ = numpy.median.__doc__

################### Series ###################
class Series(Array):
    """A one-dimensional data series
        Series([ 1., 2., 3., 2., 4., 3.]
            unit: Unit("nm"),
            name: None,
            epoch: None,
            channel: None,
            x0: 0.0 W,
            dx: 2.0 W,
            xindex: [  0.   2.   4.   6.   8.  10.] W)
        https://github.com/gwpy/gwpy/blob/v2.0.4/gwpy/types/series.py
    """
    _metadata_slots = Array._metadata_slots + ('x0', 'dx', 'xindex')
    _default_xunit = Unit('')
    _ndim = 1

    def __new__(cls, value, unit=None, x0=None, dx=None, xindex=None,
                xunit=None, **kwargs):
        # check input data dimensions are OK
        shape = numpy.shape(value)
        if len(shape) != cls._ndim:
            raise ValueError("Cannot generate %s with %d-dimensional data"
                             % (cls.__name__, len(shape)))

        # create new object
        new = super().__new__(cls, value, unit=unit, **kwargs)

        # set x-axis metadata from xindex
        if xindex is not None:
            # warn about duplicate settings
            if dx is not None:
                warnings.warn("xindex was given to %s(), dx will be ignored"
                     % cls.__name__)
            if x0 is not None:
                warnings.warn("xindex was given to %s(), x0 will be ignored"
                     % cls.__name__)
            # get unit
            if xunit is None and isinstance(xindex, Quantity):
                xunit = xindex.unit
            elif xunit is None:
                xunit = cls._default_xunit
            new.xindex = Quantity(xindex, unit=xunit)
        # or from x0 and dx
        else:
            if xunit is None and isinstance(dx, Quantity):
                xunit = dx.unit
            elif xunit is None and isinstance(x0, Quantity):
                xunit = x0.unit
            elif xunit is None:
                xunit = cls._default_xunit
            if dx is not None:
                new.dx = Quantity(dx, xunit)
            if x0 is not None:
                new.x0 = Quantity(x0, xunit)
        return new

    # -- series properties ----------------------

    def _update_index(self, axis, key, value):
        """Update the current axis index based on a given key or value
        This is an internal method designed to set the origin or step for
        an index, whilst updating existing Index arrays as appropriate
        Examples
        --------
        >>> self._update_index("x0", 0)
        >>> self._update_index("dx", 0)
        To actually set an index array, use `_set_index`
        """
        # delete current value if given None
        if value is None:
            return delattr(self, key)

        _key = "_{}".format(key)
        index = "{[0]}index".format(axis)
        unit = "{[0]}unit".format(axis)

        # convert float to Quantity
        if not isinstance(value, Quantity):
            try:
                value = Quantity(value, getattr(self, unit))
            except TypeError:
                value = Quantity(float(value), getattr(self, unit))

        # if value is changing, delete current index
        try:
            curr = getattr(self, _key)
        except AttributeError:
            delattr(self, index)
        else:
            if (
                    value is None
                    or getattr(self, key) is None
                    or not value.unit.is_equivalent(curr.unit)
                    or value != curr
            ):
                delattr(self, index)

        # set new value
        setattr(self, _key, value)
        return value

    def _set_index(self, key, index):
        """Set a new index array for this series
        """
        axis = key[0]
        origin = "{}0".format(axis)
        delta = "d{}".format(axis)
        if index is None:
            return delattr(self, key)
        if not isinstance(index, Index):
            try:
                unit = index.unit
            except AttributeError:
                unit = getattr(self, "_default_{}unit".format(axis))
            index = Index(index, unit=unit, copy=False)
        setattr(self, origin, index[0])
        if index.regular:
            setattr(self, delta, index[1] - index[0])
        else:
            delattr(self, delta)
        setattr(self, "_{}".format(key), index)

    def _index_span(self, axis):

        axisidx = ("x", "y", "z").index(axis)
        unit = getattr(self, "{}unit".format(axis))
        try:
            delta = getattr(self, "d{}".format(axis)).to(unit).value
        except AttributeError:  # irregular xindex
            index = getattr(self, "{}index".format(axis))
            try:
                delta = index.value[-1] - index.value[-2]
            except IndexError:
                raise ValueError("Cannot determine x-axis stride (dx)"
                                 "from a single data point")
            return Segment(index.value[0], index.value[-1] + delta)
        else:
            origin = getattr(self, "{}0".format(axis)).to(unit).value
            return Segment(origin, origin + self.shape[axisidx] * delta)

    # x0
    @property
    def x0(self):
        """X-axis coordinate of the first data point
        :type: `~astropy.units.Quantity` scalar
        """
        try:
            return self._x0
        except AttributeError:
            self._x0 = Quantity(0, self.xunit)
            return self._x0

    @x0.setter
    def x0(self, value):
        self._update_index("x", "x0", value)

    @x0.deleter
    def x0(self):
        try:
            del self._x0
        except AttributeError:
            pass

    # dx
    @property
    def dx(self):
        """X-axis sample separation
        :type: `~astropy.units.Quantity` scalar
        """
        try:
            return self._dx
        except AttributeError:
            try:
                self._xindex
            except AttributeError:
                self._dx = Quantity(1, self.xunit)
            else:
                if not self.xindex.regular:
                    raise AttributeError("This series has an irregular x-axis "
                                         "index, so 'dx' is not well defined")
                self._dx = self.xindex[1] - self.xindex[0]
            return self._dx

    @dx.setter
    def dx(self, value):
        self._update_index("x", "dx", value)

    @dx.deleter
    def dx(self):
        try:
            del self._dx
        except AttributeError:
            pass

    # xindex
    @property
    def xindex(self):
        """Positions of the data on the x-axis
        :type: `~astropy.units.Quantity` array
        """
        try:
            return self._xindex
        except AttributeError:
            self._xindex = Index.define(self.x0, self.dx, self.shape[0])
            return self._xindex

    @xindex.setter
    def xindex(self, index):
        self._set_index("xindex", index)

    @xindex.deleter
    def xindex(self):
        try:
            del self._xindex
        except AttributeError:
            pass

    # xunit
    @property
    def xunit(self):
        """Unit of x-axis index
        :type: `~astropy.units.Unit`
        """
        try:
            return self._dx.unit
        except AttributeError:
            try:
                return self._x0.unit
            except AttributeError:
                return self._default_xunit

    @xunit.setter
    def xunit(self, unit):
        unit = Unit(unit)
        try:  # set the index, if present
            self.xindex = self._xindex.to(unit)
        except AttributeError:  # or just set the start and step
            self.dx = self.dx.to(unit)
            self.x0 = self.x0.to(unit)

    @property
    def xspan(self):
        """X-axis [low, high) segment encompassed by these data
        :type: `~gwpy.segments.Segment`
        """
        return self._index_span("x")


################### Array2D ###################
class Array2D(Series):
    """A two-dimensional array with metadata
    """
    _metadata_slots = Series._metadata_slots + ('y0', 'dy', 'yindex')
    _default_xunit = Unit('')
    _default_yunit = Unit('')
    _rowclass = Series
    _columnclass = Series
    _ndim = 2

    def __new__(cls, data, unit=None,
                x0=None, dx=None, xindex=None, xunit=None,
                y0=None, dy=None, yindex=None, yunit=None, **kwargs):
        """Define a new `Array2D`
        """

        # create new object
        new = super().__new__(cls, data, unit=unit, xindex=xindex,
                              xunit=xunit, x0=x0, dx=dx, **kwargs)

        # set y-axis metadata from yindex
        if yindex is not None:
            # warn about duplicate settings
            if dy is not None:
                warnings.warn("yindex was given to %s(), dy will be ignored"
                     % cls.__name__)
            if y0 is not None:
                warnings.warn("yindex was given to %s(), y0 will be ignored"
                     % cls.__name__)
            # get unit
            if yunit is None and isinstance(yindex, Quantity):
                yunit = yindex.unit
            elif yunit is None:
                yunit = cls._default_yunit
            new.yindex = Quantity(yindex, unit=yunit)
        # or from y0 and dy
        else:
            if yunit is None and isinstance(dy, Quantity):
                yunit = dy.unit
            elif yunit is None and isinstance(y0, Quantity):
                yunit = y0.unit
            elif yunit is None:
                yunit = cls._default_yunit
            if dy is not None:
                new.dy = Quantity(dy, yunit)
            if y0 is not None:
                new.y0 = Quantity(y0, yunit)

        return new

    # -- Array2d properties ---------------------

    # y0
    @property
    def y0(self):
        """Y-axis coordinate of the first data point
        :type: `~astropy.units.Quantity` scalar
        """
        try:
            return self._y0
        except AttributeError:
            self._y0 = Quantity(0, self.yunit)
            return self._y0

    @y0.setter
    def y0(self, value):
        self._update_index("y", "y0", value)

    @y0.deleter
    def y0(self):
        try:
            del self._y0
        except AttributeError:
            pass

    # dy
    @property
    def dy(self):
        """Y-axis sample separation
        :type: `~astropy.units.Quantity` scalar
        """
        try:
            return self._dy
        except AttributeError:
            try:
                self._yindex
            except AttributeError:
                self._dy = Quantity(1, self.yunit)
            else:
                if not self.yindex.regular:
                    raise AttributeError(
                        "This series has an irregular y-axis "
                        "index, so 'dy' is not well defined")
                self._dy = self.yindex[1] - self.yindex[0]
            return self._dy

    @dy.setter
    def dy(self, value):
        self._update_index("y", "dy", value)

    @dy.deleter
    def dy(self):
        try:
            del self._dy
        except AttributeError:
            pass

    @property
    def yunit(self):
        """Unit of Y-axis index
        :type: `~astropy.units.Unit`
        """
        try:
            return self._dy.unit
        except AttributeError:
            try:
                return self._y0.unit
            except AttributeError:
                return self._default_yunit

    # yindex
    @property
    def yindex(self):
        """Positions of the data on the y-axis
        :type: `~astropy.units.Quantity` array
        """
        try:
            return self._yindex
        except AttributeError:
            self._yindex = Index.define(self.y0, self.dy, self.shape[1])
            return self._yindex

    @yindex.setter
    def yindex(self, index):
        self._set_index("yindex", index)

    @yindex.deleter
    def yindex(self):
        try:
            del self._yindex
        except AttributeError:
            pass

    @property
    def yspan(self):
        """Y-axis [low, high) segment encompassed by these data
        :type: `~gwpy.segments.Segment`
        """
        return self._index_span("y")

    @property
    def T(self):
        trans = self.value.T.view(type(self))
        trans.__array_finalize__(self)
        if hasattr(self, '_xindex'):
            trans.yindex = self.xindex.view()
        else:
            trans.y0 = self.x0
            trans.dy = self.dx
        if hasattr(self, '_yindex'):
            trans.xindex = self.yindex.view()
        else:
            trans.x0 = self.y0
            trans.dx = self.dy
        return trans