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

################### ligosegments ###################
class segment(tuple):
	"""
	The segment class defines objects that represent a range of values.
    https://github.com/duncanmmacleod/ligo-segments/blob/master/ligo/segments/__init__.py
	"""

	# basic class methods

	def __new__(cls, *args):
		if len(args) == 1:
			args = args[0]
		if len(args) != 2:
			raise TypeError("__new__() takes 2 arguments, or 1 argument when it is a sequence of length 2")
		if args[0] <= args[1]:
			return tuple.__new__(cls, args)
		else:
			return tuple.__new__(cls, (args[1], args[0]))

	def __abs__(self):
		"""
		Returns the length of the interval represented by the
		segment.  Requires the bounds to support the subtract
		operation.
		"""
		return self[1] - self[0]

################### Segment ###################
class Segment(segment):
    """A tuple defining a semi-open interval ``[start, end)``
        https://github.com/gwpy/gwpy/blob/v2.0.4/gwpy/segments/segments.py
    """
    @property
    def start(self):
        """The GPS start time of this segment
        """
        return self[0]

    @property
    def end(self):
        """The GPS end time of this segment
        """
        return self[1]

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, self[0], self[1])

    def __str__(self):
        return "[%s ... %s)" % (self[0], self[1])

################### SegmentList ###################
class SegmentList(list):
    '''https://github.com/gwpy/gwpy/blob/v2.0.4/gwpy/segments/segments.py'''
    # -- representations ------------------------

    def __repr__(self):
        return "<SegmentList([%s])>" % "\n              ".join(map(repr, self))

    def __str__(self):
        return "[%s]" % "\n ".join(map(str, self))

del segment     # del ligosegments segment name
