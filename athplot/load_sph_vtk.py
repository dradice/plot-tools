# AthPlot: manipulate and plot AthenaK binary data
# Copyright (C) 2025, Jacob Fields <fields@ias.edu>
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from vtk.util import numpy_support
from vtk import vtkStructuredGridReader

class SphericalData:
  """
    This class represents a VTK spherical data dump from AthenaK.

    Since AthenaK PR #772, a single sph output block can hold multiple radii
    (<output*>/radii = ...), in which case the file contains all of them
    stacked together (one combined file per variable per dump, instead of one
    file per radius). Pass `radius=<value>` (nearest match) or
    `radius_index=<int>` to select which radius' data to load; both are
    ignored (no selection needed) for a single-radius file.

    Note that this utility relies on the VTK python package.
  """
  def __init__(self, filename, encoding='latin_1', radius=None, radius_index=None):
    # Extract metadata stored as a comment in the second line
    string = ""
    with open(filename, 'r', encoding=encoding) as f:
        l = 0
        for line in f:
            if l == 1:
                string = line.strip()
                break
            else:
                l += 1

    substrings = string.split()
    for s in substrings:
        if s[:5] == 'time=':
            self.time = float(s[5:])
        if s[:6] == 'cycle=':
            self.cycle = int(s[6:])
        if s[:3] == 'xc=':
            self.xc = float(s[3:])
        if s[:3] == 'yc=':
            self.yc = float(s[3:])
        if s[:3] == 'zc=':
            self.zc = float(s[3:])

    # Open the file with the VTK reader
    reader = vtkStructuredGridReader()

    reader.SetFileName(filename)
    reader.ReadAllScalarsOn()

    reader.update()

    # Reorganize the data into separate r, theta, and phi quantities.
    # DIMENSIONS is (nradii, ntheta, 2*ntheta): radius is the fastest-varying
    # point/scalar index, so the flat layout is angle-major, radius-minor
    # (flat index = angle_index * nradii + radius_index).
    dims = [0, 0, 0]
    reader.GetOutput().GetDimensions(dims)
    self.nradii = dims[0]
    self.shape = (dims[2], dims[1])

    # Radii present in the file, written explicitly as FieldData (in the same
    # order as the radius-minor index above). Older single-radius files that
    # predate PR #772 won't have this array.
    radii_vtk = reader.GetOutput().GetFieldData().GetArray('RADII')
    self.radii = numpy_support.vtk_to_numpy(radii_vtk) if radii_vtk is not None else None

    if self.nradii == 1:
      ridx = 0
    elif radius_index is not None:
      ridx = radius_index
    elif radius is not None:
      if self.radii is None:
        raise RuntimeError(f"{filename}: cannot select radius={radius}: no RADII "
                            "field data found in file")
      ridx = int(np.argmin(np.abs(self.radii - radius)))
    else:
      raise RuntimeError(f"{filename}: contains {self.nradii} radii; pass "
                          "radius=<value> or radius_index=<int> to select one")

    cells_vtk = reader.GetOutput().GetPoints().GetData()
    cells = numpy_support.vtk_to_numpy(cells_vtk)

    self.r = self.radii[ridx] if self.radii is not None else cells[ridx, 0]

    sel = slice(ridx, None, self.nradii)

    self.theta = cells[sel, 1]
    self.theta = np.reshape(self.theta, self.shape)

    self.phi = cells[sel, 2]
    self.phi = np.reshape(self.phi, self.shape)

    # Read in the field data
    nfields = reader.GetOutput().GetPointData().GetNumberOfArrays()

    self.field_names = []
    self.fields = {}
    for i in range(nfields):
      field = reader.GetOutput().GetPointData().GetArrayName(i)
      self.field_names.append(field)
      field_data = reader.GetOutput().GetPointData().GetArray(i)
      full = numpy_support.vtk_to_numpy(field_data)
      self.fields[field] = np.reshape(full[sel], self.shape)

  def plot_data(self, var, ax, cmap='viridis', norm=None, vmin=None, vmax=None):
    # Note that phi is periodic, so for plotting purposes we need to append an extra
    # column of data to get a closed surface
    r = self.r
    theta = np.append(self.theta, [self.theta[0]], axis=0)
    phi = np.append(self.phi, [2*np.pi*np.ones(self.shape[1])], axis=0)

    data = np.copy(self.fields[var])
    data = np.append(data, [data[0]], axis=0)

    X = r*np.sin(theta)*np.cos(phi)
    Y = r*np.sin(theta)*np.sin(phi)
    Z = r*np.cos(theta)

    if norm != None:
      if vmin != None:
        raise RuntimeError("Cannot specify vmin and norm at the same time!")
      if vmax != None:
        raise RuntimeError("Cannot specify vmax and norm at the same time!")
      vmin = norm.vmin
      vmax = norm.vmax

    if vmin==None:
      vmin = data.min()
    if vmax==None:
      vmax = data.max()

    if norm == None:
      norm = Normalize(vmin=vmin, vmax=vmax)

    plot_data = norm(data)

    colors = plt.get_cmap(cmap)(plot_data)

    surf = ax.plot_surface(X, Y, Z, shade=False, facecolors=colors)

    sm = ScalarMappable(norm=norm, cmap=cmap)

    return surf, sm
