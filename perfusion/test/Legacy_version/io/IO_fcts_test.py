import pytest
from dolfin import *


from perfusion.test.mesh_file_generation import *
from perfusion.src.Legacy_version.io.IO_fcts import (
    mesh_reader
)


# Test for mesh_reader
# In test_mesh_reader
def test_mesh_reader(hdf5_mesh_file):
    mesh_file_path = hdf5_mesh_file["file"]
    read_mesh, read_subdomains, read_boundaries = mesh_reader(mesh_file_path)

    print(f"Type of read_subdomains: {type(read_subdomains)}")
    print(f"Type of dolfin.MeshFunction: {type(MeshFunction)}")
    print(f"Is read_subdomains an instance of dolfin.MeshFunction? {isinstance(read_subdomains, MeshFunction)}")


    # Check that mesh properties match
    assert read_mesh.num_vertices() == hdf5_mesh_file["mesh"].num_vertices()
    assert read_mesh.num_cells() == hdf5_mesh_file["mesh"].num_cells()
    assert read_subdomains.dim() == hdf5_mesh_file["mesh"].topology().dim()
    assert read_boundaries.dim() == hdf5_mesh_file["mesh"].topology().dim() - 1

    # Check coordinates (accounting for potential float precision issues)
    assert np.allclose(read_mesh.coordinates(), hdf5_mesh_file["mesh"].coordinates())

    # Check MeshFunction values (if they were explicitly set beyond default zeros)
    assert np.array_equal(read_subdomains.array(), hdf5_mesh_file["subdomains"].array())
    assert np.array_equal(read_boundaries.array(), hdf5_mesh_file["boundaries"].array())
