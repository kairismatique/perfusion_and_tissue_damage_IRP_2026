import pytest as pytest
from dolfin import *
import numpy as np

@pytest.fixture
def hdf5_mesh_file(tmp_path):
    def create_rotated_unit_cube_mesh(n=2, angle_deg=15):
        mesh = UnitCubeMesh(n, n, n)
        angle_rad = np.deg2rad(angle_deg)
        R = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0.0],
            [np.sin(angle_rad), np.cos(angle_rad), 0.0],
            [0.0, 0.0, 1.0]
        ])
        coords = mesh.coordinates()
        coords[:] = np.dot(coords, R.T)
        return mesh

    mesh = create_rotated_unit_cube_mesh()
    subdomains = MeshFunction("size_t", mesh, 3, 0)
    boundaries = MeshFunction("size_t", mesh, 2, 0)

    hdf5_path = tmp_path / "test_mesh_rotated.h5"
    hdf = HDF5File(MPI.comm_world, str(hdf5_path), "w")
    hdf.write(mesh, "/mesh")
    hdf.write(subdomains, "/subdomains")
    hdf.write(boundaries, "/boundaries")
    hdf.close()

    return {
        "mesh": mesh,
        "subdomains": subdomains,
        "boundaries": boundaries,
        "file": str(hdf5_path),
        "tmp_path": tmp_path
    }