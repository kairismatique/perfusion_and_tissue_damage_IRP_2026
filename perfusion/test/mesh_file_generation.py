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

    # Base path for the files
    base_path = tmp_path / "test_mesh_rotated"

    # Write the main mesh file
    mesh_xdmf_path = str(base_path) + ".xdmf"
    with XDMFFile(MPI.comm_world, mesh_xdmf_path) as file:
        file.write(mesh)

    # Write the subdomains file
    subdomains_xdmf_path = str(base_path) + '_physical_region.xdmf'
    with XDMFFile(MPI.comm_world, subdomains_xdmf_path) as file:
        file.write(subdomains)

    # Write the boundaries file
    boundaries_xdmf_path = str(base_path) + '_facet_region.xdmf'
    with XDMFFile(MPI.comm_world, boundaries_xdmf_path) as file:
        file.write(boundaries)

    return {
        "mesh": mesh,
        "subdomains": subdomains,
        "boundaries": boundaries,
        "file": mesh_xdmf_path,
        "tmp_path": tmp_path
    }