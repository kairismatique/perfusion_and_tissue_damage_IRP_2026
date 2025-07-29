import pytest
from dolfin import *


from perfusion.test.mesh_file_generation import *
from perfusion.src.Legacy_version.utils.finite_element_fcts import (
    mesh_reader,
    allocation_functions_space,
    read_boundary_conditions,
    apply_dirichlet_BC,
    apply_neumann_BC,
    apply_mixed_BC,
    set_up_fe_solver,
    solve_lin_sys
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


# Tests for allocation_functions_space
@pytest.mark.parametrize("model_type, fe_degr, vel_order, expected_Vp_sub_elements", [
    ('acv', 1, 0, 3),  # acv model, lowest degree for velocity DG0
    ('acv', 2, 1, 3),  # acv model, higher degree
    ('a', 1, 0, 0),  # a model, lowest degree for velocity DG0
    ('a', 2, 1, 0),  # a model, higher degree
])
def test_allocation_functions_space_valid_models(hdf5_mesh_file, model_type, fe_degr, vel_order,
                                                 expected_Vp_sub_elements):
    mesh = hdf5_mesh_file["mesh"]
    boundaries = hdf5_mesh_file["boundaries"]

    Vp, Vvel, v_1, v_2, v_3, p, p_1, p_2, p_3, K1_space, K2_space = \
        allocation_functions_space(mesh, fe_degr, model_type=model_type, vel_order=vel_order)

    if model_type == 'acv':
        # Mixed element: expect 3 subspaces of scalar functions
        assert isinstance(Vp.ufl_element(), MixedElement)
        assert Vp.num_sub_spaces() == expected_Vp_sub_elements
        assert all(Vp.sub(i).ufl_element().value_shape() == () for i in range(expected_Vp_sub_elements))
        assert Vp.sub(0).ufl_element().degree() == fe_degr
    elif model_type == 'a':
        # Scalar Lagrange space
        assert not isinstance(Vp.ufl_element(), MixedElement)
        assert Vp.ufl_element().value_shape() == ()
        assert Vp.ufl_element().degree() == fe_degr
        assert v_2 == []
        assert v_3 == []
        assert p_1 == []
        assert p_2 == []
        assert p_3 == []

    # Velocity space checks
    if vel_order == 0:
        assert Vvel.ufl_element().family() in ['DG', 'Discontinuous Lagrange']
    else:
        assert Vvel.ufl_element().family() == 'Lagrange'
    assert Vvel.ufl_element().degree() == vel_order

    # Permeability space checks
    assert K1_space.ufl_element().family() in ['DG', 'Discontinuous Lagrange']
    assert K1_space.ufl_element().degree() == 0
    assert K2_space.ufl_element().family() in ['DG', 'Discontinuous Lagrange']
    assert K2_space.ufl_element().degree() == 0


def test_allocation_functions_space_unknown_model(hdf5_mesh_file):
    mesh = hdf5_mesh_file["mesh"]
    with pytest.raises(Exception, match="Unknown model type: invalid"):
        allocation_functions_space(mesh, 1, model_type='invalid')


# Tests for read_boundary_conditions
@pytest.fixture
def bc_csv_multi_row(tmp_path):
    csv_content = """label,flow_rate,pressure_val,dummy1,bc_type
10,0.01,100.0,0,0
11,0.02,200.0,0,1
"""
    file_path = tmp_path / "bc_multi.csv"
    file_path.write_text(csv_content)
    return str(file_path)


@pytest.fixture
def bc_csv_single_row(tmp_path):
    csv_content = """label,flow_rate,pressure_val,dummy1,bc_type
10,0.01,100.0,0,0
"""
    file_path = tmp_path / "bc_single.csv"
    file_path.write_text(csv_content)
    return str(file_path)


def test_read_boundary_conditions(bc_csv_multi_row, bc_csv_single_row):
    data, b1, labels, data_dimension_greater_than_1 = read_boundary_conditions(bc_csv_multi_row)

    assert isinstance(data, np.ndarray)
    assert data.shape == (2, 5)
    assert np.allclose(data[0], [10, 0.01, 100.0, 0, 0])
    assert np.allclose(data[1], [11, 0.02, 200.0, 0, 1])

    assert np.allclose(b1, [1000 * 0.01, 1000 * 0.02])

    assert isinstance(labels, list)
    assert labels == [10.0, 11.0]

    assert data_dimension_greater_than_1 is True

    data, b1, labels, data_dimension_greater_than_1 = read_boundary_conditions(bc_csv_single_row)

    assert isinstance(data, np.ndarray)
    assert data.shape == (5,)
    assert np.allclose(data, [10, 0.01, 100.0, 0, 0])

    assert np.allclose(b1, [1000 * 0.01])

    assert isinstance(labels, list)
    assert labels == [10.0]

    assert data_dimension_greater_than_1 is False



# Tests for apply_dirichlet_BC
def test_apply_dirichlet_BC(hdf5_mesh_file):
    mesh = hdf5_mesh_file["mesh"]
    boundaries = hdf5_mesh_file["boundaries"]
    V_space = FunctionSpace(mesh, "Lagrange", 1)

    # Test multi-dimensional data
    bc_data_multi = np.array([[10, 0.01, 100.0, 0, 0]])
    bcs_list_multi = []
    updated_bcs_multi = apply_dirichlet_BC(V_space, bc_data_multi, bcs_list_multi, boundaries, True, 0, 1)
    assert len(updated_bcs_multi) == 1
    assert isinstance(updated_bcs_multi[0], DirichletBC)

    # Test single-dimensional data
    bc_data_single = np.array([10, 0.01, 200.0, 0, 0])
    bcs_list_single = []
    updated_bcs_single = apply_dirichlet_BC(V_space, bc_data_single, bcs_list_single, boundaries, False, 0, 2)
    assert len(updated_bcs_single) == 1
    assert isinstance(updated_bcs_single[0], DirichletBC)
