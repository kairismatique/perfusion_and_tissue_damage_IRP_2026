import basix.ufl
from ufl import *
import ufl.finiteelement
import basix
import dolfinx
from dolfinx.fem.petsc import assemble_matrix, LinearProblem

from dolfinx import fem
from dolfinx.fem import *

from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
from tqdm import tqdm
from typing import List, Optional
import ufl

import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), './'))
sys.path.append(root_dir)
from ..io import IO_functions
from ..utils import suppl_fcts
import builtins

rank = MPI.COMM_WORLD.rank


def print0(*args, **kwargs):
    """stdout."""
    if rank == 0:
        builtins.print(*args, **kwargs)


def alloc_fct_spaces(mesh, fe_degr, **kwargs):
    """
    Allocate and return the function spaces required for the "a" (single-pressure)
    and "acv" (multi-pressure) perfusion models, along with associated UFL functions.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        Mesh object defining the spatial domain.
    fe_degr : int
        Polynomial degree for pressure elements.
    **kwargs :
        Optional keyword arguments:
            - model_type : str
                Either "a" (single-pressure) or "acv" (default).
            - vel_order : int
                Degree for the velocity finite elements (default: fe_degr - 1).

    Returns
    -------
    tuple
        (Vp, Vvel, v_1, v_2, v_3, p, p_1, p_2, p_3, K1_space, K2_space)
        Vp         : Pressure function space (scalar or mixed)
        Vvel       : Velocity vector function space
        v_1,2,3    : Components of test functions
        p, p_1,2,3 : Trial functions and their splits
        K1_space   : Tensor-valued permeability function space
        K2_space   : Scalar-valued permeability function space

    Notes
    -----
    - Supports DolfinX 0.9 with `basix.ufl.element` for consistent space creation.
    - Modular to easily extend for further model types.
    - MPI-aware printing via print0 is assumed available.
    """

    print0(f"alloc_fct_spaces: Initializing with fe_degr = {fe_degr}")

    # Extract optional arguments
    model_type = kwargs.get("model_type", "acv")
    vel_order = kwargs.get("vel_order", fe_degr - 1)

    # Mesh info
    cell_type = mesh.ufl_cell().cellname()
    mesh_dim = mesh.topology.dim
    num_cells = mesh.topology.index_map(0).size_local

    print0(f"alloc_fct_spaces: Mesh type = {cell_type}, dimension = {mesh_dim}")
    print0(f"alloc_fct_spaces: Number of cells = {num_cells}")
    print0(f"alloc_fct_spaces: Model type = '{model_type}', velocity order = {vel_order}")

    # --- Pressure space ---
    if model_type == "acv":
        print0("alloc_fct_spaces: Building mixed pressure space for 'acv' model")

        # Three pressure components (P1, P2, P3)
        scalar_p = basix.ufl.element("Lagrange", cell_type, fe_degr)
        mixed_p = basix.ufl.mixed_element([scalar_p] * 3)
        Vp = fem.functionspace(mesh, mixed_p)

        # Split test/trial functions
        v = ufl.TestFunction(Vp)
        v_1, v_2, v_3 = ufl.split(v)

        p = ufl.TrialFunction(Vp)
        p_1, p_2, p_3 = ufl.split(p)

    elif model_type == "a":
        print0("alloc_fct_spaces: Building scalar pressure space for 'a' model")

        Vp = fem.functionspace(mesh, ("CG", fe_degr))
        v = ufl.TestFunction(Vp)
        p = ufl.TrialFunction(Vp)

        # No splitting needed for single-pressure model
        v_1 = v
        v_2 = v_3 = None
        p_1 = p
        p_2 = p_3 = None

    else:
        raise ValueError(f"Unknown model type: '{model_type}'")

    print0(f"alloc_fct_spaces: Vp space dofs (global) = {Vp.dofmap.index_map.size_global}")

    # --- Permeability Tensor K1 ---
    print0("alloc_fct_spaces: Creating K1_space (tensor permeability)")
    k_scalar = basix.ufl.element("DG", cell_type, 0)
    k_tensor = basix.ufl.mixed_element([k_scalar] * (mesh_dim * mesh_dim))
    K1_space = fem.functionspace(mesh, k_tensor)
    print0(f"alloc_fct_spaces: K1_space dofs (global) = {K1_space.dofmap.index_map.size_global}")

    # --- Scalar Permeability K2 ---
    print0("alloc_fct_spaces: Creating K2_space (scalar permeability)")
    K2_element = basix.ufl.element("DG", cell_type, 0)
    K2_space = fem.functionspace(mesh, K2_element)
    print0(f"alloc_fct_spaces: K2_space dofs (global) = {K2_space.dofmap.index_map.size_global}")

    # --- Velocity space ---
    print0("alloc_fct_spaces: Creating Vvel (velocity space)")
    vel_element = basix.ufl.element("Lagrange" if vel_order > 0 else "DG", cell_type, vel_order, shape=(mesh_dim,))
    Vvel = fem.functionspace(mesh, vel_element)
    print0(f"alloc_fct_spaces: Vvel space dofs (global) = {Vvel.dofmap.index_map.size_global}")

    print0("alloc_fct_spaces: Allocation complete.")
    MPI.COMM_WORLD.Barrier()
    return Vp, Vvel, v_1, v_2, v_3, p, p_1, p_2, p_3, K1_space, K2_space


def set_up_fe_solver2(mesh: Optional[dolfinx.mesh],
                      subdomains: Optional[dolfinx.mesh.Mesh],
                      boundaries: Optional[dolfinx.mesh.Mesh],
                      V: Optional[dolfinx.fem.FunctionSpace],
                      v_1: Optional[ufl.Argument],
                      v_2: Optional[ufl.Argument],
                      v_3: Optional[ufl.Argument],
                      p: Optional[ufl.Coefficient],
                      p_1: Optional[ufl.Coefficient],
                      p_2: Optional[ufl.Coefficient],
                      p_3: Optional[ufl.Coefficient],
                      K1: Optional[ufl.Coefficient],
                      K2: Optional[ufl.Coefficient],
                      K3: Optional[ufl.Coefficient],
                      beta12: Optional[ufl.Coefficient],
                      beta23: Optional[ufl.Coefficient],
                      pa: float,
                      pv: float,
                      read_inlet_boundary: bool,
                      inlet_boundary_file: str,
                      inlet_BC_type: str,
                      **kwarg
                      ) -> tuple[fem.Form, fem.Form, fem.Constant, fem.Constant, fem.Constant, List[fem.DirichletBC]]:
    """Set up and configure the finite element solver for blood flow modeling.

    This function prepares the finite element system for either:
    - ACV model (arterial-capillary-venous) with three compartments
    - A model (arterial-only) with single compartment

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        Computational mesh
    subdomains : dolfinx.mesh.MeshTags
        Subdomain markers for different tissue types
    boundaries : dolfinx.mesh.MeshTags
        Boundary markers for different boundary conditions
    V : dolfinx.fem.FunctionSpace
        Function space for the solution
    v_1, v_2, v_3 : ufl.Argument
        Test functions for each compartment
    p, p_1, p_2, p_3 : ufl.Coefficient
        Trial functions for pressure in each compartment
    K1, K2, K3 : ufl.Coefficient
        Conductivity tensors for each compartment
    beta12, beta23 : ufl.Coefficient
        Coupling coefficients between compartments
    pa : float
        Arterial pressure value
    pv : float
        Venous pressure value
    read_inlet_boundary : bool
        Flag to read boundary conditions from file
    inlet_boundary_file : str
        Path to boundary condition file
    inlet_BC_type : str
        Type of inlet boundary condition ("DBC", "NBC", or "mixed")
    **kwargs
        Additional parameters including:
        - model_type: "acv" or "a" (default: "acv")

    Returns
    -------
    Tuple containing:
        - LHS: ufl.Form - Left-hand side of the variational problem
        - RHS: ufl.Form - Right-hand side of the variational problem
        - sigma1: fem.Constant - Source term for compartment 1
        - sigma2: fem.Constant - Source term for compartment 2
        - sigma3: fem.Constant - Source term for compartment 3
        - BCs: List[fem.DirichletBC] - List of boundary conditions

    Notes
    -----
    - Handles both Dirichlet (DBC) and Neumann (NBC) boundary conditions
    - Supports reading boundary conditions from file or using defaults
    - Includes extensive debugging output
    - Parallel-safe printing with rank checking
    """

    # Initialize MPI communication
    model_type = kwarg.get("model_type", "acv")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = 0

    # Helper function for parallel-safe printing
    def print0(*args):
        if rank == 0:
            print(*args)

    # Initialize source terms (set to zero)
    sigma1 = fem.Constant(mesh, 0.0)
    sigma2 = fem.Constant(mesh, 0.0)
    sigma3 = fem.Constant(mesh, 0.0)
    b1 = []  # Used in reading BC_data, Will store boundary condition data

    if model_type == "acv":

        BCs = []
        integrals_N = []
        if read_inlet_boundary == True:

            # Load boundary condition data from CSV file
            BC_data = np.loadtxt(inlet_boundary_file, skiprows=1, delimiter=',')
            b1 = 1000 * BC_data[:, 1] if BC_data.ndim > 1 else [1000 * BC_data[1]]
            print('b1 = ', b1)
            boundary_labels = list(BC_data[:, 0]) if BC_data.ndim > 1 else [BC_data[0]]
            n_labels = len(boundary_labels)
            for i in range(n_labels):
                boundary_id = boundary_labels[i]
                # Identify facets corresponding to the given boundary label.
                facet_indices = np.where(boundaries.values == boundary_id)[0]
                boundary_facets = boundaries.indices[facet_indices]

                # For NBC, we may also prescribe a Dirichlet condition on compartment 3 (venous).
                dofs_comp3 = fem.locate_dofs_topological(V.sub(2), mesh.topology.dim - 1, boundary_facets)
                bc_venous = fem.dirichletbc(fem.Constant(mesh, pv), dofs_comp3, V.sub(2))
                BCs.append(bc_venous)

            if inlet_BC_type == "DBC":
                # Dirichlet Boundary Condition loop
                for i in range(n_labels):
                    boundary_id = boundary_labels[i]
                    n_labels = len(boundary_labels)
                    facet_indices = np.where(boundaries.values == boundary_id)[0]
                    boundary_facets = boundaries.indices[facet_indices]
                    # Apply to compartment 1 if DBC
                    value = BC_data[i, 2] if BC_data.ndim > 1 else BC_data[2]
                    bc_func1 = fem.Constant(mesh, value)
                    dofs0 = fem.locate_dofs_topological(V.sub(0), mesh.topology.dim - 1, boundary_facets)
                    bc1 = fem.dirichletbc(bc_func1, dofs0, V.sub(0))
                    BCs.append(bc1)


            elif inlet_BC_type == 'NBC':
                # Neumann boundary conditions: Setup the measure using ufl.ds with subdomain_data provided.

                # First, adjust the flux values by the area associated with each boundary label.
                for i in range(n_labels):
                    boundary_id = int(boundary_labels[i])
                    # Define a form to compute the area of the boundary.
                    area_form = fem.form(fem.Constant(mesh, 1.0) * ufl.dS)
                    area = fem.assemble_scalar(area_form)
                    b1[i] = b1[i] / area

                # Next, compute the Neumann boundary integrals.
                # Here, ufl.avg is used to properly handle the discretization for v_1.
                for i in range(n_labels):
                    if b1[i] > 0:
                        integrals_N.append(b1[i] * ufl.avg(v_1) * ufl.dS)

            elif inlet_BC_type == 'mixed':
                # Neumann boundary conditions
                for i in range(n_labels):
                    boundary_id = boundary_labels[i]
                    n_labels = len(boundary_labels)
                    facet_indices = np.where(boundaries.values == boundary_id)[0]
                    boundary_facets = boundaries.indices[facet_indices]
                    if BC_data[i, 4] == 0:
                        value = BC_data[i, 2] if BC_data.ndim > 1 else BC_data[2]
                        bc_func1 = fem.Constant(mesh, value)
                        dofs0 = fem.locate_dofs_topological(V.sub(0), mesh.topology.dim - 1, boundary_facets)
                        bc1 = fem.dirichletbc(bc_func1, dofs0, V.sub(0))
                        BCs.append(bc1)
                    elif BC_data[i, 4] == 1:  # Neumann (pressure gradient ~ flux) boundary condition

                        area_form = fem.form(fem.Constant(mesh, 1.0) * ufl.dS)
                        area = fem.assemble_scalar(area_form)
                        b1[i] = b1[i] / area
                    else:
                        raise Exception("Boundary condition in the BCs.csv file must be 0 (Dirichlet) or 1 (Neumann)")

                    # b1 includes the average surface-normal velocity for the perfusion regions
                    # computed from the volumentric flow rate [mm^3/s] and the surface area [mm^2]

                if BC_data[i, 4] == 1:  # Neumann (pressure gradient ~ flux) boundary condition
                    if b1[i] > 0:
                        integrals_N.append(b1[i] * ufl.avg(v_1) * ufl.dS)
            else:
                raise Exception("inlet_BC_type must be Neumann or Dirichlet ('NBC' or 'DBC')")
        else:
            # Get boundary labels and number from your region_label_assembler
            boundary_labels, n_labels = suppl_fcts.region_label_assembler(boundaries)
            if inlet_BC_type == 'DBC':
                # Loop through the labels
                for i in range(n_labels):
                    label = boundary_labels[i]
                    facet_indices = np.where(boundaries.values == label)[0]
                    boundary_facets = boundaries.indices[facet_indices]

                    if label > 2:
                        dofs0 = fem.locate_dofs_topological(V.sub(0), mesh.topology.dim - 1, boundary_facets)
                        dofs2 = fem.locate_dofs_topological(V.sub(2), mesh.topology.dim - 1, boundary_facets)
                        bc_func1 = fem.Constant(mesh, pa)
                        bc_func2 = fem.Constant(mesh, pv)

                        bc1 = fem.dirichletbc(bc_func1, dofs0, V.sub(0))
                        bc2 = fem.dirichletbc(bc_func2, dofs2, V.sub(2))
                        BCs.extend([bc1, bc2])

            if inlet_BC_type == 'NBC':
                print0(f"Inlet Type must be DBC for model 'a' if 'read inlet Boundary: False'")
                exit()
            if inlet_BC_type == 'mixed':
                print0(f"Inlet Type must be DBC for model 'a' if 'read inlet Boundary: False'")
                exit()
        # Compute K1_tensor (as in model 'a')
        # Build the permeability tensors (assuming mesh_dim is defined and K1, K2, K3 have been read from file)

        mesh_dim = mesh.topology.dim
        K1_components = ufl.split(K1)
        K1_tensor = ufl.as_matrix([[K1_components[i * mesh_dim + j] for j in range(mesh_dim)]
                                   for i in range(mesh_dim)])

        K2_tensor = ufl.Identity(mesh_dim) * K2

        K3_components = ufl.split(K3)
        K3_tensor = ufl.as_matrix([[K3_components[i * mesh_dim + j] for j in range(mesh_dim)]
                                   for i in range(mesh_dim)])
        # Define the variational form for the ACV model.
        # p_1, p_2, p_3 are the split components of the mixed pressure trial function,
        # and v_1, v_2, v_3 are the corresponding split test functions.

        LHS = fem.form(
            ufl.inner(K1_tensor * ufl.grad(p_1), ufl.grad(v_1)) * ufl.dx +
            beta12 * (p_1 - p_2) * v_1 * ufl.dx +
            ufl.inner(K2_tensor * ufl.grad(p_2), ufl.grad(v_2)) * ufl.dx +
            beta12 * (p_2 - p_1) * v_2 * ufl.dx +
            beta23 * (p_2 - p_3) * v_2 * ufl.dx +
            ufl.inner(K3_tensor * ufl.grad(p_3), ufl.grad(v_3)) * ufl.dx +
            beta23 * (p_3 - p_2) * v_3 * ufl.dx
        )

        RHS = fem.form(
            sigma1 * v_1 * ufl.dx +
            sigma2 * v_2 * ufl.dx +
            sigma3 * v_3 * ufl.dx +
            sum(integrals_N)
        )

        print("DEBUG: RHS form:", RHS)

        print("DEBUG: Assembling LHS...")
        a = fem.petsc.assemble_matrix(LHS)
        a.assemble()

        print("DEBUG: Assembling RHS to test non-zero contributions...")
        b = fem.petsc.assemble_vector(RHS)
        print('b created')
        b.assemble()




    elif model_type == "a":
        # Arterial-only model configuration
        print('pa', pa)
        print('pv', pv)
        print('p', p)
        p_a = fem.Constant(mesh, pa)
        p_venous = fem.Constant(mesh, pv)
        BCs = []
        integrals_N = []
        if read_inlet_boundary == True:
            # Similar boundary condition handling as ACV model but simplified
            BC_data = np.loadtxt(inlet_boundary_file, skiprows=1, delimiter=',')
            b1 = 1000 * BC_data[:, 1] if BC_data.ndim > 1 else [1000 * BC_data[1]]
            boundary_labels = list(BC_data[:, 0]) if BC_data.ndim > 1 else [BC_data[0]]
            n_labels = len(boundary_labels)

            # Dirichlet Boundary Conditions
            if inlet_BC_type == "DBC":
                # Dirichlet Boundary Condition loop
                for i in range(n_labels):
                    boundary_id = boundary_labels[i]
                    n_labels = len(boundary_labels)
                    facet_indices = np.where(boundaries.values == boundary_id)[0]
                    boundary_facets = boundaries.indices[facet_indices]
                    # Apply to compartment 1 if DBC
                    value = BC_data[i, 2] if BC_data.ndim > 1 else BC_data[2]
                    bc_func = fem.Constant(mesh, value)
                    dofs = fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
                    bc = fem.dirichletbc(bc_func, dofs, V)
                    BCs.append(bc)
            # Neumann Boundary Conditions
            elif inlet_BC_type == "NBC":
                # Neumann BC handling
                for i, boundary_id in enumerate(boundary_labels):
                    boundary_id = int(boundary_id)

                    area_form = fem.form(fem.Constant(mesh, 1.0) * ufl.dS)
                    area = fem.assemble_scalar(area_form)
                    b1[i] = b1[i] / area
                for i, boundary_id in enumerate(boundary_labels):
                    if b1[i] > 0:
                        # Use avg(v_1) to properly handle DG space restrictions
                        integrals_N.append(b1[i] * ufl.avg(v_1) * ufl.dS)

            # Mixed Dirichlet and Neumann Boundary Conditions
            elif inlet_BC_type == 'mixed':
                n_labels = len(boundary_labels)
                for i in range(n_labels):
                    boundary_id = boundary_labels[i]
                    facet_indices = np.where(boundaries.values == boundary_id)[0]
                    boundary_facets = boundaries.indices[facet_indices]

                    # Defensive check
                    flag = BC_data[i, 4]
                    if flag == 0:  # Dirichlet BC
                        value = BC_data[i, 2] if BC_data.ndim > 1 else BC_data[2]
                        bc_func1 = fem.Constant(mesh, value)
                        dofs = fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
                        bc1 = fem.dirichletbc(bc_func1, dofs, V)
                        BCs.append(bc1)

                    elif flag == 1:  # Neumann BC (flux)
                        area_form = fem.form(fem.Constant(mesh, 1.0) * ufl.dS)
                        area = fem.assemble_scalar(area_form)
                        b1[i] = b1[i] / area

                    else:
                        raise Exception(f"BC flag must be 0 or 1, but got {flag} at row {i}")

                    # b1 includes the average surface-normal velocity for the perfusion regions
                    # computed from the volumentric flow rate [mm^3/s] and the surface area [mm^2]

                if BC_data[i, 4] == 1:  # Neumann (pressure gradient ~ flux) boundary condition
                    if b1[i] > 0:
                        integrals_N.append(b1[i] * ufl.avg(v_1) * ufl.dS)
            else:
                raise Exception("inlet_BC_type must be Neumann or Dirichlet ('NBC' or 'DBC')")

        else:
            # Get boundary labels and number from your region_label_assembler
            boundary_labels, n_labels = suppl_fcts.region_label_assembler(boundaries)
            if inlet_BC_type == 'DBC':
                # Loop through the labels
                for i in range(n_labels):
                    label = boundary_labels[i]
                    facet_indices = np.where(boundaries.values == label)[0]
                    boundary_facets = boundaries.indices[facet_indices]

                    if label > 2:
                        dofs = fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)

                        bc_func = fem.Constant(mesh, pa)

                        bc = fem.dirichletbc(bc_func, dofs, V)

                        BCs.append(bc)
            elif inlet_BC_type == 'NBC':
                print0("Inlet Type must be DBC for model 'a' if 'read inlet Boundary' is False")
                exit()
            elif inlet_BC_type == 'mixed':
                print0("Inlet Type must be DBC for model 'a' if 'read inlet Boundary' is False")
                exit()
            else:
                raise ValueError(f"Unknown inlet_BC_type '{inlet_BC_type}' for model 'a'")

        # Assemble arterial-only variational forms
        beta_total = 1 / (1 / beta12 + 1 / beta23)
        mesh_dim = mesh.topology.dim

        # Handle tensor-valued K1
        K1_components = ufl.split(K1)
        K1_tensor = ufl.as_matrix([[K1_components[i * mesh_dim + j] for j in range(mesh_dim)]
                                   for i in range(mesh_dim)])

        LHS = fem.form(
            ufl.inner(K1_tensor * ufl.grad(p), ufl.grad(v_1)) * ufl.dx +
            beta12 * p * v_1 * ufl.dx
        )
        print("DEBUG: LHS form:", LHS)

        print("debug: integrals_N", integrals_N)

        RHS = fem.form(
            sigma1 * v_1 * ufl.dx +
            sum(integrals_N) +
            beta_total * p_venous * v_1 * ufl.dx
        )
        print("DEBUG: RHS form:", RHS)

        print("DEBUG: Assembling LHS...")
        a = fem.petsc.assemble_matrix(LHS)
        a.assemble()
        # Print values for local rows only
        local_rows = range(a.getSize()[0])  # Get the size of the local matrix
        b = fem.petsc.assemble_vector(RHS)
        b.assemble()

    else:
        raise Exception("unknown model type: " + model_type)
    MPI.COMM_WORLD.Barrier()
    return LHS, RHS, sigma1, sigma2, sigma3, BCs


def solve_lin_sys(V: Optional[dolfinx.fem.FunctionSpace],
                  LHS: Optional[ufl.Form],
                  RHS: Optional[ufl.Form],
                  BCs: Optional[dolfinx.fem.DirichletBC],
                  lin_solver: str,
                  precond: str,
                  rtol: float,
                  mon_conv: bool,
                  init_sol: Optional[dolfinx.fem.Function],
                  inlet_BC_type: str,
                  **kwargs) -> Optional[dolfinx.fem.Function]:
    """Solve a linear system of equations using PETSc solver with configurable options.

    1. Validates input arguments and function space compatibility.
    2. Assembles the linear system (matrix A and vector b).
    3. Applies boundary conditions and performs lifting operations.
    4. Configures PETSc solver based on boundary condition type.
    5. Solves the system and monitors convergence.
    6. Returns the solution as a Function in the specified function space.

    Parameters
    ----------
    V : dolfinx.fem.FunctionSpace
        The function space for the solution.
    LHS : ufl.Form
        The left-hand side bilinear form (matrix A).
    RHS : ufl.Form
        The right-hand side linear form (vector b).
    BCs : list[dolfinx.fem.DirichletBC]
        List of Dirichlet boundary conditions to apply.
    lin_solver : str
        PETSc solver type (e.g., 'gmres', 'cg', 'bcgs').
    precond : str
        PETSc preconditioner type (e.g., 'jacobi', 'hypre').
    rtol : float
        Relative tolerance for solver convergence.
    mon_conv : bool
        Flag to monitor convergence during solving.
    init_sol : Optional[dolfinx.fem.Function]
        Initial guess for the solution (optional).
    inlet_BC_type : str
        Type of inlet boundary condition ('DBC' for Dirichlet, 'NBC' for Neumann).
    **kwargs
        Additional optional arguments:
        - model_type: str - Type of model being solved
        - anchor_pressure: bool - Whether to anchor pressure at a point

    Returns
    -------
    dolfinx.fem.Function
        The solution to the linear system in the specified function space.

    Notes
    -----
    - Includes extensive debugging output when called.
    - Automatically handles both Dirichlet and Neumann boundary conditions.
    - Uses PETSc's linear algebra backend for efficient solving.
    - Performs validation of function space compatibility before solving.
    """

    # Extract model type from kwargs or set to None if not provided
    model_type = kwargs.get("model_type", None)
    # Set boundary condition type from input (defaults to "DBC" if not provided)
    bc_type = inlet_BC_type  # defaults to "DBC" if not provided
    # Check if pressure anchoring is requested (default False)
    anchor_pressure = kwargs.get("anchor_pressure", False)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print0("\n===================== solve_lin_sys DEBUG START =====================")
    # --- DEBUG: Argument Types ---
    print0(f"[DEBUG] V                  : {type(V)}")  # Function space type
    print0(f"[DEBUG] LHS                : {type(LHS)}")  # Left-hand side form type
    print0(f"[DEBUG] RHS                : {type(RHS)}")  # Right-hand side form type
    print0(f"[DEBUG] BCs                : {type(BCs)}, len = {len(BCs)}")  # Boundary conditions
    print0(f"[DEBUG] lin_solver         : {lin_solver} ({type(lin_solver)})")  # Solver type
    print0(f"[DEBUG] precond            : {precond} ({type(precond)})")  # Preconditioner type
    print0(f"[DEBUG] rtol               : {rtol} ({type(rtol)})")  # Relative tolerance
    print0(f"[DEBUG] mon_conv           : {mon_conv} ({type(mon_conv)})")  # Monitor flag
    print0(f"[DEBUG] init_sol           : {type(init_sol)}")  # Initial solution
    print0(f"[DEBUG] kwargs             : {kwargs}")  # Additional arguments

    # --- DEBUG: Function Space Compatibility ---
    try:
        # Get function spaces from LHS and RHS forms
        lhs_space = LHS.function_spaces[0]
        rhs_space = RHS.function_spaces[0]
        print0(f"[DEBUG] LHS.function_space[0]: {lhs_space}")
        print0(f"[DEBUG] RHS.function_space[0]: {rhs_space}")
        print0(f"[DEBUG] V                    : {V}")
        # Check if elements match
        if hasattr(lhs_space, "ufl_element") and hasattr(V, "ufl_element"):
            print0(f"[DEBUG] LHS.element == V.element? {lhs_space.ufl_element() == V.ufl_element()}")
            print0(f"[DEBUG] RHS.element == V.element? {rhs_space.ufl_element() == V.ufl_element()}")
        # Verify spaces match
        assert lhs_space == V, "[DEBUG] LHS test space mismatch with V"
        assert rhs_space == V, "[DEBUG] RHS test space mismatch with V"
    except Exception as e:
        print0(f"[ERROR] Function space validation failed: {e}")

    # --- DEBUG: Boundary Conditions ---
    for bc_id, bc in enumerate(BCs):
        dofs = bc.function_space.dofmap.index_map.local_range  # This gets the local range of DoFs

        try:
            bc_func = bc.function
            # Print first 5 BC function values if available
        except AttributeError:
            pass
        if hasattr(bc, 'get_boundary_dofs'):
            boundary_dofs = bc.get_boundary_dofs()

        else:
            pass

    # --- Assemble system ---
    print0("[DEBUG] Assembling matrix A...")
    A = dolfinx.fem.petsc.assemble_matrix(LHS, bcs=BCs)  # Create PETSc matrix
    A.assemble()  # Finalize matrix assembly
    print0(f"[DEBUG] Matrix A norm: {A.norm():.3e}")

    print0("[DEBUG] Assembling vector b...")
    b = dolfinx.fem.petsc.assemble_vector(RHS)  # Create PETSc vector
    print0(f"[DEBUG] RHS norm before lifting: {b.norm():.3e}")
    # Apply lifting operation for BCs
    dolfinx.fem.petsc.apply_lifting(b, [LHS], [BCs])
    # Update ghost values
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    # Apply BCs to vector
    dolfinx.fem.set_bc(b, BCs)
    print0(f"[DEBUG] RHS norm after lifting: {b.norm():.3e}")
    print0(f"[DEBUG] First 10 RHS entries: {b.getArray()[:10]}")

    # --- Initialize solution vector ---
    x = b.duplicate()  # Create vector of same size as b
    x.set(0.0)  # Zero initial guess
    print0(f"[DEBUG] Initial solution norm: {x.norm():.3e}")

    # --- PETSc Solver Configuration ---
    solver = PETSc.KSP().create()  # Create solver object
    solver.setOperators(A)  # Set system matrix
    # Use the lin_solver argument provided
    solver.setType(lin_solver)
    solver.getPC().setType(precond)
    solver.setTolerances(rtol=1e-20, atol=1e-10)
    solver.setFromOptions()

    # --- Select Solver Based on BC Type ---
    if bc_type == "NBC":
        print0("[DEBUG] Using NBC-specific solver configuration...")
        # For NBC problems, you might choose an alternative solver type (e.g. gcr instead of bcgs)
        solver.setType("bcgs")  # Switch to a different Krylov solver type for NBC problems
        solver.getPC().setType("hypre")  # Set the hypre preconditioner
        # Set tighter tolerances (or adjust as necessary)
        solver.setTolerances(rtol=1e-20, atol=1e-10)
    elif bc_type == "DBC":
        print0("[DEBUG] Using DBC-specific solver configuration...")
        # For DBC problems, use the provided settings
        solver.setType(lin_solver)
        solver.getPC().setType(precond)
    else:
        print0("[DEBUG] Unrecognized bc_type provided; using default solver configuration.")
        solver.setType(lin_solver)
        solver.getPC().setType(precond)

    print0(f"[DEBUG] Solver type         : {solver.getType()}")
    print0(f"[DEBUG] Preconditioner type : {solver.getPC().getType()}")
    print0(f"[DEBUG] Relative tolerance  : {rtol}")

    # --- (Optional) Set a Monitor to print residuals during solve ---
    def ksp_monitor(ksp, it, rnorm):
        print0(f"[DEBUG] Iteration {it}: Residual norm = {rnorm:.3e}")

    solver.setMonitor(ksp_monitor)

    # --- Solve system ---
    print0("[DEBUG] Solving linear system...")
    solver.solve(b, x)
    # You can also enable PETSc's built-in monitor by uncommenting the next line:
    # PETSc.Options.setValue("ksp_monitor", "")

    print0("[DEBUG] Solver converged reason : {solver.getConvergedReason()}")
    print0("[DEBUG] Solver iterations       : {solver.getIterationNumber()}")
    print0("[DEBUG] Final residual norm     : {solver.getResidualNorm():.3e}")
    print0("[DEBUG] First 10 entries of x   : {x.getArray()[:10]}")
    print0("[DEBUG] Final solution norm     : {x.norm():.3e}")

    # --- Create Function from solution vector ---
    u = Function(V)  # Initialize function in space V

    # Assign solution to owned DoFs only
    u.x.array[:x.getOwnershipRange()[1] - x.getOwnershipRange()[0]] = x.array

    # Update ghost values
    u.x.scatter_forward()

    print0("===================== solve_lin_sys DEBUG END =======================\n")
    MPI.COMM_WORLD.Barrier()
    return u


def func_space(mesh, eleD, configs):
    # fenite element type and solution function space
    P = basix.ufl.element(
        "Lagrange",  # family (positional argument)
        "tetrahedron",  # cell (positional argument)
        1,  # order
        shape=(3,)  # degree (positional argument)
    )
    DG = basix.ufl.element(
        "DG",  # family (positional argument)
        "tetrahedron",  # cell (positional argument)
        0,  # order
        shape=(3,)  # degree (positional argument)
    )
    CG = basix.ufl.element(
        "CG",  # family (positional argument)
        "tetrahedron",  # cell (positional argument)
        1,  # order
        shape=(3,)  # degree (positional argument)
    )
    print('Defined the finite element successfully')

    Vc = fem.functionspace(mesh, P)
    # function spaces
    DGSpace = fem.functionspace(mesh, DG)
    CGSpace = fem.functionspace(mesh, CG)

    # Velocity vector Space
    if eleD == 1:
        uSpace = fem.functionspace(mesh, P)
    else:
        uSpace = fem.functionspace(mesh, ufl.VectorElement("Lagrange", mesh.ufl_cell(), eleD - 1))

    # beta
    # usage: xdmf_reader(variable, variable_name, folder)
    beta_ac, beta_cv = fem.Function(DGSpace), fem.Function(DGSpace)
    beta_ac = IO_functions.xdmf_reader(beta_ac, configs.input.beta_ac, configs.input.para_path)
    beta_cv = IO_functions.xdmf_reader(beta_cv, configs.input.beta_cv, configs.input.para_path)

    # pressure
    pa, pc, pv = fem.Function(CGSpace), fem.Function(CGSpace), fem.Function(CGSpace)
    pa = IO_functions.xdmf_reader(pa, configs.input.pa, configs.input.para_path)
    pc = IO_functions.xdmf_reader(pc, configs.input.pc, configs.input.para_path)
    pv = IO_functions.xdmf_reader(pv, configs.input.pv, configs.input.para_path)
    # velocity
    ua, uc = fem.Function(uSpace), fem.Function(uSpace)
    ua = IO_functions.xdmf_reader(ua, configs.input.ua, configs.input.para_path)
    uc = IO_functions.xdmf_reader(uc, configs.input.uc, configs.input.para_path)

    # brain depth expression
    depth = fem.Function(DGSpace)
    # hdf5_reader(mesh, variable, variable_name, folder):
    # file_path = folder + variable_name + '.h5'
    depth = IO_functions.hdf5_reader(mesh, depth, configs.input.depth, "src/Gem_X/core/oxygen/")

    depth_func_CG1 = fem.Function(CGSpace)
    depth_func_CG1.interpolate(depth)
    return Vc, DGSpace, CGSpace, uSpace, beta_ac, beta_cv, pa, pc, pv, ua, uc, depth, depth_func_CG1


# %% Calculation of artifitial diffusion
def art_diff(mesh, ua, D_a, DGSpace, depth, configs):
    # Calculate Peclet number
    uaMag = ufl.sqrt(ufl.dot(ua, ua))
    h = ufl.CellDiameter(mesh)

    # Create functions for PehVal and Pehlim
    PehVal = fem.Function(DGSpace)
    Pehlim = fem.Function(DGSpace)

    # Interpolation for Pehlim based on depth condition
    if configs.simulation.Pehdepth:
        Pehlim.interpolate(lambda x: depth)

    # Compute alpha values
    num_dofs = DGSpace.dofmap.index_map.size_global
    alpha = np.zeros(num_dofs)
    Peh_array = np.zeros(num_dofs)
    Peh_array = PehVal.x.array
    Pehlim_array = Pehlim.x.array

    # Vectorized computation of alpha
    for i in tqdm(range(min(len(alpha), len(Peh_array), len(Pehlim_array))), desc="Computing alpha"):
        if Peh_array[i] != 0:
            alpha[i] = (1 / np.tanh(Peh_array[i]) - 1 / Peh_array[i]) / Pehlim_array[i]

    uaMag = ufl.sqrt(ufl.dot(ua, ua))

    # Calculate dalta
    dalta = alpha * uaMag * h / 2 + D_a
    return dalta


# %% Compute DBC in arteriole compartment

# boundary labels: 0 interior face, 1 brain stem cut plane, 2 ventricular surface, 2+ (two digits) brain surface
def BC(boundaries, Vc, configs):
    BCa = []
    # based on inlet boundary file
    if configs.input.read_inlet_boundary == True:
        BC_data = np.loadtxt(configs.input.pialBC_file, skiprows=1, delimiter=',')
        boundary_labels = BC_data[:, 0] if BC_data.ndim > 1 else [BC_data[0]]

        mesh = Vc.mesh
        mesh.topology.create_entities(2)
        mesh.topology.create_connectivity(2, 3)

        for label in tqdm(boundary_labels, desc="Processing boundary labels"):
            label = int(label)
            # Find suitable `label` facet
            facets = boundaries.find(label)
            if facets.size == 0:
                raise ValueError(f"No facets found for label {label}")
            facets = np.array(facets, dtype=np.int32)
            dofs = fem.locate_dofs_topological(Vc.sub(0), 2, facets)

            # Create Dirichlet boundary conditions
            BCa.append(fem.dirichletbc(configs.simulation.BCa, dofs, Vc.sub(0)))

    return BCa


def O2_Linear(beta12, beta23, mesh, Vc, pa, pc, pv, ua, uc, phiA, phiC, phiT,
              D_a, D_c, D_t, SaVa, ScVc, gammaA, gammaC, tau, M, BCa):
    # Test function and variables
    v_a, v_c, v_t = ufl.TestFunctions(Vc)
    C = ufl.TrialFunction(Vc)
    C_a, C_c, C_t = ufl.split(C)
    beta12_expr = ufl.as_vector([beta12.sub(i) for i in range(3)])
    beta23_expr = ufl.as_vector([beta23.sub(i) for i in range(3)])
    pa_expr = ufl.as_vector([pa.sub(i) for i in range(3)])
    pc_expr = ufl.as_vector([pc.sub(i) for i in range(3)])
    pv_expr = ufl.as_vector([pv.sub(i) for i in range(3)])
    ua_expr = ufl.as_vector([ua.sub(i) for i in range(3)])
    uc_expr = ufl.as_vector([uc.sub(i) for i in range(3)])
    v_a_expr, v_c_expr, v_t_expr = v_a, v_c, v_t

    # Variables expression(weak)
    LHS = (
            ufl.inner(ua_expr, ufl.grad(C_a)) * v_a * ufl.dx
            + 0.01813 * 1.0e-3 * ufl.inner(ufl.grad(C_a), ufl.grad(v_a)) * ufl.dx
            + ufl.inner(beta12_expr, (pa_expr - pc_expr)) * C_a * v_a * ufl.dx
            + SaVa * phiA * gammaA * (tau * C_a - C_t) * v_a * ufl.dx
            + ufl.inner(uc_expr, ufl.grad(C_c)) * v_c * ufl.dx
            + phiC * D_c * ufl.inner(ufl.grad(C_c), ufl.grad(v_c)) * ufl.dx
            - ufl.inner(beta12_expr, (pa_expr - pc_expr)) * C_a * v_c * ufl.dx
            + ufl.inner(beta23_expr, (pc_expr - pv_expr)) * C_c * v_c * ufl.dx
            + ScVc * phiC * gammaC * (tau * C_c - C_t) * v_c * ufl.dx
            + phiT * D_t * ufl.inner(ufl.grad(C_t), ufl.grad(v_t)) * ufl.dx
            - SaVa * phiA * gammaA * (tau * C_a - C_t) * v_t * ufl.dx
            - ScVc * phiC * gammaC * (tau * C_c - C_t) * v_t * ufl.dx
            + phiT * M * C_t * v_t * ufl.dx
    )

    # Right Hand Side
    RHS = fem.Constant(mesh, PETSc.ScalarType(0.0)) * v_a_expr * ufl.dx \
          + fem.Constant(mesh, PETSc.ScalarType(0.0)) * v_c_expr * ufl.dx \
          + fem.Constant(mesh, PETSc.ScalarType(0.0)) * v_t_expr * ufl.dx

    problem = LinearProblem(LHS, RHS, bcs=BCa,
                            petsc_options={"ksp_type": "bcgs"})

    C_sol = problem.solve()

    # Extract each component
    Ca = C_sol.sub(0)
    Cc = C_sol.sub(1)
    Ct = C_sol.sub(2)
    return Ca, Cc, Ct


def O2_nonLinear(beta12, beta23, mesh, ele, pa, pc, pv, ua, uc,
                 phiA, phiC, phiT, D_a, D_c, D_t, SaVa, ScVc,
                 gammaA, gammaC, tau, G, C50, BCa):
    # Create FiniteElements Space
    Vc = fem.FunctionSpace(mesh, ele)
    # Test function and variables
    v_a, v_c, v_t = ufl.TestFunctions(Vc)
    C = fem.Function(Vc)
    C_a, C_c, C_t = ufl.split(C)
    beta12_expr = ufl.as_vector([beta12.sub(i) for i in range(3)])
    beta23_expr = ufl.as_vector([beta23.sub(i) for i in range(3)])
    pa_expr = ufl.as_vector([pa.sub(i) for i in range(3)])
    pc_expr = ufl.as_vector([pc.sub(i) for i in range(3)])
    pv_expr = ufl.as_vector([pv.sub(i) for i in range(3)])
    ua_expr = ufl.as_vector([ua.sub(i) for i in range(3)])
    uc_expr = ufl.as_vector([uc.sub(i) for i in range(3)])
    # Variables expression(weak)
    LHS = (ufl.inner(ua_expr, ufl.grad(C_a)) * v_a * ufl.dx
           + 0.01813 * 1.0e-3 * ufl.inner(ufl.grad(C_a), ufl.grad(v_a)) * ufl.dx
           + ufl.inner(beta12_expr, (pa_expr - pc_expr)) * C_a * v_a * ufl.dx
           + SaVa * phiA * gammaA * (tau * C_a - C_t) * v_a * ufl.dx
           + ufl.inner(uc_expr, ufl.grad(C_c)) * v_c * ufl.dx
           + phiC * D_c * ufl.inner(ufl.grad(C_c), ufl.grad(v_c)) * ufl.dx
           - ufl.inner(beta12_expr, (pa_expr - pc_expr)) * C_a * v_c * ufl.dx
           + ufl.inner(beta23_expr, (pc_expr - pv_expr)) * C_c * v_c * ufl.dx
           + ScVc * phiC * gammaC * (tau * C_c - C_t) * v_c * ufl.dx
           + phiT * D_t * ufl.inner(ufl.grad(C_t), ufl.grad(v_t)) * ufl.dx
           - SaVa * phiA * gammaA * (tau * C_a - C_t) * v_t * ufl.dx
           - ScVc * phiC * gammaC * (tau * C_c - C_t) * v_t * ufl.dx
           + phiT * G * C_t / (C50 - C_t) * v_t * ufl.dx)
    # Calculate Jacobian determinant
    J = ufl.derivative(LHS, C)
    # Create Nonlinear Problem
    problem = fem.petsc.NonlinearProblem(LHS, C, BCa, J)
    # Creat Newton Solver
    solver = nls.petsc.NewtonSolver(problem)
    # Set solver parameters
    solver.krylov_solver.setType("gmres")  # Linear Solver GMRES
    solver.krylov_solver.getPC().setType("ilu")  # Preconditioner ILU
    solver.convergence_criterion = "incremental"
    solver.solve(C)
    Ca = C.sub(0)
    Cc = C.sub(1)
    Ct = C.sub(2)
    return Ca, Cc, Ct

