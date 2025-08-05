from dolfin import *
import numpy as np
import time
from . import suppl_fcts


def mesh_reader(mesh_file):
    """
    Reads mesh, subdomain, and boundary data from an HDF5 file.

    Args:
        mesh_file (str): The path to the HDF5 file containing the mesh data.

    Returns:
        tuple: A tuple containing:
            - mesh (dolfin.cpp.mesh.Mesh): The mesh object.
            - subdomains (dolfin.cpp.mesh.MeshFunction): MeshFunction storing subdomain markers (dimension 3).
            - boundaries (dolfin.cpp.mesh.MeshFunction): MeshFunction storing boundary markers (dimension 2).
    """
    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), mesh_file, "r")
    hdf.read(mesh, "/mesh", False)
    subdomains = MeshFunction("size_t", mesh, 3)
    hdf.read(subdomains, "/subdomains")
    boundaries = MeshFunction("size_t", mesh, 2)
    hdf.read(boundaries, "/boundaries")
    hdf.close()
    return mesh, subdomains, boundaries


def allocation_functions_space(mesh, fe_degr, **kwarg):
    """
    Allocates finite element function spaces for pressure and velocity based on the specified model type.

    Args:
        mesh (dolfin.cpp.mesh.Mesh): The mesh object on which to define the function spaces.
        fe_degr (int): The polynomial degree for the pressure function space.
        **kwarg:
            model_type (str, optional): The type of model to use ('acv' for arterial-capillary-venous,
                                        'a' for arterial only). Defaults to 'acv'.
            vel_order (int, optional): The polynomial degree for the velocity function space.
                                       Defaults to fe_degr - 1.

    Returns:
        tuple: A tuple containing:
            - Vp (dolfin.fem.functionspace.FunctionSpace): The function space for pressure(s).
            - Vvel (dolfin.fem.functionspace.FunctionSpace): The function space for velocity vectors.
            - v_1 (dolfin.fem.form.TestFunction): Test function for the first pressure component.
            - v_2 (dolfin.fem.form.TestFunction or list): Test function for the second pressure component (or empty list if not applicable).
            - v_3 (dolfin.fem.form.TestFunction or list): Test function for the third pressure component (or empty list if not applicable).
            - p (dolfin.fem.form.TrialFunction): Trial function for pressure(s).
            - p_1 (dolfin.fem.form.Argument or list): Split component of the trial function for the first pressure.
            - p_2 (dolfin.fem.form.Argument or list): Split component of the trial function for the second pressure (or empty list if not applicable).
            - p_3 (dolfin.fem.form.Argument or list): Split component of the trial function for the third pressure (or empty list if not applicable).
            - K1_space (dolfin.fem.functionspace.TensorFunctionSpace): Function space for the first permeability tensor.
            - K2_space (dolfin.fem.functionspace.FunctionSpace): Function space for the second permeability (scalar).

    Raises:
        Exception: If an unknown `model_type` is provided.
    """
    # Retrieve parameters
    model_type = kwarg.get('model_type', 'acv')
    vel_order = kwarg.get('vel_order', fe_degr - 1)
    
    if model_type == 'acv':
        # Element type and degree
        P = FiniteElement('Lagrange', tetrahedron, fe_degr)
        # Define mixed element (3D vector)
        element = MixedElement([P, P, P])
        # Define continous function space for pressure
        Vp = FunctionSpace(mesh, element)
        
        # Define test functions
        v_1, v_2, v_3 = TestFunctions(Vp)
        
        # Define trial function for pressures
        p = TrialFunction(Vp)
        # Split pressure function to access components
        p_1, p_2, p_3 = split(p)

    elif model_type == 'a':
        # Define continous function space for pressure
        Vp = FunctionSpace(mesh, "Lagrange", fe_degr)
        
        # Define test functions
        v_1, v_2, v_3 = TestFunction(Vp), [], []
        
        # Define trial function for pressures
        p = TrialFunction(Vp)
        # Split pressure function to access components
        p_1, p_2, p_3 = [], [], []

    else:
        raise Exception("Unknown model type: " + model_type)

    # Define discontinuous function space for permeability tensors
    K1_space = TensorFunctionSpace(mesh, "DG", 0)
    K2_space = FunctionSpace(mesh, "DG", 0)

    # Define function space for velocity vectors
    if vel_order == 0:
        Vvel = VectorFunctionSpace(mesh, "DG", vel_order)
    else:
        Vvel = VectorFunctionSpace(mesh, "Lagrange", vel_order)
    
    return Vp, Vvel, v_1, v_2, v_3, p, p_1, p_2, p_3, K1_space, K2_space


def read_boundary_conditions(file_path):
    """
    Reads boundary condition data from a CSV file.

    The CSV file is expected to have a header row, and data separated by commas.
    The function handles both single-row and multi-row data.

    Args:
        file_path (str): The path to the CSV file containing boundary condition data.

    Returns:
        tuple: A tuple containing:
            - data (numpy.ndarray): The raw NumPy array loaded from the CSV file.
            - b1 (list): A list of scaled flow rates (1000 * value from the second column).
            - labels (list): A list of boundary labels (from the first column).
            - data_dimension_greater_than_1 (bool): True if the loaded data has more than one dimension (multiple rows),
                                                    False otherwise.
    """
    data = np.loadtxt(file_path, skiprows=1, delimiter=',')
    data_dimension_greater_than_1 = data.ndim > 1
    if data_dimension_greater_than_1:
        b1 = 1000 * data[:, 1]
        labels = list(data[:, 0])
    else:
        b1 = [1000 * data[1]]
        labels = [data[0]]
    return data, b1, labels, data_dimension_greater_than_1


def apply_dirichlet_BC(V_space, BC_data, BCs, boundaries, data_dimension_greater_than_1, index, boundary_id):
    """
    Applies a Dirichlet boundary condition to a given function space.

    Args:
        V_space (dolfin.fem.functionspace.FunctionSpace or dolfin.fem.functionspace.SubSpace):
            The function space or subspace to which the DirichletBC will be applied.
        BC_data (numpy.ndarray): The NumPy array containing the boundary condition data.
        BCs (list): A list of existing DirichletBC objects to which the new BC will be appended.
        boundaries (dolfin.cpp.mesh.MeshFunction): MeshFunction storing boundary markers.
        data_dimension_greater_than_1 (bool): True if `BC_data` is multi-dimensional.
        index (int): The row index in `BC_data` to extract the boundary value if `BC_data` is multi-dimensional.
        boundary_id (int): The integer ID of the boundary where the condition is applied.

    Returns:
        list: The updated list of DirichletBC objects.
    """
    if data_dimension_greater_than_1:
        BCs.append(DirichletBC(V_space, Constant(BC_data[index, 2]), boundaries, boundary_id))
    else:
        BCs.append(DirichletBC(V_space, Constant(BC_data[2]), boundaries, boundary_id))
    return BCs


def apply_neumann_BC(mesh, b1, integrals_N, index, v_1, dS, boundary_id):
    """
    Applies a Neumann boundary condition (flux) and computes the corresponding integral term.

    The flow rate `b1[index]` is normalized by the surface area of the boundary.

    Args:
        mesh (dolfin.cpp.mesh.Mesh): The mesh object.
        b1 (list): A list of flow rates. The value at `b1[index]` will be modified in-place by division with area.
        integrals_N (list): A list of integral terms for Neumann boundary conditions, to which the new term will be appended.
        index (int): The index in the `b1` list corresponding to the current boundary.
        v_1 (dolfin.fem.form.TestFunction): The test function for the first pressure component.
        dS (dolfin.fem.form.Measure): The measure for surface integrals (e.g., `ds(subdomain_data=boundaries)`).
        boundary_id (int): The integer ID of the boundary where the condition is applied.

    Returns:
        list: The updated list of integral terms for Neumann boundary conditions.
    """
    area = assemble(Constant(1) * dS(boundary_id, domain=mesh))
    b1[index] = b1[index] / area
    # b1 includes the average surface-normal velocity for the perfusion regions
    # computed from the volumetric flow rate [mm^3/s] and the surface area [mm^2]

    if b1[index] > 0:
        integrals_N.append(b1[index] * v_1 * dS(boundary_id))

    return integrals_N


def apply_mixed_BC(mesh, boundaries, integrals_N, BCs, BC_data, V_space, b1, dS, v_1, index,
                   data_dimension_greater_than_1, boundary_id):
    """
    Applies either a Dirichlet or Neumann boundary condition based on the `BC_data` for a mixed boundary type.

    Args:
        mesh (dolfin.cpp.mesh.Mesh): The mesh object.
        boundaries (dolfin.cpp.mesh.MeshFunction): MeshFunction storing boundary markers.
        integrals_N (list): A list of integral terms for Neumann boundary conditions.
        BCs (list): A list of DirichletBC objects.
        BC_data (numpy.ndarray): The NumPy array containing the boundary condition data.
                                 Expected to have a fifth column (index 4) indicating BC type (0 for Dirichlet, 1 for Neumann).
        V_space (dolfin.fem.functionspace.FunctionSpace or dolfin.fem.functionspace.SubSpace):
            The function space or subspace to which the DirichletBC will be applied.
        b1 (list): A list of flow rates. The value at `b1[index]` might be modified for Neumann BCs.
        dS (dolfin.fem.form.Measure): The measure for surface integrals.
        v_1 (dolfin.fem.form.TestFunction): The test function for the first pressure component.
        index (int): The row index in `BC_data` for the current boundary.
        data_dimension_greater_than_1 (bool): True if `BC_data` is multi-dimensional.
        boundary_id (int): The integer ID of the boundary where the condition is applied.

    Returns:
        tuple: A tuple containing:
            - integrals_N (list): The updated list of integral terms.
            - BCs (list): The updated list of DirichletBC objects.

    Raises:
        Exception: If the boundary condition type in `BC_data` is neither 0 (Dirichlet) nor 1 (Neumann).
    """
    # Dirichlet (pressure) boundary condition
    if BC_data[index, 4] == 0:
        BCs = apply_dirichlet_BC(V_space, BC_data, BCs, boundaries, data_dimension_greater_than_1, index, boundary_id)

    # Neumann (pressure gradient ~ flux) boundary condition
    elif BC_data[index, 4] == 1:
        integrals_N = apply_neumann_BC(mesh, b1, integrals_N, index, v_1, dS, boundary_id)

    else:
        raise Exception("Boundary condition in the BCs.csv file must be 0 (Dirichlet) or 1 (Neumann)")
    return integrals_N, BCs


def set_up_fe_solver(mesh, boundaries, V, v_1, v_2, v_3, \
                         p, p_1, p_2, p_3, K1, K2, K3, beta12, beta23, \
                         pa, pv, read_inlet_boundary, inlet_boundary_file, inlet_BC_type, **kwarg):
    """
    Sets up the finite element variational problem (LHS and RHS) and applies boundary conditions
    for the perfusion solver, based on the specified model type and boundary conditions.

    Args:
        mesh (dolfin.cpp.mesh.Mesh): The mesh object.
        boundaries (dolfin.cpp.mesh.MeshFunction): MeshFunction storing boundary markers.
        V (dolfin.fem.functionspace.FunctionSpace): The main function space for pressure (mixed for 'acv', scalar for 'a').
        v_1 (dolfin.fem.form.TestFunction): Test function for the first pressure component.
        v_2 (dolfin.fem.form.TestFunction or list): Test function for the second pressure component (or empty list).
        v_3 (dolfin.fem.form.TestFunction or list): Test function for the third pressure component (or empty list).
        p (dolfin.fem.form.TrialFunction): Trial function for pressure(s).
        p_1 (dolfin.fem.form.Argument or list): Split component of the trial function for the first pressure.
        p_2 (dolfin.fem.form.Argument or list): Split component of the trial function for the second pressure (or empty list).
        p_3 (dolfin.fem.form.Argument or list): Split component of the trial function for the third pressure (or empty list).
        K1 (dolfin.fem.function.Constant or dolfin.fem.function.Function): Permeability tensor/scalar for the first compartment.
        K2 (dolfin.fem.function.Constant or dolfin.fem.function.Function): Permeability tensor/scalar for the second compartment.
        K3 (dolfin.fem.function.Constant or dolfin.fem.function.Function): Permeability tensor/scalar for the third compartment.
        beta12 (dolfin.fem.function.Constant): Exchange coefficient between compartment 1 and 2.
        beta23 (dolfin.fem.function.Constant): Exchange coefficient between compartment 2 and 3.
        pa (float): Constant value for arterial pressure (Dirichlet boundary condition).
        pv (float): Constant value for venous pressure (Dirichlet boundary condition).
        read_inlet_boundary (bool): If True, boundary conditions are read from `inlet_boundary_file`.
        inlet_boundary_file (str): Path to the CSV file containing inlet boundary conditions (if `read_inlet_boundary` is True).
        inlet_BC_type (str): Type of inlet boundary condition ('DBC' for Dirichlet, 'NBC' for Neumann, 'mixed' for mixed).
        **kwarg:
            model_type (str, optional): The type of model to use ('acv' or 'a'). Defaults to 'acv'.

    Returns:
        tuple: A tuple containing:
            - LHS (dolfin.fem.form.Form): The left-hand side of the variational problem.
            - RHS (dolfin.fem.form.Form): The right-hand side of the variational problem.
            - sigma1 (dolfin.fem.function.Constant): Source term for the first compartment (typically 0.0).
            - sigma2 (dolfin.fem.function.Constant): Source term for the second compartment (typically 0.0).
            - sigma3 (dolfin.fem.function.Constant): Source term for the third compartment (typically 0.0).
            - BCs (list): A list of dolfin.fem.dirichletbc.DirichletBC objects applied to the problem.

    Raises:
        Exception: If `inlet_BC_type` is unknown or `model_type` is unknown.
    """
    model_type = kwarg.get('model_type', 'acv')

    # Source terms are equal to zero
    sigma1 = Constant(0.0)
    sigma2 = Constant(0.0)
    sigma3 = Constant(0.0)

    BCs = []
    integrals_N = []
    V_space = V.sub(0) if model_type == 'acv' else V
    # Based on inlet boundary file
    if read_inlet_boundary:
        BC_data, b1, boundary_labels, data_dimension_greater_than_1 = read_boundary_conditions(inlet_boundary_file)
        n_labels = len(boundary_labels)

        if model_type == 'acv':
            for i in range(n_labels):
                BCs.append(DirichletBC(V.sub(2), Constant(pv), boundaries, int(boundary_labels[i])))

        dS = ds(subdomain_data=boundaries)
        if inlet_BC_type == 'DBC': # Dirichlet BC
            for i in range(n_labels):
                boundary_id = int(boundary_labels[i])
                BCs = apply_dirichlet_BC(V_space, BC_data, BCs, boundaries, data_dimension_greater_than_1, i, boundary_id)

        elif inlet_BC_type == 'NBC': # Neumann BC
            for i in range(n_labels):
                boundary_id = int(boundary_labels[i])
                integrals_N = apply_neumann_BC(mesh, b1, integrals_N, i, v_1, dS, boundary_id)

        elif inlet_BC_type == 'mixed': # Mixed BC
            for i in range(n_labels):
                boundary_id = int(boundary_labels[i])
                integrals_N, BCs = apply_mixed_BC(mesh, boundaries, integrals_N, BCs, BC_data, V_space, b1, dS, v_1, i,
                   data_dimension_greater_than_1, boundary_id)

        else:
            raise Exception("inlet_BC_type must be Neumann, Dirichlet or mixed ('NBC', 'DBC' or 'mixed)")

    else:
        boundary_labels, n_labels = suppl_fcts.region_label_assembler(boundaries)
        for i in range(n_labels):
            if boundary_labels[i] > 2:
                # Brain surface boundary
                if inlet_BC_type == 'DBC':
                    BCs.append(DirichletBC(V_space, Constant(pa), boundaries, boundary_labels[i]))
                if model_type == "acv":
                    BCs.append(DirichletBC(V.sub(2), Constant(pv), boundaries, boundary_labels[i]))

    if model_type == 'acv':
        # Define variational problem
        LHS = \
        inner(K1 * grad(p_1), grad(v_1)) * dx + beta12 * (p_1 - p_2) * v_1 * dx \
        + inner(K2 * grad(p_2), grad(v_2)) * dx + beta12 * (p_2 - p_1) * v_2 * dx + beta23 * (p_2-p_3) * v_2 * dx \
        + inner(K3 * grad(p_3), grad(v_3)) * dx + beta23 * (p_3 - p_2) * v_3 * dx
        RHS = sigma1 * v_1 * dx + sigma2 * v_2 * dx + sigma3 * v_3 * dx + sum(integrals_N)

    elif model_type == 'a':
        # Set constant venous pressure
        p_venous =  Constant(pv)

        # Define variational problem
        beta_total = 1 / (1/beta12+1/beta23)
        LHS = \
        inner(K1 * grad(p), grad(v_1)) * dx + beta_total * p * v_1 * dx
        RHS = sigma1 * v_1 * dx + sum(integrals_N) + beta_total * p_venous * v_1 * dx

    else:
        raise Exception("unknown model type: " + model_type)

    return LHS, RHS, sigma1, sigma2, sigma3, BCs


def solve_lin_sys(Vp, LHS, RHS, BCs, lin_solver, precond, rtol, mon_conv, init_sol, **kwarg):
    """
    Solves the linear system defined by the variational problem using a specified solver and preconditioner.

    Args:
        Vp (dolfin.fem.functionspace.FunctionSpace): The function space for pressure.
        LHS (dolfin.fem.form.Form): The left-hand side of the variational problem.
        RHS (dolfin.fem.form.Form): The right-hand side of the variational problem.
        BCs (list): A list of dolfin.fem.dirichletbc.DirichletBC objects applied to the problem.
        lin_solver (str): The type of linear solver to use (e.g., 'gmres', 'lu').
        precond (str or bool): The type of preconditioner to use (e.g., 'ilu', 'jacobi'), or False if no preconditioner.
        rtol (float or bool): The relative tolerance for the Krylov solver, or False if default is used.
        mon_conv (bool): If True, monitor the convergence of the Krylov solver.
        init_sol (bool): If True, use a non-zero initial guess for the solver.
        **kwarg:
            timer (bool, optional): If True, prints the computation time for solving the system. Defaults to True if not specified.

    Returns:
        dolfin.fem.function.Function: The solved pressure field as a DOLFIN Function object.
    """
    comm = MPI.comm_world
    rank = comm.Get_rank()
    
    # Define functions for pressures
    p = Function(Vp)
    
    # Define weak form
    problem = LinearVariationalProblem(LHS, RHS, p, BCs)
        
    # TODO: set up initialisation with first order:
    # https://fenicsproject.org/qa/1124/is-there-a-way-to-set-the-inital-guess-in-the-krylov-solver/?show=1124#q1124
    
    # Solver settings
    solver = LinearVariationalSolver(problem)
    parameters = solver.parameters
    parameters['linear_solver'] = lin_solver
    if precond != False:
        parameters['preconditioner'] = precond
    if rtol != False:
        PETScOptions.set('ksp_rtol', str(rtol))
        parameters['krylov_solver']['relative_tolerance']=rtol
    parameters['krylov_solver']["monitor_convergence"] = mon_conv
    parameters['krylov_solver']["nonzero_initial_guess"] = init_sol
    
    # Solve equation system
    start = time.time()
    solver.solve()
    end = time.time()
    if rank == 0:
        if 'timer' in kwarg:
            if kwarg.get('timer')==True:
                print ('\t\t pressure computation took', end - start, '[s]')
        else:
            print ('\t\t pressure computation took', end - start, '[s]')
    
    # Mesh refinement syntax
    # new_mesh = refine(mesh)
    # File("new_mesh.pvd") << new_mesh
    return p