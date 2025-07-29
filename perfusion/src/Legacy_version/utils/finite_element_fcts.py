from dolfin import *
import numpy as np
import time
from . import suppl_fcts


def mesh_reader(mesh_file):
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
    if data_dimension_greater_than_1:
        BCs.append(DirichletBC(V_space, Constant(BC_data[index, 2]), boundaries, boundary_id))
    else:
        BCs.append(DirichletBC(V_space, Constant(BC_data[2]), boundaries, boundary_id))
    return BCs


def apply_neumann_BC(mesh, b1, integrals_N, index, v_1, dS, boundary_id):
    area = assemble(Constant(1) * dS(boundary_id, domain=mesh))
    b1[index] = b1[index] / area
    # b1 includes the average surface-normal velocity for the perfusion regions
    # computed from the volumetric flow rate [mm^3/s] and the surface area [mm^2]

    if b1[index] > 0:
        integrals_N.append(b1[index] * v_1 * dS(boundary_id))

    return integrals_N


def apply_mixed_BC(mesh, boundaries, integrals_N, BCs, BC_data, V_space, b1, dS, v_1, index,
                   data_dimension_greater_than_1, boundary_id):
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
                    BCs.append(DirichletBC(V.sub(0), Constant(pa), boundaries, boundary_labels[i]))
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