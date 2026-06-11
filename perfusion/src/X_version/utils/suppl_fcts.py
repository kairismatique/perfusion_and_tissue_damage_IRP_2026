import dolfinx
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
import ufl
import basix
import numpy as np
from typing import Optional
from mpi4py import MPI
import warnings
import os


# Local import
from src.X_version.io.IO_functions import print0

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
root = 0

def set_coupling_coeff(beta):
    beta12 = beta[0,1]
    beta13 = beta[0,2]
    beta23 = beta[1,2]
    
    beta21 = beta12
    beta31 = beta13
    beta32 = beta23
    
    return beta12, beta13, beta21, beta23, beta31, beta32


def region_label_assembler(region: Optional[dolfinx.mesh.MeshTags]) -> tuple[np.ndarray, int]:
    """
    Returns a list of unique region tags and how many unique regions there are.

    The 2,164,216 faces from the mesh are split between each process, and the
    attached meshtags are noted. The integer tags are recast. A list of the number
    of faces that each process has managed is gathered in an array on the root
    using comm.Gather. This is summed, and an empty receive array is created of
    the proper size. Then comm.Gatherv receives each process' list of tags and
    concatenates them into a single list on the root. The root calculates the
    list of unique labels and the number of unique labels. These are then
    broadcasted to the processes as a tuple containing a NumPy array and an integer.

    Parameters:
        region: dolfinx.mesh.Meshtags
            The boundary facet meshtags from mesh_reader

    Returns:
        region_labels : np.array.int64
            A NumPy array of unique labels; now identical on all processes.

        n_labels : int
            An integer count of unique labels; now identical on all processes.

    Notes
    -----
    - For this particular clustered.xdmf mesh:
    - 1,042,301 cells
    - 2,164,216 facets (in the whole mesh, with no repeats)
    - 159,228 facets on a boundary

    """
    # Access tags from facets counted by process
    region_labels = region.values

    # Cast to int64
    region_labels = np.array(region_labels, dtype=np.int64)
    # Force into positive int32 range
    region_labels = (region_labels % (2 ** 31)).astype(np.int64)

    # Package data
    sendbuf = region_labels
    # Gathers the length of the locally computed facet tags
    sendcounts = np.array(comm.gather(len(sendbuf), root=root))

    # Creates an empty NumPy array on the root process for storing all region labels
    if rank == root:
        recvbuf = np.empty(sum(sendcounts), dtype=np.int64)
    else:
        recvbuf = None

    # Collects all sendbuf arrays from each process and concatenates onto recvbuf on root
    comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, sendcounts), root=root)

    if rank == root:
        # Calculates the distinct labels
        unique_labels = np.array(list(set(recvbuf)))
        # Calculates the number of distinct labels
        n_labels = np.array([len(unique_labels)], dtype=np.int64)
    else:
        n_labels = np.array([0], dtype=np.int64)

    # Broadcast n_labels to all processes
    comm.Bcast(n_labels, root=root)

    # Change name unique_labels to region_labels
    if rank == root:
        region_labels = unique_labels
    else:
        region_labels = np.zeros(n_labels[0], dtype=np.int64)

    # Broadcast region_labels to all processes
    comm.Bcast(region_labels, root=root)

    # Convert n_labels to scalar
    n_labels = int(n_labels[0])

    # Return tuple (region_labels, n_labels) or e.g. ([1, 2, ...], 2)
    return region_labels, n_labels


def compute_vessel_orientation(subdomains, boundaries, mesh, res_fldr, save_subres):
    """
    Orientation is computed based on a flow field originating from the cortical
    surface and running towards the ventricles.

    Args:
        subdomains (dolfinx.mesh.MeshTags or None): MeshTags for cell regions (not directly used in this function, but passed for consistency).
        boundaries (dolfinx.mesh.MeshTags or None): MeshTags for facet regions (boundary markers).
        mesh_obj (dolfinx.mesh.Mesh): The Dolfinx mesh object.
        res_fldr (str): Folder to save results.
        save_subres (bool): Whether to save intermediate results (pe field).

    Returns:
        tuple: A tuple containing:
            - e (dolfinx.fem.Function): Normalised penetrating vessel axis direction.
            - main_direction (dolfinx.fem.Function): Main direction of penetrating vessel axes.
    """
    fe_degr = 2

    # Scalar function space for the thickness values.
    Vpe = fem.functionspace(mesh, ("Lagrange", fe_degr))

    pe_in = 1.0  # Pial surface
    pe_out = 0.0  # Ventricular surface

    # NOTATION:
    # 0: Interior surfaces, 1: Brainstem-cut plane,
    # 2: Ventricular surface, 2+: Brain surface (pial)

    # Ventricular surface
    facet_indices = np.array(boundaries.indices[np.where(boundaries.values == 2)], dtype=np.int32)
    boundary_dofs = fem.locate_dofs_topological(Vpe, entity_dim=2, entities=facet_indices)
    BCs = [fem.dirichletbc(pe_out, boundary_dofs, Vpe)]

    # Pial surface
    boundary_labels, n_labels = region_label_assembler(boundaries)

    for i in range(n_labels):
        if boundary_labels[i] > 2:
            facet_indices = np.array(boundaries.indices[np.where(boundaries.values == boundary_labels[i])],
                                     dtype=np.int32)
            boundary_dofs = fem.locate_dofs_topological(Vpe, entity_dim=2, entities=facet_indices)
            BCs.append(fem.dirichletbc(pe_in, boundary_dofs, Vpe))

    # Solve for scalar thickness field
    pe_trial = ufl.TrialFunction(Vpe)
    ve_test = ufl.TestFunction(Vpe)
    f = fem.Constant(mesh, 0.0)
    LHS = ufl.inner(ufl.grad(pe_trial), ufl.grad(ve_test)) * ufl.dx  # Bilinear form
    RHS = f * ve_test * ufl.dx  # Linear form
    # Initialise the function pe in the predefined function space
    pe = fem.Function(Vpe)
    problem = LinearProblem(LHS, RHS, bcs=BCs, petsc_options={"ksp_type": "bcgs", "pc_type": "hypre"}, petsc_options_prefix="pe_")
    pe = problem.solve()
    # Lower the order to save or visualise the solution
    V_write = fem.functionspace(mesh, ("Lagrange", 1))
    pe_interpolated = fem.Function(V_write)
    pe_interpolated.interpolate(pe)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, res_fldr + 'pe.xdmf', "w") as myfile:
        myfile.write_mesh(mesh)  # NOTE: needed?
        myfile.write_function(pe_interpolated)

    # Define function spaces based on fe_degr
    if fe_degr == 1:
        # Vector element for DG degree 0
        # A function space for piecewise constant vectors over tetrahedra
        element_vec = basix.ufl.element("DG", "tetrahedron", 0, shape=(3,))
        Ve = fem.functionspace(mesh, element_vec)
    else:
        # Vector element for Lagrange and DG degree (fe_degr-1)
        element_vec_lagrange = basix.ufl.element("Lagrange", "tetrahedron", fe_degr - 1, shape=(3,))
        Ve = fem.functionspace(mesh, element_vec_lagrange)

        # DG degree 0 element for Ve_DG
        element_vec_DG = basix.ufl.element("DG", "tetrahedron", 0, shape=(3,))
        Ve_DG = fem.functionspace(mesh, element_vec_DG)

    # Solve finite element problem for projection
    u_proj = ufl.TrialFunction(Ve)
    v_proj = ufl.TestFunction(Ve)
    f_expr = -ufl.grad(pe)
    a_proj = ufl.inner(u_proj, v_proj) * ufl.dx(mesh)
    L_proj = ufl.inner(f_expr, v_proj) * ufl.dx(mesh)
    problem_proj = LinearProblem(a_proj, L_proj, bcs=[], petsc_options={"ksp_type": "bcgs"}, petsc_options_prefix="proj_")
    E = problem_proj.solve()

    # Solve finite element problem for interpolation
    if fe_degr > 1:
        # Cell-averaging evaluation
        # Negligible difference with Legacy after normalisation
        u_dg = ufl.TrialFunction(Ve_DG)
        v_dg = ufl.TestFunction(Ve_DG)
        a_proj_dg = ufl.inner(u_dg, v_dg) * ufl.dx(mesh)
        L_proj_dg = ufl.inner(E, v_dg) * ufl.dx(mesh)
        problem_proj_dg = LinearProblem(a_proj_dg, L_proj_dg, bcs=[], petsc_options={"ksp_type": "bcgs"}, petsc_options_prefix="proj_dg_")
        e = problem_proj_dg.solve()
    else:
        e = E

    # Normalise e
    e_array = e.x.array
    for i in range(int(len(e_array) / 3)):
        norm_val = np.linalg.norm(e_array[i * 3:(i + 1) * 3])
        if norm_val > 0:
            e_array[i * 3:(i + 1) * 3] /= norm_val
    e.x.array[:] = e_array

    # Define Vdir (using a DG0 space for scalar values)
    Vdir = fem.functionspace(mesh, ("DG", 0))
    # Compute main direction of the vessels
    print0('step 2.3.1: COMPUTING MAIN DIRECTION')
    main_direction = fem.Function(Vdir)
    main_direction_array = main_direction.x.array
    for i in range(int(len(e_array) / 3)):
        indices = np.where(abs(e_array[i * 3:(i + 1) * 3]) >= np.sqrt(1 / 3))[0]
        if indices.size > 0:
            main_direction_array[i] = indices[0]
        else:
            main_direction_array[i] = -1

    main_direction.x.array[:] = main_direction_array

    return e, main_direction


def comp_transf_mat(e0, e1):
    v = np.cross(e0, e1)
    s = np.linalg.norm(v)  # sine
    c = np.dot(e0, e1)  # cosine
    I = np.identity(3)  # identity matrix
    u = v / s
    ux = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])

    T = c * I + s * ux + (1 - c) * np.tensordot(u, u, axes=0)

    return T


def permeability_tensor_computation(
        K_space: Optional[dolfinx.fem.function.FunctionSpace],
        subdomains: Optional[dolfinx.mesh.MeshTags],
        mesh: Optional[dolfinx.mesh.Mesh],
        e_ref,
        e_loc,
        K1_form
):
    """
    Computes a function which gives the rotated permeability tensor
    at every point in the brain, as it follows the local direction of
    blood vessels.

    Parameters
    ----------
    K_space : dolfinx.fem.function.FunctionSpace
        The function space defined for the permeabilities.
    subdomains : dolfinx.mesh.MeshTags
        Subdomains from the mesh.
    mesh : dolfinx.mesh.Mesh
        Information about the mesh.
    e_ref : type
        Direction vector representing vessel orientation at the reference point.
    e_loc : type
        The local direction vector.
    K1_form : type
        The normalised form of the permeability tensor at the reference point,
        to be rotated and scaled.

    Returns
    -------
    type
        Description of the return value.
    """
    # Get the mesh's topological dimension
    dim   = mesh.topology.dim
    # Get an index map of facets
    imap  = mesh.topology.index_map(dim)
    # Get the local ownership rank
    start, end = imap.local_range

    # Extract array from the function
    e_loc_arr = e_loc.x.array

    # Initialise a new function in K_space
    K1 = fem.Function(K_space)
    K1_arr = np.zeros(K1.x.array.shape, dtype=K1.x.array.dtype)
    print(f"[Rank {rank}] Initialized K1_arr dtype={K1_arr.dtype}, shape={K1_arr.shape}")

    # Loop over owned cells only (local indices)
    for local_cell in range(start, end):
        off = local_cell - start
        # Extract local direction vector segment
        segment_start = off*3
        segment_end = (off+1)*3
        e1 = e_loc_arr[segment_start:segment_end]
        # Debug: e1 shape
        if e1.shape != (3,):
            print(f"[Rank {rank}] WARNING: e1 slice shape mismatch at local_cell={local_cell}, got {e1.shape}, expected (3,)")

        # Compute transformation matrix
        T = comp_transf_mat(e_ref, e1)
        # Debug: T shape and finite check
        if T.shape != (3,3):
            print(f"[Rank {rank}] WARNING: T shape mismatch at local_cell={local_cell}, got {T.shape}, expected (3,3)")

        # Rotate permeability tensor
        K1_rot = T @ K1_form @ T.T
        # Debug: rotated tensor shape and values
        if K1_rot.shape != (3,3):
            print(f"[Rank {rank}] WARNING: K1_rot shape mismatch at local_cell={local_cell}, got {K1_rot.shape}, expected (3,3)")
        if not np.all(np.isfinite(K1_rot)):
            print(f"[Rank {rank}] WARNING: Non-finite values in K1_rot at local_cell={local_cell}")

        flat = K1_rot.reshape(9)
        # Debug: flat shape
        if flat.shape != (9,):
            print(f"[Rank {rank}] WARNING: flat reshape mismatch at local_cell={local_cell}, got {flat.shape}, expected (9,)")
        # Assign to output vector
        start_idx = off*9
        end_idx = (off+1)*9
        K1_arr[start_idx:end_idx] = flat

    # Zero‐tolerance cleanup
    tol = 1e-9
    mask = np.abs(K1_arr) < tol
    cleanup_count = np.count_nonzero(mask)
    K1_arr[mask] = 0.0

    # Assign to Function and sync ghost values
    K1.x.array[:] = K1_arr
    K1.x.scatter_forward()

    return K1

