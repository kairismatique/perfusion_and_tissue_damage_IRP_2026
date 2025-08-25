from dolfin import *
import numpy as np
import untangle


def mesh_reader(mesh_file):
    """
    Loads a volumetric mesh and associated subdomain and boundary information from XDMF files.

    This function reads a mesh, its physical region tags (volume subdomains), and its facet tags
    (boundary surfaces) from separate XDMF files. The mesh is assumed to be stored in `mesh_file`,
    while the associated subdomain and facet files are derived from the base name of `mesh_file`.

    Args:
        mesh_file (str): Path to the main volumetric mesh file (XDMF format).

    Returns:
        tuple: A tuple containing:
            - mesh (dolfin.Mesh): The loaded mesh object.
            - subdomains (dolfin.MeshFunction): Subdomain markers (dimension 3).
            - boundaries (dolfin.MeshFunction): Boundary markers (dimension 2).
    """
    comm = MPI.comm_world

    mesh = Mesh()
    with XDMFFile(comm, mesh_file) as file: file.read(mesh)
    subdomains = MeshFunction("size_t", mesh, 3)
    with XDMFFile(comm, mesh_file[:-5] + '_physical_region.xdmf') as file: file.read(subdomains)
    boundaries = MeshFunction("size_t", mesh, 2)
    with XDMFFile(comm, mesh_file[:-5] + '_facet_region.xdmf') as file: file.read(boundaries)

    return mesh, subdomains, boundaries


def input_file_reader(input_file_path):
    """
    Parses an XML configuration file and extracts input parameters for poroelastic brain simulation.

    This function uses `untangle` to read and parse a structured XML file containing simulation
    inputs such as mesh paths, boundary conditions, reference permeabilities, and model constants.

    Args:
        input_file_path (str): Path to the XML input file (typically ending in `.xml`).

    Returns:
        tuple: A tuple containing:
            - mesh_file (str): Path to the volumetric mesh file.
            - p_arterial (float): Arterial pressure in Pascals.
            - p_venous (float): Venous pressure in Pascals.
            - e_ref (numpy.ndarray): Reference extracellular porosity (3-element array).
            - K1_ref (numpy.ndarray): Reference permeability tensor for compartment 1 (3x3).
            - K2_ref (numpy.ndarray): Reference permeability tensor for compartment 2 (3x3).
            - K3_ref (numpy.ndarray): Reference permeability tensor for compartment 3 (3x3).
            - beta (numpy.ndarray): Coupling coefficient matrix between compartments (3x3).
            - fe_degr (int): Finite element polynomial degree.
            - res_fldr (str): Path to folder where results will be stored.
            - pial_surf_file (str): Path to surface mesh file of the pial region.
            - inflow_file (str): Path to file with volumetric inflow boundary conditions.
    """
    # Get configs
    configs = untangle.parse(input_file_path).porobrain

    # Get files
    mesh_file = configs.files_and_folders.mesh_file.cdata.strip()
    pial_surf_file = configs.files_and_folders.pial_surf_file.cdata.strip()
    inflow_file = configs.files_and_folders.inflow_file.cdata.strip()
    res_fldr = configs.files_and_folders.res_fldr.cdata.strip()

    # Get physical parameters
    p_arterial = float(configs.physical_variables.p_arterial.cdata)
    p_venous = float(configs.physical_variables.p_venous.cdata)
    e_ref = np.array(list(map(float, configs.physical_variables.e_ref.cdata.split(','))))
    K1_ref = np.array(list(map(float, configs.physical_variables.K1_ref.cdata.split(',')))).reshape((3, 3))
    K2_ref = np.array(list(map(float, configs.physical_variables.K2_ref.cdata.split(',')))).reshape((3, 3))
    K3_ref = np.array(list(map(float, configs.physical_variables.K3_ref.cdata.split(',')))).reshape((3, 3))
    beta = np.array(list(map(float, configs.physical_variables.beta.cdata.split(',')))).reshape((3, 3))

    # Get simulation parameters
    fe_degr = int(configs.simulation_settings.fe_degr.cdata)

    return mesh_file, p_arterial, p_venous, e_ref, \
        K1_ref, K2_ref, K3_ref, beta, fe_degr, res_fldr, pial_surf_file, inflow_file


def inlet_file_reader(inlet_boundary_file):
    """
    Reads inlet boundary condition data from a text file.

    The file is expected to contain tabular data with columns: boundary ID, flow rate [ml/s],
    and pressure [Pa]. The flow rates are converted from ml/s to mm³/s (i.e., multiplied by 1000).

    Args:
        inlet_boundary_file (str): Path to the file containing inlet boundary condition data.

    Returns:
        numpy.ndarray: A NumPy array with shape (N, 3) where each row represents
        [boundary ID, flow rate (mm³/s), pressure (Pa)].
    """
    # ID, Q [ml/s], p [Pa]
    boundary_data = np.loadtxt(inlet_boundary_file, skiprows=1)
    boundary_data[:, 1] = 1000 * boundary_data[:, 1]
    return boundary_data


def pvd_saver(variable, folder, name):
    """
    Saves a FEniCS Function variable to a .pvd file for visualization.

    This function renames the variable and saves it using the XML-based PVD format,
    which is compatible with ParaView.

    Args:
        variable (dolfin.Function): The function to be saved.
        folder (str): Directory where the .pvd file will be saved.
        name (str): Base name for the saved file (without extension).
    """
    variable.rename(name, "1")
    vtkfile = File(folder + name + '.pvd')
    vtkfile << variable


def hdf5_saver(mesh, variable, folder, file_name, variable_name):
    """
    Saves a FEniCS Function to an HDF5 file.

    This function writes a given variable to an HDF5 file for efficient storage and retrieval
    in distributed or large-scale simulations.

    Args:
        mesh (dolfin.Mesh): The mesh associated with the variable.
        variable (dolfin.Function): The function to be saved.
        folder (str): Directory where the file will be saved.
        file_name (str): Name of the output HDF5 file.
        variable_name (str): Name of the variable inside the HDF5 structure.
    """
    hdf = HDF5File(mesh.mpi_comm(), folder + file_name, "w")
    hdf.write(variable, "/" + variable_name)
    hdf.close()


def hdf5_reader(mesh, variable, folder, file_name, variable_name):
    """
    Loads a FEniCS Function from an HDF5 file.

    This function reads a stored variable from an HDF5 file and loads it into the provided
    FEniCS Function object, which must be pre-allocated.

    Args:
        mesh (dolfin.Mesh): The mesh associated with the variable.
        variable (dolfin.Function): The pre-defined function where data will be loaded.
        folder (str): Directory where the HDF5 file is located.
        file_name (str): Name of the HDF5 file to read from.
        variable_name (str): Name of the variable in the HDF5 structure.
    """
    hdf = HDF5File(mesh.mpi_comm(), folder + file_name, "r")
    hdf.read(variable, "/" + variable_name)
    hdf.close()


def xdmf_reader(file, function, checkpoint_function):
    """
    Load a FEniCS Function from an XDMF file checkpoint.

    Reads a previously stored function from disk and returns it in the given function space.

    Args:
        file (str): Path to the XDMF file containing the checkpoint.
        function (FunctionSpace): FEniCS function space into which the data will be loaded.
        checkpoint_function (str): Name of the checkpoint variable stored in the file.

    Returns:
        Function: The loaded FEniCS function containing the stored simulation data.
    """
    variable = Function(function)
    f_in = XDMFFile(file)
    f_in.read_checkpoint(variable, checkpoint_function, 0)
    f_in.close()
    return variable
