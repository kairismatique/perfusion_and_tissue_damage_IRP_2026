"""
This script generates an anisotropic permeability field based on a predefined
form of the permeability tensor given in a reference coordinate system.

K1_form is the forms of the permeability tensor defined in a reference
coordinate system, in which e_ref [0,0,1] unit vector is the coordinate
direction perpendicular to the cortical surface.

Then K1_loc is computed as
K1_loc = T*K1_form*T'

T is the transformation matrix handling rotation based on e_ref and e_loc,
where e_loc is the unit vector showing the direction normal to the cortical
surface locally.

e_loc = - grad(pe)/|grad(pe)| where Laplacian(pe) = 0 with the following BCs:
pe = 1 @ cortical surface
pe = 0 @ ventricular surface
d pe / d n = 0 @ brain stem cut plane

@author: Tamas Istvan Jozsa
"""

# Python imports
import time
import argparse
import os
import sys

# FEniCSx imports
import dolfinx
from dolfinx import log
import basix
import basix.ufl
import numpy as np
from mpi4py import MPI

# Local imports
from src.X_version.io import IO_functions, permeability_initialiser_IO as perm_IO
from src.X_version.utils import suppl_fcts

# Global settings
log.set_log_level(log.LogLevel.WARNING) # Equivalent to set_log_level(50) in legacy FEniCS


def create_perm_initialiser_parser():
    """
    Create an argument parser for permeability tensor initialization.

    Returns:
        argparse.ArgumentParser: Parser with an option to set a config file path.
    """
    parser = argparse.ArgumentParser(description="perfusion computation based on multi-compartment Darcy flow model")
    parser.add_argument("--config_file", help="path to configuration file",
                        type=str, default='./configs/config_permeability_initialiser.yaml')
    return parser


def save_outputs_perm_initialiser(comm, results_folder, mesh, K1, e_loc, main_direction):
    """
    Save key output fields (K1_form, e_loc, main_direction) to XDMF files.

    Args:
        comm (mpi4py.MPI.Comm): MPI communicator.
        results_folder (str): Path to the output directory.
        mesh (dolfinx.mesh.Mesh): The mesh object.
        K1 (dolfinx.fem.Function): Computed permeability tensor field.
        e_loc (dolfinx.fem.Function): Local vessel direction vector field.
        main_direction (dolfinx.fem.Function): Primary direction vector (for visualization).
    """
    # K1_form
    k1_filename = results_folder + 'K1_form.xdmf'
    with dolfinx.io.XDMFFile(comm, k1_filename, "w") as file:
        file.write_mesh(mesh)
        file.write_function(K1)

    # e_loc
    eloc_filename = results_folder + 'e_loc.xdmf'
    with dolfinx.io.XDMFFile(comm, eloc_filename, "w") as file:
        file.write_mesh(mesh)
        file.write_function(e_loc)

    # Main_direction is non-essential output
    main_dir_filename = results_folder + 'main_direction.xdmf'
    with dolfinx.io.XDMFFile(comm, main_dir_filename, "w") as file:
        file.write_mesh(mesh)
        file.write_function(main_direction)


def save_variables(rank, comm, configs, results_folder, mesh, K1, e_loc, main_direction):
    """
    Save output variables as specified in the config file.

    Args:
        rank (int): MPI rank of the process (only rank 0 prints warnings).
        comm (mpi4py.MPI.Comm): MPI communicator.
        configs (dict): Configuration dictionary loaded from YAML.
        results_folder (str): Directory where results will be saved.
        mesh (dolfinx.mesh.Mesh): The mesh object.
        K1 (dolfinx.fem.Function): Permeability tensor field.
        e_loc (dolfinx.fem.Function): Local vessel direction vector field.
        main_direction (dolfinx.fem.Function): Primary direction vector field.
    """
    results = {}
    output_variables = configs['output']['res_vars']
    if len(output_variables) > 0:
        results['K1_form'] = K1
        results['e_loc'] = e_loc
        results['main_direction'] = main_direction
    else:
        IO_functions.print0('No variables have been defined for saving!')

    # Save variables
    res_keys = set(results.keys())
    for variable in output_variables:
        if variable in res_keys:
            filename = results_folder + variable + '.xdmf'
            with dolfinx.io.XDMFFile(comm, filename, "w") as file:
                file.write_mesh(mesh)
                file.write_function(results[variable])
        else:
            IO_functions.print0('warning: ' + variable + ' variable cannot be saved - variable undefined!')


def saving_results(rank, comm, configs, results_folder, mesh, K1, e_loc, main_direction):
    """
    Save computed permeability-related fields using two helper functions.

    Args:
        rank (int): MPI rank of the process.
        comm (mpi4py.MPI.Comm): MPI communicator.
        configs (dict): Configuration dictionary.
        results_folder (str): Output directory.
        mesh (dolfinx.mesh.Mesh): The mesh object.
        K1 (dolfinx.fem.Function): Permeability tensor field.
        e_loc (dolfinx.fem.Function): Local vessel orientation vector.
        main_direction (dolfinx.fem.Function): Main direction vector.
    """
    save_outputs_perm_initialiser(comm, results_folder, mesh, K1, e_loc, main_direction)
    save_variables(rank, comm, configs, results_folder, mesh, K1, e_loc, main_direction)


def main():
    # Define MPI variables
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    IO_functions.print0('Step 1: Reading input files')

    # Get arguments
    parser = create_perm_initialiser_parser()
    config_file = parser.parse_args().config_file

    # Assuming IO_fcts.perm_init_config_reader_yml is now adapted for FEniCSx and handles YAML reading
    configs = perm_IO.permeability_initialiser_config_reader_yaml(config_file)
    physical_configs = configs['physical']
    results_folder = configs['output']['res_fldr']

    # Read mesh
    mesh, subdomains, boundaries = IO_functions.mesh_reader(comm, configs['input']['mesh_file'])

    # Compute permeability tensor
    IO_functions.print0('Step 2: Computing permeability tensor')

    # Geometric dimension of the mesh
    geometric_dimension = mesh.geometry.dim

    element_tensor = basix.ufl.element("DG", mesh.basix_cell(), 0, shape=(geometric_dimension, geometric_dimension))
    K_space = dolfinx.fem.functionspace(mesh, element_tensor)

    # Compute vessel orientation and main direction
    e_loc, main_direction = suppl_fcts.compute_vessel_orientation(subdomains, boundaries, mesh, results_folder,
                                                               configs['output']['save_subres'])

    start1 = time.time()
    # Compute permeability tensor
    K1 = suppl_fcts.permeability_tensor_computation(K_space, subdomains, mesh, physical_configs.get('e_ref'), e_loc,
                                   physical_configs.get('K1_form'))
    end1 = time.time()
    IO_functions.print0("Permeability tensor computation on processor 0 took ", '{:.2f}'.format(end1 - start1), '[s]\n')

    IO_functions.print0('Step 3: Saving output files')
    saving_results(rank, comm, configs, results_folder, mesh, K1, e_loc, main_direction)


if __name__ == "__main__":
    main()