Using a container
=================

Using a container is a way to reproduce quickly the results of the simulations.
A container image includes all the dependencies needed to run the software.
The project supports Singularity, also named Apptainer, as a container platform.
A definition of the specific container is available in the container folder, under the container.def file.

Building the container
----------------------

To build the container, you need to use the following command:

.. code-block:: bash

    apptainer build containers/container.sif containers/container.def

This step can take up to 20 min, you will need to be patient! The container will create both the environments
for FEniCS Legacy and FEniCS X.

If you are using Ares as a running environment, it is important to know that Singularity/Apptainer is not installed on
the login nodes.
To build the container, you can submit a job to do so. However, we recommend to build it on your local machine and
import it using a scp command.

Running the code inside the container
-------------------------------------

Once the container is created, ie. you have access to the .sif file, you are able to run the files.
The container is adapted to the Legacy and the X version of the code.

You can use a similar command, adapted to your needs. The following one is adapted to the Legacy version.

.. code-block:: bash

    apptainer exec \
    --bind $PBS_O_WORKDIR:/mnt \
    --bind $PBS_O_WORKDIR/.jitcache:/conda_stuff/more_files/envs/fenics-legacy/.cache \
    --pwd /mnt \
    containers/container.sif \
    /usr/local/bin/mpirun-fenics-legacy -np 4 --host localhost \
    python3 -m src.Legacy_version.simulation.basic_flow_solver \
    --config_file configs/config_basic_flow_solver.yaml

The repository contains submission files already filled for the two simulations, permeability_initialiser.py and
basic_flow_solver.py, in serial and parallel and for all cases, healthy, LMCAo and RMCAo.
