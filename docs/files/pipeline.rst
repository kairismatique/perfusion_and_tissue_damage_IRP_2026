Pipeline
========

The ``perfusion_runnner.sh`` file introduces a generic pipeline for the simulations.

Main pipeline
-------------

The usual pipeline for the simulation is described below.

1. **Extract the brain meshes**:

- Unpack `brain_meshes.tar.xz` into the project directory.
- This command only need to be run once.
- You can use the following command if necessary:

.. code-block:: bash

    tar xf ../brain_meshes.tar.xz

2. **Initialise the permeability tensor**:

- Run `permeability_initialiser.py` to compute the permeability tensor.
- This simulation only need to be run once.
- You can use the following command to run it in parallel from the project directory:

.. code-block:: bash

    mpirun -n 4 python3 -m src.Legacy_version.simulation.permeability_initialiser --config_file ./configs/config_permeability_initialiser.yaml

- Uses `config_permeability_initialiser.yaml` as a configuration file. It contains information about the mesh file, the output folder or physical parameters such as the arteriole/venule permeability tensor form.
- This simulation will take up to 5 min with 4 cores, depending of the environment you are using (HPC systems, local). More information about the performance can be found here: `Performance <performance.html>`_.

3. **Create boundary conditions**:

- Generate boundary conditions for occluded scenarios (e.g., healthy, LMCAo, RMCAo):
- You can use the following command to run it in serial from the project directory:

.. code-block:: bash

    python3 -m src.Legacy_version.simulation.BC_creator --config_file ./configs/config_basic_flow_solver.yaml

- The command presented here is adapted to a healthy case. To adapt it to an occluded case, please update the argument of the command.
- Update `config_basic_flow_solver.yaml` to point to the generated boundary conditions file and change the output folder to avoid overwriting.

4. **Solve Flow**:

- Run `basic_flow_solver.py` for pressure and velocity fields.
- You can use the following command to run it in parallel from the project directory:

.. code-block:: bash

    mpirun -n 4 python3 -m src.Legacy_version.simulation.basic_flow_solver --config_file ./configs/config_basic_flow_solver.yaml

- Uses `config_basic_flow_solver.yaml` as a configuration file. It contains information about the input and outputs files and folders, about physical, simulation and optimisation parameters.
- This simulation will take up to 6 min with 4 cores, depending of the environment you are using (HPC systems, local). More information about the performance can be found here: `Performance <performance.html>`_.
- The simulation has already three different cases supported: a healthy brain, a LMCAo and a RMCAo case. Three configurations files are available.:
    - ``config_basic_flow_solver.yaml``
    - ``config_basic_flow_solver_LMCAo.yaml``
    - ``config_basic_flow_solver_RMCAo.yaml``

Additional simulations
----------------------

Additional results can be produced using the following simulations, after the main pipeline have been produced one.

1. **Convert the finite element results to a NIFTI format**:

- Run `convert_res2img.py` to convert a ``.h5`` or ``.xdmf`` to a ``.nii.gz`` format.
- You can use the following command to do so:

.. code-block:: bash

    python3 -m src.Legacy_version.io.convert_res2img --config_file ./results/p0000/perfusion_healthy/settings.yaml

- Use the configuration file of the results you want to convert. Three files are available, once the main pipeline has been run for the three cases (healthy, LMCAo, RMCAo):
    - ``./results/p0000/perfusion_healthy/settings.yaml``
    - ``./results/p0000/perfusion_LMCAo/settings.yaml``
    - ``./results/p0000/perfusion_RMCAo/settings.yaml``

2. **Compute lesion volume proxies from brain perfusion images**:

- Run `lesion_comp_from_img.py` to compute the lesion volume proxies form a healthy and occluded brain perfusion images.
- Use the following command to run it in serial from the project directory:

.. code-block:: bash

    python3 -m src.Legacy_version.simulation.lesion_comp_from_img --healthy_file ./results/p0000/perfusion_healthy/perfusion.nii.gz --occluded_file ./results/p0000/perfusion_RMCAo/perfusion.nii.gz

- Use the healthy and occluded images you want to test. The inputted files should be in the NIFTI format (``.nii.gz``).

4. **Compute infarct volumes across a range of perfusion thresholds**:

- Run `infarct_calculation_thresholds.py` to compute infarct volumes across a range of perfusion thresholds.
- Use the following command to run it in parallel from the project directory:

.. code-block:: bash

    mpirun -n 4 python3 -m src.Legacy_version.simulation.infarct_calculation_thresholds --config_file ./configs/config_basic_flow_solver_RMCAo.yaml --baseline ./results/p0000/perfusion_healthy/perfusion.xdmf --occluded ./results/p0000/perfusion_RMCAo/perfusion.xdmf

- Use the configuration file of the results you want to convert. Three files are available, once the main pipeline has been run for the three cases (healthy, LMCAo, RMCAo):
    - ``./results/p0000/perfusion_healthy/settings.yaml``;
    - ``./results/p0000/perfusion_LMCAo/settings.yaml``;
    - ``./results/p0000/perfusion_RMCAo/settings.yaml``.
- Adapt the occluded file to the configuration you have chosen.
