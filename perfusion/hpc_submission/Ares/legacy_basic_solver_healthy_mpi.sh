#!/bin/bash -l
#SBATCH -J legacy_basic_solver_healthy_mpi
#SBATCH -N 1
#SBATCH --time=1:00:00
#SBATCH -A plggemini2025-cpu
#SBATCH -p plgrid
#SBATCH --output="/net/pr2/projects/plgrid/plgggemini/perfusion_charlotte_thesis/hpc_submission/Ares/outputs/out/legacy_basic_solver_healthy_mpi.out"
#SBATCH --error="/net/pr2/projects/plgrid/plgggemini/perfusion_charlotte_thesis/hpc_submission/Ares/outputs/err/legacy_basic_solver_healthy_mpi.err"

# Paths
REPO_DIR="/net/pr2/projects/plgrid/plgggemini/perfusion_charlotte_thesis/perfusion"
singularity exec \
    --bind "$REPO_DIR:/mnt/project" \
    containers/container.sif \
    bash -c "
        export HYDRA_LAUNCHER=fork \
        && export DIJITSO_CACHE_DIR=/mnt/project/tmp_dijitso_cache \
        && export XDG_CACHE_HOME=\$DIJITSO_CACHE_DIR \
        && export FFC_CACHE_DIR=\$DIJITSO_CACHE_DIR \
        && mkdir -p \$DIJITSO_CACHE_DIR \
        && echo 'Running the basic flow solver for a healthy case' \
        && cd /mnt/project \
        && /usr/local/bin/mpirun-fenics-legacy -np 4 --host localhost python3 -m src.Legacy_version.simulation.basic_flow_solver --config_file ./configs/config_basic_flow_solver.yaml
    "