# Containers

This folder contains the information and files related to the container supported by the project. 
Three files are present:

- [container.def](container.def) which contains the information about the container and how to build it.
- [environment.yaml](environment.yaml) which contains information about the conda environment associated to FEniCS Legacy.
- [create_container.sh](create_container.sh) which is a script to create the container image (it could take up to 30 min to build).
- (Optional) container.sif, the container image create by the [create_container.sh](create_container.sh) script.
