import os
from dolfin import *
import untangle
import yaml


def basic_flow_config_reader2(input_file_path, parser):
    """
    Read and process simulation configuration from either XML or YAML file formats.

    This function loads configuration settings for the basic flow solver from the specified
    input file (`.xml` or `.yaml`). It converts the data into Python-native types using
    either `untangle` (for XML) or `yaml` (for YAML). It also applies command-line argument
    overrides using the provided `argparse.ArgumentParser` parser instance.

    Supported override fields:
        - `res_fldr`: Output results folder
        - `mesh_file`: Path to mesh file
        - `inlet_boundary_file`: Path to inlet boundary condition file

    Args:
        input_file_path (str): Path to the input configuration file (.xml or .yaml).
        parser (argparse.ArgumentParser): Argument parser that may provide override values.

    Returns:
        object: Configuration object with accessible attributes (via dot notation).

    Raises:
        Exception: If the input file format is unsupported.
    """
    if input_file_path.endswith('xml'):
        configs = untangle.parse(input_file_path).basic_flow_solver_settings
        from pydoc import locate

        for mydir in dir(configs):
            for mysubdir in dir(getattr(configs, mydir)):
                mydata = getattr(getattr(configs, mydir), mysubdir)
                mydata_type = mydata['type']
                if mydata_type == None:
                    setattr(getattr(configs, mydir), mysubdir, mydata.cdata.strip())
                elif mydata_type != 'bool':
                    converter = locate(mydata_type)
                    setattr(getattr(configs, mydir), mysubdir, converter(mydata.cdata.strip()))
                else:
                    setattr(getattr(configs, mydir), mysubdir, mydata.cdata.strip() == 'True')
    elif input_file_path.endswith('yaml'):
        with open(input_file_path, "r") as config_file:
            configs = yaml.load(config_file, yaml.SafeLoader)
        configs = dict2obj(configs)
    else:
        config_format = os.path.splitext(input_file_path)[-1]
        raise Exception("unknown input file format: " + config_format)

    if parser.parse_args().res_fldr != None:
        configs.output.res_fldr = parser.parse_args().res_fldr

    if parser.parse_args().mesh_file != None:
        configs.input.mesh_file = parser.parse_args().mesh_file

    if parser.parse_args().inlet_boundary_file != None:
        configs.input.inlet_boundary_file = parser.parse_args().inlet_boundary_file
    return configs


def basic_flow_config_reader_yml(input_file_path, parser):
    """
    Load and update simulation configuration from a YAML file and command-line parser overrides.

    This function reads simulation parameters from a `.yaml` configuration file and overrides
    specific fields (if provided) using arguments parsed from an `argparse.ArgumentParser` instance.
    It also ensures the results folder exists and saves the final configuration into it.

    Args:
        input_file_path (str): Path to the YAML configuration file.
        parser (argparse.ArgumentParser): Argument parser containing optional overrides.

    Returns:
        dict: Parsed and possibly updated configuration dictionary.

    Raises:
        Exception: If the input file format is unsupported (non-YAML).
    """
    if input_file_path.endswith('yaml'):
        with open(input_file_path, "r") as config_file:
            configs = yaml.load(config_file, yaml.SafeLoader)
    else:
        config_format = os.path.splitext(input_file_path)[-1]
        raise Exception("Unknown input file format: " + config_format)

    if hasattr(parser.parse_args(), 'res_fldr'):
        if parser.parse_args().res_fldr is not None:
            configs['output']['res_fldr'] = parser.parse_args().res_fldr

    if hasattr(parser.parse_args(), 'mesh_file'):
        if parser.parse_args().mesh_file is not None:
            configs['input']['mesh_file'] = parser.parse_args().mesh_file

    if hasattr(parser.parse_args(), 'inlet_boundary_file'):
        if parser.parse_args().inlet_boundary_file is not None:
            configs['input']['inlet_boundary_file'] = parser.parse_args().inlet_boundary_file

    comm = MPI.comm_world
    rank = comm.Get_rank()
    if rank == 0:
        if not os.path.exists(configs['output']['res_fldr']):
            os.makedirs(configs['output']['res_fldr'])
        with open(configs['output']['res_fldr'] + 'settings.yaml', 'w') as outfile:
            yaml.dump(configs, outfile, default_flow_style=False)

    return configs


class dict2obj(dict):
    def __init__(self, my_dict):
        for a, b in my_dict.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [dict2obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, dict2obj(b) if isinstance(b, dict) else b)