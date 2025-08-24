# Plotter

This folder contains scripts to plot the results. 

Particularly, the [creating_png_slices_paraview](creating_png_slices_paraview.py) creates ``.png`` files in
Paraview (an open source visualisation tool) automatically. However, the script must be run from the 
terminal with the following command: 

````bash
& "path_to_pvpython.exe" .\src\Legacy_version\plotter\creating_png_slices_paraview.py
````

The ``path_to_pvpython.exe`` variable need to be updated to the correct path of the Paraview python executable. 
Typically, the path should be something like the following: ``C:\Program Files\ParaView 5.13.3\bin\pvpython.exe``

It will need a ``perfusion.xdmf`` file. 