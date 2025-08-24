Performance benchmark
=====================

The project supports two different HPC systems - potentially three in the future - and provide container to use the
software on those platforms. However, the performance might still be different on each systems.
Some tests have been run to create a benchmark of the performance of the software.

The tests performed gives the average of 10 runs of each simulations. Only the two main simulations are assessed: the
permeability_initialiser.py and the basic_flow_solver.py. However, the basic_flow_solver is tested for the three
different use cases (healthy, LMCAo and RMCAo). The results are similar for each case, but they have still be run each
10 times.
All times given are in seconds.
The tests have been performed with the container provided in the project, on the Legacy version of the code.

Crescent2
---------

.. list-table:: Performance on Crescent2
   :header-rows: 1
   :widths: 25 15 15

   * - Simulation
     - Serial execution
     - Parallel execution
   * - Tensor initialisation
     - 421,46
     - 324,24
   * - Basic flow solver (total)
     - 101,41
     - 77,92
   * - Step 1
     - 28,09
     - 60,04
   * - Step 2
     - 8,42
     - 2,30
   * - Step 3
     - 64,88
     - 15,57


Ares
----

.. list-table:: Performance on Ares
   :header-rows: 1
   :widths: 25 15 15

   * - Simulation
     - Serial execution
     - Parallel execution
   * - Tensor initialisation
     - 131,94
     - 133,02
   * - Basic flow solver (total)
     - 86,47
     - 365,03
   * - Step 1
     - 22,32
     - 103,09
   * - Step 2
     - 8
     - 72,75
   * - Step 3
     - 56,10
     - 189,18


Local
-----

The local computer used to make those tests is equipped with the processor AMD Ryzen 7 5800H which has 8 cores.
It is supposed to be a powerful computer designed for video games. The performance on your local computer might change
a lot.

.. list-table:: Performance on local
   :header-rows: 1
   :widths: 25 15 15

   * - Simulation
     - Serial execution
     - Parallel execution
   * - Tensor initialisation
     - 71,90
     - 30,91
   * - Basic flow solver (total)
     - 69,35
     - 110,13
   * - Step 1
     - 18,30
     - 85,08
   * - Step 2
     - 5,75
     - 3,26
   * - Step 3
     - 45,30
     - 21,78


Additional remarks
------------------

As the results for Crescent2 and Ares, two supercomputers, are very different, it is important to remark that even with
a container which transports the environment, the performance might not be conserved across environments.
A change in a version of a compiler, for example, can introduce a huge performance change.

The parallel optimisation on your local computer might also introduce a loss that is not expected.
It explains why the basic_flow_solver can take more time to run in parallel than in serial. The first step, responsible
for mesh reading and variable initialisation, is a lot longer in parallel in local, probably because the local computer
used wasn't optimised as an high-performance computing system.
The local performance must be assessed on your personal computer.
