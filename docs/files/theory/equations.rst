Governing equations
===================

The software aims to solve cerebral perfusion under both healthy and stroke conditions.
Two different models are implemented: a **single-compartment Darcy-Flow model** and a **multi-compartment Darcy-Flow model**.

- The **single-compartment model** focuses on the arterioles and is named *a*.
- The **multi-compartment model** is named *acv* and includes the arterioles, capillaries, and venules.

Single-Compartment Model
------------------------

The *a* model is governed by the following equation:

.. math::

   \nabla \cdot (K_a \nabla p_a) - \beta_{total} (p_a - p_v) = 0

where:

- :math:`K_a` represents the permeability tensor of the arterioles,
- :math:`p_a` is the pressure in the arterioles,
- :math:`p_v` is the pressure in the venules,
- :math:`\beta_{total}` models the total fluid exchange between the arterioles and venules
  (see `Miller et al. 2021 <https://doi.org/10.1007/s10439-021-02788-y>`_ and
  `Józsa et al. 2020 <https://doi.org/10.1016/j.jbiomech.2020.109906>`_).

Multi-Compartment Model
-----------------------

The **single-compartment model** assumes constant venous pressure, governed by the equation above.

In contrast, the **multi-compartment model** introduces variable venous pressure and uses three governing equations:

.. math::

   \nabla \cdot (K_a \nabla p_a) - \beta_{ac} (p_a - p_c) = 0

.. math::

   \nabla \cdot (K_c \nabla p_c) + \beta_{ac} (p_a - p_c) - \beta_{cv}(p_c - p_v) = 0

.. math::

   \nabla \cdot (K_v \nabla p_v) + \beta_{cv} (p_c - p_v) = 0

where:

- :math:`K_a`, :math:`K_c`, :math:`K_v` are the permeability tensors of arterial, capillary, and venous compartments,
- :math:`p_a`, :math:`p_c`, :math:`p_v` are the pressures in each compartment,
- :math:`\beta_{ac}` and :math:`\beta_{cv}` represent fluid exchange coefficients between compartments
  (see `Miller et al. 2021 <https://doi.org/10.1007/s10439-021-02788-y>`_ and
  `Józsa et al. 2020 <https://doi.org/10.1016/j.jbiomech.2020.109906>`_).

Numerical Implementation
------------------------

The equations are **linear in pressure** and solved using the **finite element method (FEM)**.

This is achieved using the **FEniCS Project**, an open-source platform for solving partial differential equations (PDEs) with FEM
(`FEniCS Project <https://fenicsproject.org/>`_).
The GEMINI project uses the **FEniCS Legacy version**, which allows PDEs to be expressed in a notation very close to their mathematical form.

This is achieved through the **Unified Form Language (UFL)**, embedded in Python, enabling the definition of:

- weak forms of PDEs,
- boundary conditions,
- material properties,

in a natural and readable way, without manually coding finite element assembly routines.
**DOLFIN**, the user interface and computational backend of FEniCS Legacy, handles these implementations.

Stroke Simulations
------------------

Two main simulations are implemented to solve the governing equations in the context of stroke:

- ``permeability_initialiser.py`` – initialises the permeability tensors required for the models.
- ``basic_flow_solver.py`` – solves the equations using finite element methods.

Learn more about these in the `pipeline page <pipeline.html>`_.

