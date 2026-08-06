.. _plans_module:

Feature Evaluation Plans
========================

Plans implement the forward evaluation and adjoint derivatives associated
with settings objects.  Backend adapters extract the data needed
to compute the features from the density and orbitals and pass it to the plans.
Each plan preserves raw feature order and spin conventions.

.. automodule:: ciderpress.dft.plans
   :members: SemilocalPlan, SemilocalPlan2, SDMXPlan, SDMXFullPlan, SDMXIntPlan, NLDFAuxiliaryPlan, NLDFGaussianPlan, NLDFSplinePlan
   :undoc-members:
