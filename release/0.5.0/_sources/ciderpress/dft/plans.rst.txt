.. _plans_module:

Feature Evaluation Plans
========================

Plans implement the forward evaluation and adjoint derivatives associated
with settings objects.  Backend adapters supply their grid and electronic
arrays, and the plan preserves raw feature order and spin conventions.

.. automodule:: ciderpress.dft.plans
   :members: SemilocalPlan, SemilocalPlan2, SDMXPlan, SDMXFullPlan, SDMXIntPlan, NLDFAuxiliaryPlan, NLDFGaussianPlan, NLDFSplinePlan
