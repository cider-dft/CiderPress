Theory and Functional Design
============================

The CIDER framework separates three questions that are often combined in a
single analytic density functional:

* Which electronic information is exposed to the model?
* Which exact constraints are built into that representation and energy form?
* How is the remaining functional dependence learned and evaluated?

The pages in this section answer those questions, covering the CIDER functional
form, Gaussian-process fitting, and numerical evaluation of features.  The feature
definitions themselves are collected in :doc:`../features/features`.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   framework
   uniform_scaling
   full_xc
   gp
   nldf_numerical
   numerical_evaluation
