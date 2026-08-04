Theory and Functional Design
============================

The CIDER framework separates three questions that are often combined in a
single analytic density functional:

* Which electronic information is exposed to the model?
* Which exact constraints are built into that representation and energy form?
* How is the remaining functional dependence learned and evaluated?

The pages in this section follow those questions from the CIDER functional
form through Gaussian-process fitting and numerical evaluation.  The feature
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
