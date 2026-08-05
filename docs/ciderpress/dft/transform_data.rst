Bounded Regression Coordinates
==============================

After physical normalization, a feature can still be unbounded or strongly
skewed.  The map classes in this module define the algebraic coordinates used
by the covariance kernel or mapped evaluator.  Each map records its raw input
indices and parameters, evaluates one transformed coordinate, and propagates
its derivative back to every raw input it uses.

``FeatureList`` stores these maps in regression order.  ``fill_vals_`` applies
the forward maps, ``fill_derivs_`` applies their adjoints, and YAML
serialization preserves map types and parameters.  Examples include the
rational map :math:`y=\gamma x/(1+\gamma x)` and the ``SLTMap`` used for the
bounded CIDER26XC kinetic-energy indicator.  Model loading supplies the exact
list associated with a packaged functional.

.. automodule:: ciderpress.dft.transform_data
   :members:
