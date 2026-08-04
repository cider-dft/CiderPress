Model Loading
=============

Packaged names and explicit files enter the same mapped-model interface.
``BUILTIN_MODELS`` is the authoritative list shipped in the installed
package.  A real filesystem path takes precedence over a packaged alias, and
the optional ``.yaml`` suffix is accepted for built-in names.

YAML and joblib inputs reconstruct Python objects and must be trusted.  Use
the release checksums in :doc:`../../usage/production_models` when an exact
artifact identity matters.

.. automodule:: ciderpress.dft.model_utils
   :members:
   :undoc-members:
   :show-inheritance:
