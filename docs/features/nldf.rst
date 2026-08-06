.. _nldf_feat:

Nonlocal Density Features (NLDF)
================================

Nonlocal density features (NLDFs) describe the density in a neighborhood of
each integration point.  A fixed-kernel convolution has the form

.. math::

   G[n](\mathbf{r}) = \int \mathrm{d}^3\mathbf{r}'\,
   k(\mathbf{r}-\mathbf{r}') n(\mathbf{r}')

An exponentially decaying fixed kernel introduces a dimensional length scale
that conflicts with uniform coordinate scaling.  CIDER resolves this
problem uses density-dependent kernel exponents.  For a scaled
density :math:`n_\lambda(\mathbf{r})=\lambda^3 n(\lambda\mathbf{r})`,
if :math:`k[n_\lambda](\mathbf{r}-\mathbf{r}')=\lambda^j k[n](\lambda(\mathbf{r}-\mathbf{r}'))`,
then

.. math:: G[n_\lambda](\mathbf{r}) = \lambda^j G[n](\lambda\mathbf{r})

The resulting power law can be combined with a normalizer to produce the
scaling behavior required by the functional. Typically, this means
constructing the normalized feature to be scale-invariant, as explained
in :doc:`uniform_scaling`.

Three NLDF versions are implemented in the molecular feature framework:
``i``, ``j``, and ``k``.  Version ``i`` and ``j`` features can be evaluated
together through the combined ``ij`` setting.  The packaged CIDER23X and
CIDER26XC models use version ``j``.  In release 0.5.0, GPAW supports version
``j``.  The other versions and alternative kernel specifications are
model-development interfaces.


Version J
---------

Version J is the NLDF implemented in :footcite:t:`CIDER23X`. In the raw form
computed by the settings layer, it reads

.. math:: G_i[n](\mathbf{r}) = \int \text{d}^3\mathbf{r}' \exp(-(a_0[n](\mathbf{r}')+a_i[n](\mathbf{r}))|\mathbf{r}-\mathbf{r}'|^2) n(\mathbf{r}')

Equation 18 of :footcite:t:`CIDER23X` includes a constant prefactor determined
by the two exponent parameters. CiderPress represents the convolution and its
normalization in separate serialized objects: ``NLDFSettingsVJ`` defines the
raw feature, and the model's normalizer defines its uniform-electron-gas
value.

Equations 20 and 22 of :footcite:t:`CIDER23X` write the exponent with a
constant term :math:`B_i` and one modulation term :math:`C_i`, using the
reduced gradient for a GGA functional and the kinetic-energy density for a
meta-GGA functional:

.. math:: a_i[n](\mathbf{r}) = \pi\left(\frac{n}{2}\right)^{2/3} \left[B_i + C_i\left(\frac{|\nabla n|^2}{8n\tau_0}\right)\right]

.. math:: a_i[n](\mathbf{r}) = \pi\left(\frac{n}{2}\right)^{2/3} \left[B_i + C_i\left(\frac{\tau}{\tau_0}-1\right)\right]

Here :math:`\tau` is the kinetic-energy density and :math:`\tau_0` its
uniform electron gas value, both defined in :doc:`sl`. The settings layer
generalizes these to a single expression that carries both modulations,

.. math:: a_i[n](\mathbf{r}) = \pi\left(\frac{n}{2}\right)^{2/3} \left[B_i + D_i\left(\frac{|\nabla n|^2}{8n\tau_0}\right) + C_i\left(\frac{\tau}{\tau_0}-1\right)\right]

with :math:`D_i` the gradient coefficient. The packaged CIDER23X and
CIDER26XC meta-GGA models set :math:`D_i=0`, recovering Equation 22.

In the serialized settings the parameter array is ``[a0, grad_mul,
tau_mul]``, and ``tau_mul`` is ignored for a GGA exponent. Only the constant
term maps directly: :math:`B_i` is ``a0``, while :math:`D_i` and :math:`C_i`
are ``grad_mul`` and ``tau_mul`` multiplied by
:math:`1.2\,(6\pi^2)^{2/3}/\pi \approx 5.803`, the conversion applied in
:func:`~ciderpress.dft.settings.get_cider_exponent`.

Using different parameters for each :math:`i` produces several feature
length scales in one auxiliary expansion. The source exponent
:math:`a_0` is shared by all features. Both the GGA and meta-GGA
constructions obey

.. math:: a_i[n_\lambda](\mathbf{r}) = \lambda^2 a_i[n](\lambda \mathbf{r})

This exponent scaling makes :math:`G_i[n](\mathbf{r})` scale invariant:

.. math:: G_i[n_\lambda](\mathbf{r}) = G_i[n](\lambda\mathbf{r})

The settings format can multiply the source density by another semilocal
quantity :math:`b(\mathbf{r}')`. The multiplier contributes its uniform
scaling power to the raw feature. Note the the introduction of this
additional multiplier is currently experimental and not used in any
packaged production functionals.

The settings layer defines the squared-exponential specification ``se`` and
the ``se_ar2``, ``se_a2r4``, and
``se_erf_rinv`` variants. Rational-kernel variants use the ``vdw_param``
setting.  The packaged models select ``se``.  Kernel strings, their ordered
parameters, and their normalization form part of the serialized model; see
:class:`~ciderpress.dft.settings.NLDFSettingsVJ`.

Version I
---------

Version I uses the source exponent :math:`a_0[n](\mathbf r')` and omits the
target exponent :math:`a_i[n](\mathbf r)`.  Several integration kernels are
available, giving the general form

.. math:: G_*[n](\mathbf{r}) = \int \text{d}^3\mathbf{r}' k_*(a_0[n](\mathbf{r}'), |\mathbf{r}-\mathbf{r}'|) n(\mathbf{r}')

The :math:`*` symbol stands for the kernel specification. With
:math:`R=|\mathbf r-\mathbf r'|`, the scalar kernels implemented by
:class:`~ciderpress.dft.settings.NLDFSettingsVI` are:

.. list-table:: Version-I scalar kernels
   :header-rows: 1
   :widths: 25 75

   * - Setting
     - :math:`k_*(a,R)`
   * - ``se``
     - :math:`\exp(-aR^2)`
   * - ``se_r2``
     - :math:`R^2\exp(-aR^2)`
   * - ``se_apr2``
     - :math:`aR^2\exp(-aR^2)`
   * - ``se_ap``
     - :math:`a\exp(-aR^2)`
   * - ``se_ap2r2``
     - :math:`a^2R^2\exp(-aR^2)`
   * - ``se_lapl``
     - :math:`(4a^2R^2-2a)\exp(-aR^2)`

The historical name ``se_lapl`` refers to the explicit expression shown in
the table. It does not actually compute the Laplacian of the ``se``
feature as originally intended, but the implementation is subject to
change in future releases to fix this issue. No current packaged production
functionals use Version I or the ``se_lapl`` feature.

Version I also provides vector features of the form

.. math::

   \mathbf{g}_*[n](\mathbf{r}) = \int \mathrm{d}^3\mathbf{r}'\,
   (\mathbf{r}'-\mathbf{r})
   k_*(a_0[n](\mathbf{r}'), |\mathbf{r}-\mathbf{r}'|)
   n(\mathbf{r}')

The implemented vector kernels are:

* ``se_grad`` uses :math:`k_*(a,R)=a\exp(-aR^2)`.
* ``se_rvec`` uses :math:`k_*(a,R)=\exp(-aR^2)`.

To construct rotationally invariant descriptors, the vector integrals :math:`\mathbf{g}_*[n](\mathbf{r})`
must be dotted with the density gradient or with another vector integral. For
example, the ``se_grad`` vector can be dotted with itself to form the scalar
:math:`G=\mathbf{g}_*\cdot\mathbf{g}_*`.

Version I also accepts a semilocal source multiplier
:math:`b(\mathbf r')`.

Version-I variants are model-construction options.  A custom combination
defines a distinct descriptor and requires independent forward/adjoint,
scaling, and low-density validation.

NOTE: While all the above Version I variants are supported in the code (in any combination),
most of them either have numerical precision issues (i.e. they are difficult to compute with high
precision using the fast NLDF algorithms) or are physically unrealistic (being too large in the core
region or similar issues). The two most physically and numerically sensible seem to be ``se_ap``
and ``se_grad`` dotted with itself. Version I features are subject to change
in future releases pending improved numerical implementations.

Version K
---------

Version K is a modification of Version J meant to remove the need for the squared-exponential kernel
to depend on the density at :math:`\mathbf{r}'`. However, without the dependence on :math:`a_0[n](\mathbf{r}')`,
the NLDFs can have large contributions from the core electrons in the valence region, which 
is physically unrealistic. To fix this, we multiply the density by a function that
decays when :math:`a_i(\mathbf{r})<<a_0(\mathbf{r}')`:
:math:`a_i(\mathbf{r})\ll a_0(\mathbf{r}')`:

.. math::

   G_i[n](\mathbf{r}) = \int \mathrm{d}^3\mathbf{r}'\,
   \exp\!\left(-a_i[n](\mathbf{r})|\mathbf{r}-\mathbf{r}'|^2\right)
   \exp\!\left(-\frac{3a_0[n](\mathbf{r}')}{2a_i[n](\mathbf{r})}\right)
   n(\mathbf{r}')

.. footbibliography::
