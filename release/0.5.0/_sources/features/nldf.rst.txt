.. _nldf_feat:

Nonlocal Density Features (NLDF)
================================

Nonlocal density features, or NLDFs, are the original type of nonlocal feature used
to train Cider models. The general idea is to take the density :math:`n(\mathbf{r})`,
multiply it by a kernel function :math:`k(\mathbf{r}-\mathbf{r}')`, and integrate
over real space to obtain features :math:`G(\mathbf{r})` that are nonlocal in the density:

.. math::

   G[n](\mathbf{r}) = \int \mathrm{d}^3\mathbf{r}'\,
   k(\mathbf{r}-\mathbf{r}') n(\mathbf{r}')

This is a fairly simple convolution-based descriptor, but it has a significant shortcoming.
In order to guarantee a reasonable and finite value for :math:`G(\mathbf{r})`,
:math:`k(\mathbf{r}-\mathbf{r}')` must decay exponentially on some length scale
as :math:`|\mathbf{r}-\mathbf{r}'|` grows large. However, choosing a single
length-scale will result in nontrivial behavior of the feature under
:ref:`Uniform Scaling <unif_scaling>`. To cure this problem, we need a
density-dependent kernel :math:`k[n](\mathbf{r}-\mathbf{r}')`. For a scaled
density :math:`n_\lambda(\mathbf{r})=\lambda^3 n(\lambda\mathbf{r})`,
if :math:`k[n_\lambda](\mathbf{r}-\mathbf{r}')=\lambda^j k[n](\lambda(\mathbf{r}-\mathbf{r}'))`,
then

.. math:: G[n_\lambda](\mathbf{r}) = \lambda^j G[n](\lambda\mathbf{r})

These features can then be regularized to be scale invariant and used in
exchange-functional models.

Three NLDF versions are implemented in the molecular feature framework:
``i``, ``j``, and ``k``. Version ``i`` and ``j`` features can be evaluated
together through the combined ``ij`` setting. The packaged CIDER23X and
CIDER26XC models use version ``j``. In release 0.5.0, GPAW supports version
``j`` only; the other versions and the alternative kernel specifications are
developmental interfaces rather than production recommendations.


Version J
---------

Version J is the NLDF implemented in :footcite:t:`CIDER23X`. It takes the form

.. math:: G_i[n](\mathbf{r}) = \int \text{d}^3\mathbf{r}' \exp(-(a_i[n](\mathbf{r})+a_0[n](\mathbf{r}'))|\mathbf{r}-\mathbf{r}'|^2) n(\mathbf{r}')

For a functional with GGA semilocal features, :math:`a_i[n](\mathbf{r})` (and :math:`a_0[n](\mathbf{r})`) take the form

.. math:: a_i[n](\mathbf{r}) = \pi\left(\frac{n}{2}\right)^{2/3} \left[A_i + B_i\left(\frac{|\nabla n|^2}{8n\tau_0}\right)\right]

For a functional with meta-GGA semilocal features, :math:`a_i[n](\mathbf{r})` takes the form

.. math:: a_i[n](\mathbf{r}) = \pi\left(\frac{n}{2}\right)^{2/3} \left[A_i + B_i\left(\frac{|\nabla n|^2}{8n\tau_0}\right) + C_i\left(\frac{\tau}{\tau_0}-1\right)\right]

:math:`A_i,B_i,C_i` are tunable parameters. Conventionally :math:`B_i=0` is used for the meta-GGA case (with :math:`C_i` finite),
but one can choose nonzero values in the CiderPress code. Using different :math:`A_i,B_i,C_i` for different :math:`i`
allows one to compute multiple features simultaneously (at roughly the same cost as one feature using the
fast NLDF evaluation algorithm). However, :math:`A_0,B_0,C_0` must be the same for all features. Both the GGA and meta-GGA
constructions of :math:`a_i[n](\mathbf{r})` obey

.. math:: a_i[n_\lambda](\mathbf{r}) = \lambda^2 a_i[n](\lambda \mathbf{r})

As a result of the above relationship, :math:`G_i[n](\mathbf{r})` is scale-invariant:

.. math:: G_i[n_\lambda](\mathbf{r}) = G_i[n](\lambda\mathbf{r})

In CiderPress, there is also an **experimental** feature to multiply the density :math:`n(\mathbf{r}')` by another
function :math:`b(\mathbf{r}')` before integrating over the kernel function. In this case,
:math:`G_i[n](\mathbf{r})` has the same uniform scaling behavior as :math:`b(\mathbf{r}')`.

The production models use the squared-exponential specification ``se``.
The settings layer also defines ``se_ar2``, ``se_a2r4``, and
``se_erf_rinv`` variants. Rational-kernel variants require the explicit
experimental ``vdw_param`` setting. These strings, their ordered parameters,
and their normalization are part of the serialized model and must not be
changed at inference time; see
:class:`~ciderpress.dft.settings.NLDFSettingsVJ`.

Version I
---------

Version I modifies Version J in two ways. First, :math:`a_i[n](\mathbf{r})`, the exponent contribution computed
at point :math:`\mathbf{r}` is not used; only :math:`a_0[n](\mathbf{r}')`. Second, several different options
are introduced for the integration kernel, resulting in a general form

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

``se_lapl`` is the historical setting name. The expression shown is the
implemented kernel; it is not the ordinary full three-dimensional Laplacian
of :math:`\exp(-aR^2)`.

There are also two options for vector features of the form

.. math::

   \mathbf{g}_*[n](\mathbf{r}) = \int \mathrm{d}^3\mathbf{r}'\,
   (\mathbf{r}'-\mathbf{r})
   k_*(a_0[n](\mathbf{r}'), |\mathbf{r}-\mathbf{r}'|)
   n(\mathbf{r}')

There are two options for :math:`k` in the above formula:

* ``se_grad`` uses :math:`k_*(a,R)=a\exp(-aR^2)`.
* ``se_rvec`` uses :math:`k_*(a,R)=\exp(-aR^2)`.

To construct rotationally invariant descriptors, the vector integrals :math:`\mathbf{g}_*[n](\mathbf{r})`
must be dotted with the density gradient or with another vector integral. For
example, the ``se_grad`` vector can be dotted with itself to form the scalar
:math:`G=\mathbf{g}_*\cdot\mathbf{g}_*`.

As with Version J, there is experimental support for multiplying :math:`n(\mathbf{r}')` by another
function :math:`b(\mathbf{r}')` before integrating.

Unfortunately, while all the above Version I variants are supported in the code (in any combination),
most of them either have numerical precision issues (i.e. they are difficult to compute with high
precision using the fast NLDF algorithms) or are physically unrealistic (being too large in the core
region or similar issues). The two most physically and numerically sensible seem to be ``se_ap``
and ``se_grad`` dotted with itself.

Version K
---------

Version K is a modification of Version J meant to remove the need for the squared-exponential kernel
to depend on the density at :math:`\mathbf{r}'`. However, without the dependence on :math:`a_0[n](\mathbf{r}')`,
the NLDFs can have large contributions from the core electrons in the valence region, which 
is physically unrealistic. To fix this, we multiply the density by a function that
decays when :math:`a_i(\mathbf{r})\ll a_0(\mathbf{r}')`:

.. math::

   G_i[n](\mathbf{r}) = \int \mathrm{d}^3\mathbf{r}'\,
   \exp\!\left(-a_i[n](\mathbf{r})|\mathbf{r}-\mathbf{r}'|^2\right)
   \exp\!\left(-\frac{3a_0[n](\mathbf{r}')}{2a_i[n](\mathbf{r})}\right)
   n(\mathbf{r}')

We have alternative options for the damping function, but the above exponential is the only supported
one currently.

.. footbibliography::
