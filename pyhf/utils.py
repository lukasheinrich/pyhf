import json
import jsonschema
import pkg_resources

from .exceptions import InvalidSpecification
from . import get_backend


def get_default_schema():
    r"""
    Returns the absolute filepath default schema for pyhf. This usually points
    to pyhf/data/spec.json.

    Returns:
        Schema File Path: a string containing the absolute path to the default
                          schema file.
    """
    return pkg_resources.resource_filename(__name__, 'data/spec.json')


SCHEMA_CACHE = {}


def load_schema(schema):
    global SCHEMA_CACHE
    try:
        return SCHEMA_CACHE[schema]
    except KeyError:
        pass

    with open(schema) as json_schema:
        SCHEMA_CACHE[schema] = json.load(json_schema)
    return SCHEMA_CACHE[schema]


def validate(spec, schema):
    schema = load_schema(schema)
    try:
        return jsonschema.validate(spec, schema)
    except jsonschema.ValidationError as err:
        raise InvalidSpecification(err)


def loglambdav(pars, data, pdf):
    r"""
    Two times the negative log-likelihood of the model parameters, :math:`\left(\mu, \boldsymbol{\theta}\right)`, given the observed data

    .. math::

        -2\ln L\left(\mu, \boldsymbol{\theta}\right)

    It is used in the calculation of the test statistic, :math:`t_{\mu}`, as defiend in Equation (8) in `arXiv:1007.1727`_

    .. _`arXiv:1007.1727`: https://arxiv.org/abs/1007.1727

    .. math::

       t_{\mu} = -2\ln\lambda\left(\mu\right)

    where :math:`\lambda\left(\mu\right)` is the profile likelihood ratio as defined in Equation (7)

    .. math::

       \lambda\left(\mu\right) = \frac{L\left(\mu, \hat{\hat{\boldsymbol{\theta}}}\right)}{L\left(\hat{\mu}, \hat{\boldsymbol{\theta}}\right)}

    Args:
        pars (Tensor): The parameters of the HistFactory model
        data (Tensor): The data to be considered
        pdf (|pyhf.pdf.Model|_): The HistFactory statistical model used in the likelihood ratio calculation

    .. |pyhf.pdf.Model| replace:: ``pyhf.pdf.Model``
    .. _pyhf.pdf.Model: https://diana-hep.org/pyhf/_generated/pyhf.pdf.Model.html

    Returns:
        Float: Two times the negative log-likelihood, :math:`-2\ln L\left(\mu, \boldsymbol{\theta}\right)`
    """
    return -2 * pdf.logpdf(pars, data)


def qmu(mu, data, pdf, init_pars, par_bounds):
    r"""
    The test statistic, :math:`q_{\mu}`, for establishing an upper
    limit on the strength parameter, :math:`\mu`, as defiend in
    Equation (14) in `arXiv:1007.1727`_ .

    .. _`arXiv:1007.1727`: https://arxiv.org/abs/1007.1727

    .. math::
       :nowrap:

       \begin{equation}
          q_{\mu} = \left\{\begin{array}{ll}
          -2\ln\lambda\left(\mu\right), &\hat{\mu} < \mu,\\
          0, & \hat{\mu} > \mu
          \end{array}\right.
        \end{equation}


    Args:
        mu (Number or Tensor): The signal strength parameter
        data (Tensor): The data to be considered
        pdf (|pyhf.pdf.Model|_): The HistFactory statistical model used in the likelihood ratio calculation
        init_pars (Tensor): The initial parameters
        par_bounds(Tensor): The bounds on the paramter values

    .. |pyhf.pdf.Model| replace:: ``pyhf.pdf.Model``
    .. _pyhf.pdf.Model: https://diana-hep.org/pyhf/_generated/pyhf.pdf.Model.html

    Returns:
        Float: The calculated test statistic, :math:`q_{\mu}`
    """
    tensorlib, optimizer = get_backend()
    mubhathat = optimizer.constrained_bestfit(
        loglambdav, mu, data, pdf, init_pars, par_bounds
    )
    muhatbhat = optimizer.unconstrained_bestfit(
        loglambdav, data, pdf, init_pars, par_bounds
    )
    qmu = loglambdav(mubhathat, data, pdf) - loglambdav(muhatbhat, data, pdf)
    qmu = tensorlib.where(muhatbhat[pdf.config.poi_index] > mu, [0], qmu)
    return qmu


def generate_asimov_data(asimov_mu, data, pdf, init_pars, par_bounds):
    _, optimizer = get_backend()
    bestfit_nuisance_asimov = optimizer.constrained_bestfit(
        loglambdav, asimov_mu, data, pdf, init_pars, par_bounds
    )
    return pdf.expected_data(bestfit_nuisance_asimov)


def pvals_from_teststat(sqrtqmu_v, sqrtqmuA_v):
    r"""
    The :math:`p`-values for signal strength :math:`\mu` and Asimov strength :math:`\mu'` as defined in Equations (59) and (57) of `arXiv:1007.1727`_

    .. _`arXiv:1007.1727`: https://arxiv.org/abs/1007.1727

    .. math::

        p_{\mu} = 1-F\left(q_{\mu}\middle|\mu'\right) = 1- \Phi\left(q_{\mu} - \frac{\left(\mu-\mu'\right)}{\sigma}\right)

    with Equation (29)

    .. math::

        \frac{(\mu-\mu')}{\sigma} = \sqrt{\Lambda}= \sqrt{q_{\mu,A}}

    given the observed test statistics :math:`q_{\mu}` and :math:`q_{\mu,A}`.

    Args:
        sqrtqmu_v (Number or Tensor): The root of the calculated test statistic, :math:`\sqrt{q_{\mu}}`
        sqrtqmuA_v (Number or Tensor): The root of the calculated test statistic given the Asimov data, :math:`\sqrt{q_{\mu,A}}`

    Returns:
        Tuple of Floats: The :math:`p`-values for the signal + background, background only, and signal only hypotheses respectivley
    """
    tensorlib, _ = get_backend()
    CLsb = 1 - tensorlib.normal_cdf(sqrtqmu_v)
    CLb = 1 - tensorlib.normal_cdf(sqrtqmu_v - sqrtqmuA_v)
    CLs = CLsb / CLb
    return CLsb, CLb, CLs


def hypotest(poi_test, data, pdf, init_pars=None, par_bounds=None, **kwargs):
    r"""
    Computes :math:`p`-values and test statistics for a single value of the parameter of interest

    Args:
        poi_test (Number or Tensor): The value of the parameter of interest (POI)
        data (Number or Tensor): The root of the calculated test statistic given the Asimov data, :math:`\sqrt{q_{\mu,A}}`
        pdf (|pyhf.pdf.Model|_): The HistFactory statistical model
        init_pars (Array or Tensor): The initial parameter values to be used for minimization
        par_bounds (Array or Tensor): The parameter value bounds to be used for minimization

    .. |pyhf.pdf.Model| replace:: ``pyhf.pdf.Model``
    .. _pyhf.pdf.Model: https://diana-hep.org/pyhf/_generated/pyhf.pdf.Model.html

    Keyword Args:
        return_tail_probs (bool): Bool for returning :math:`\mathrm{CL}_{s+b}` and :math:`\mathrm{CL}_{b}`
        return_expected (bool): Bool for returning :math:`\mathrm{CL}_{\mathrm{exp}}`
        return_expected_set (bool): Bool for returning the :math:`(-2,-1,0,1,2)\sigma` :math:`\mathrm{CL}_{\mathrm{exp}}` --- the "Brazil band"
        return_test_statistics (bool): Bool for returning :math:`q_{\mu}` and :math:`q_{\mu,A}`

    Returns:
        Tuple of Floats and lists of Floats:

            - :math:`\mathrm{CL}_{s}`: The :math:`p`-value compared to the given threshold :math:`\alpha`, typically taken to be :math:`0.05`, defined in `arXiv:1007.1727`_ as

            .. _`arXiv:1007.1727`: https://arxiv.org/abs/1007.1727

            .. math::

                \mathrm{CL}_{s} = \frac{\mathrm{CL}_{s+b}}{\mathrm{CL}_{b}} = \frac{p_{s+b}}{1-p_{b}}

            to protect against excluding signal models in which there is little sensitivity. In the case that :math:`\mathrm{CL}_{s} \leq \alpha` the given signal model is excluded.

            - :math:`\left[\mathrm{CL}_{s+b}, \mathrm{CL}_{b}\right]`: The signal + background :math:`p`-value and 1 minus the background only :math:`p`-value as defined in Equations (75) and (76) of `arXiv:1007.1727`_

            .. math::

                \mathrm{CL}_{s+b} = p_{s+b} = \int\limits_{q_{\mathrm{obs}}}^{\infty} f\left(q\,\middle|s+b\right)\,dq = 1 - \Phi\left(\frac{q_{\mathrm{obs}} + 1/\sigma_{s+b}^{2}}{2/\sigma_{s+b}}\right)

            .. math::

                \mathrm{CL}_{b} = 1- p_{b} = 1 - \int\limits_{-\infty}^{q_{\mathrm{obs}}} f\left(q\,\middle|b\right)\,dq = 1 - \Phi\left(\frac{q_{\mathrm{obs}} - 1/\sigma_{b}^{2}}{2/\sigma_{b}}\right)

            with Equations (73) and (74) for the mean

            .. math::

                E\left[q\right] = \frac{1 - 2\mu}{\sigma^{2}}

            and variance

            .. math::

                V\left[q\right] = \frac{4}{\sigma^{2}}

            of the test statistic :math:`q` under the background only and and signal + background hypotheses. Only returned when ``return_tail_probs`` is ``True``.

            - :math:`\mathrm{CL}_{s,\mathrm{exp}}`: The expected :math:`\mathrm{CL}_{s}` value corresponding to the test statistic under the background only hypothesis :math:`\left(\mu=0\right)`. Only returned when ``return_expected`` is ``True``.

            - :math:`\mathrm{CL}_{s,\mathrm{exp}}` band: The set of expected :math:`\mathrm{CL}_{s}` values corresponding to the median significance of variations of the signal strength from the background only hypothesis :math:`\left(\mu=0\right)` at :math:`(-2,-1,0,1,2)\sigma`. That is, the :math:`p`-values that satisfy Equation (89) of `arXiv:1007.1727`_

            .. math::

                \mathrm{band}_{N\sigma} = \mu' + \sigma\,\Phi^{-1}\left(1-\alpha\right) \pm N\sigma

            for :math:`\mu'=0` and :math:`N \in \left\{-2, -1, 0, 1, 2\right\}`. These values define the boundaries of an uncertainty band sometimes referred to as the "Brazil band". Only returned when ``return_expected_set`` is ``True``.

            - :math:`\left[q_{\mu}, q_{\mu,A}\right]`: The test statistics for the observed and Asimov datasets respectively. Only returned when ``return_test_statistics`` is ``True``.
    """

    init_pars = init_pars or pdf.config.suggested_init()
    par_bounds = par_bounds or pdf.config.suggested_bounds()
    tensorlib, _ = get_backend()

    asimov_mu = 0.0
    asimov_data = generate_asimov_data(asimov_mu, data, pdf, init_pars, par_bounds)

    qmu_v = tensorlib.clip(qmu(poi_test, data, pdf, init_pars, par_bounds), 0, max=None)
    sqrtqmu_v = tensorlib.sqrt(qmu_v)

    qmuA_v = tensorlib.clip(
        qmu(poi_test, asimov_data, pdf, init_pars, par_bounds), 0, max=None
    )
    sqrtqmuA_v = tensorlib.sqrt(qmuA_v)

    CLsb, CLb, CLs = pvals_from_teststat(sqrtqmu_v, sqrtqmuA_v)

    _returns = [CLs]
    if kwargs.get('return_tail_probs'):
        _returns.append([CLsb, CLb])
    if kwargs.get('return_expected_set'):
        CLs_exp = []
        for n_sigma in [-2, -1, 0, 1, 2]:
            sqrtqmu_v_sigma = sqrtqmuA_v - n_sigma
            CLs_exp.append(pvals_from_teststat(sqrtqmu_v_sigma, sqrtqmuA_v)[-1])
        CLs_exp = tensorlib.astensor(CLs_exp)
        if kwargs.get('return_expected'):
            _returns.append(CLs_exp[2])
        _returns.append(CLs_exp)
    elif kwargs.get('return_expected'):
        _returns.append(pvals_from_teststat(sqrtqmuA_v, sqrtqmuA_v)[-1])
    if kwargs.get('return_test_statistics'):
        _returns.append([qmu_v, qmuA_v])

    # Enforce a consistent return type of the observed CLs
    return tuple(_returns) if len(_returns) > 1 else _returns[0]
