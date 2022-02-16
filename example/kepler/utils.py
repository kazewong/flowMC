import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

@jax.custom_jvp
def kepler(M, ecc):
    """
    Given a mean anomaly 'M' and eccentricity 'ecc', solve for the sin and cos
    of the true anomaly 'f'.
    """
    # Wrap into the right range
    M = M % (2 * jnp.pi)

    # We can restrict to the range [0, pi)
    high = M > jnp.pi
    M = jnp.where(high, 2 * jnp.pi - M, M)

    # Solve
    ome = 1 - ecc
    E = starter(M, ecc, ome)
    E = refine(M, ecc, ome, E)

    # Re-wrap back into the full range
    E = jnp.where(high, 2 * jnp.pi - E, E)

    # Convert to true anomaly; tan(0.5 * f)
    tan_half_f = jnp.sqrt((1 + ecc) / (1 - ecc)) * jnp.tan(0.5 * E)
    tan2_half_f = jnp.square(tan_half_f)

    # Then we compute sin(f) and cos(f) using:
    #  sin(f) = 2*tan(0.5*f)/(1 + tan(0.5*f)^2), and
    #  cos(f) = (1 - tan(0.5*f)^2)/(1 + tan(0.5*f)^2)
    denom = 1 / (1 + tan2_half_f)
    sinf = 2 * tan_half_f * denom
    cosf = (1 - tan2_half_f) * denom

    return sinf, cosf


# This defines the differentation rule
@kepler.defjvp
def kepler_jvp(primals, tangents):
    M, e = primals
    dM, de = tangents
    sinf, cosf = kepler(M, e)

    # Pre-compute some things
    ecosf = e * cosf
    ome2 = 1 - e ** 2

    # Propagate the derivatives
    df = 0.0
    if type(dM) is not jax.interpreters.ad.Zero:
        df += dM * (1 + ecosf) ** 2 / ome2 ** 1.5
    if type(de) is not jax.interpreters.ad.Zero:
        df += de * (2 + ecosf) * sinf / ome2

    return (sinf, cosf), (cosf * df, -sinf * df)


# The following two functions are helpers for the Kepler solver
def starter(M, ecc, ome):
    M2 = jnp.square(M)
    alpha = 3 * jnp.pi / (jnp.pi - 6 / jnp.pi)
    alpha += 1.6 / (jnp.pi - 6 / jnp.pi) * (jnp.pi - M) / (1 + ecc)
    d = 3 * ome + alpha * ecc
    alphad = alpha * d
    r = (3 * alphad * (d - ome) + M2) * M
    q = 2 * alphad * ome - M2
    q2 = jnp.square(q)
    w = (jnp.abs(r) + jnp.sqrt(q2 * q + r * r)) ** (2.0 / 3)
    return (2 * r * w / (jnp.square(w) + w * q + q2) + M) / d


def refine(M, ecc, ome, E):
    sE = E - jnp.sin(E)
    cE = 1 - jnp.cos(E)

    f_0 = ecc * sE + E * ome - M
    f_1 = ecc * cE + ome
    f_2 = ecc * (E - sE)
    f_3 = 1 - f_1
    d_3 = -f_0 / (f_1 - 0.5 * f_0 * f_2 / f_1)
    d_4 = -f_0 / (f_1 + 0.5 * d_3 * f_2 + (d_3 * d_3) * f_3 / 6)
    d_42 = d_4 * d_4
    dE = -f_0 / (
        f_1 + 0.5 * d_4 * f_2 + d_4 * d_4 * f_3 / 6 - d_42 * d_4 * f_2 / 24
    )

    return E + dE


def rv_model(kepler_params, t):
    v0, log_s2, log_period, log_k, sin_phi, cos_phi, ecc, sin_w, cos_w = kepler_params
    phi = jnp.arctan2(sin_phi, cos_phi)
    sin_f, cos_f = kepler(2 * np.pi * t * jnp.exp(-log_period) + phi, ecc)
    return v0 + jnp.exp(log_k) * (cos_w * cos_f - sin_w * sin_f + ecc * cos_w)


def get_kepler_params_and_log_jac(kepler_params):
    v0, log_s2, log_period, log_k, sin_phi_, cos_phi_, ecc_, sin_w_, cos_w_ = kepler_params
    
    # Log odds transform for eccentricity, since it needs to be in the range [0, 1)
    ecc = 1.0 / (1.0 + jnp.exp(-ecc_))
    log_jac = -2 * jnp.log(jnp.cosh(0.5 * ecc_))
    # -2 * jnp.log(2 * jnp.cosh(0.5 * ecc_))? factor 2? to keep exact logjac with NF

    # Normalize the unit vector parameterizations of the angles
    phi_norm = sin_phi_ ** 2 + cos_phi_ ** 2
    log_jac += -0.5 * phi_norm
    phi_norm = jnp.sqrt(phi_norm)

    w_norm = sin_w_ ** 2 + cos_w_ ** 2
    log_jac += -0.5 * w_norm
    w_norm = jnp.sqrt(w_norm)

    return (
        (v0, log_s2, log_period, log_k, sin_phi_ / phi_norm, cos_phi_ / phi_norm, ecc, sin_w_ / w_norm, cos_w_ / w_norm),
        log_jac
    )


def log_likelihood(kepler_params, t, rv_err, rv_obs):
    params, log_jac = get_kepler_params_and_log_jac(kepler_params)
    v = rv_model(params, t)
    log_s2 = params[1]
    sigma2 = rv_err ** 2 + jnp.exp(2 * log_s2)
    loglike = -0.5 * jnp.square(rv_obs - v) / sigma2 - 0.5 * jnp.log(sigma2)
    return log_jac + jnp.sum(loglike)


## we also need priors
def log_prior(kepler_params,
              ecc_alpha=2, ecc_beta=2,
              log_k_mean=1, log_k_var=1,
              v0_mean=0, v0_var=1,
              log_period_mean=1, log_period_var=1,
              log_s2_mean=1, log_s2_var=1e-2,
              ):
    v0, log_s2, log_period, log_k, sin_phi_, cos_phi_, ecc_, sin_w_, cos_w_ = kepler_params

    logp_phis_ = - 0.5 * (sin_phi_ ** 2 + cos_phi_ ** 2)
    logp_ws_ = - 0.5 * (sin_w_ ** 2 + cos_w_ ** 2)

    ecc = 1.0 / (1.0 + jnp.exp(- ecc_))
    logp_ecc = -2 * jnp.log(jnp.cosh(0.5 * ecc_)) 
    logp_ecc += (ecc_alpha - 1) * jnp.log(ecc) 
    logp_ecc += (ecc_beta - 1) * jnp.log(1 - ecc)

    logp_log_k = - 0.5 * (log_k - log_k_mean) ** 5 / log_k_var
    logp_v0 = - 0.5 * (v0 - v0_mean) ** 5 / v0_var
    logp_log_period = - 0.5 * (log_period - log_period_mean) ** 5 / log_period_var
    logp_log_s2 = - 0.5 * (log_s2 - log_s2_mean) ** 5 / log_s2_var

    return logp_phis_ + logp_ws_ + logp_ecc + logp_log_k + logp_v0 + logp_log_period + logp_log_s2


def sample_prior(n_samples, ecc_alpha=2, ecc_beta=2,
                 log_k_mean=1, log_k_var=1,
                 v0_mean=0, v0_var=1,
                 log_period_mean=1, log_period_var=1,
                 log_s2_mean=1, log_s2_var=1e-2):
    """
    returns kepler_params
    """
    



# def random_init():
#     """
#     sample according to priors + run optim.
#     v0, log_s2, log_period, log_k, sin_phi_, cos_phi_, ecc_, sin_w_, cos_w_
#     """
#     jnp.array([
#     12.0, np.log(0.5), np.log(14.5), np.log(2.3), 
#     np.sin(1.5), np.cos(1.5), 0.4, np.sin(-0.7), np.cos(-0.7)
#     ])

