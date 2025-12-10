
import numpy as np
import numpy.random as npr
import jax
import jax.numpy as jnp

import ssm
from ssm.messages import backward_pass
from ssm.util import find_permutation
from ssm.hmm import hmm_normalizer

import nemos as nmo
from nemos.glm_hmm.expectation_maximization import (
    forward_pass,
    compute_rate_per_state,
    prepare_likelihood_func,
    hmm_negative_log_likelihood,
    em_glm_hmm
)
import matplotlib.pyplot as plt


def plot_weights_recovered(true_weights, rec_weights, package: str, n_states: int, input_dimensionality: int):
    # Weight recovery plot
    fig = plt.figure(figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
    cols = ['#ff7f00', '#4daf4a', '#377eb8']
    for k in range(n_states):
        if k == 0:
            plt.plot(range(input_dimensionality), true_weights[k][0], marker='o',
                     color=cols[k], linestyle='-',
                     lw=1.5, label="generative")
            plt.plot(range(input_dimensionality), rec_weights[k][0], color=cols[k],
                     lw=1.5, label="recovered", linestyle='--')
        else:
            plt.plot(range(input_dimensionality), true_weights[k][0], marker='o',
                     color=cols[k], linestyle='-',
                     lw=1.5, label="")
            plt.plot(range(input_dimensionality), rec_weights[k][0], color=cols[k],
                     lw=1.5, label='', linestyle='--')
    plt.yticks(fontsize=10)
    plt.ylabel("GLM weight", fontsize=15)
    plt.xlabel("covariate", fontsize=15)
    plt.xticks([0, 1], ['stimulus', 'bias'], fontsize=12, rotation=45)
    plt.axhline(y=0, color="k", alpha=0.5, ls="--")
    plt.legend()
    plt.title(f"Weight recovery {package}", fontsize=15)
    plt.tight_layout()
    return fig


# Set random seed for reproducibility
npr.seed(0)

# Enable 64-bit precision for numerical accuracy
jax.config.update("jax_enable_x64", True)


# =============================================================================
# 1. Set up GLM-HMM parameters
# =============================================================================

num_states = 3        # Number of discrete latent states
obs_dim = 1           # Number of observed dimensions
num_categories = 2    # Number of categories for output (binary choice)
input_dim = 2         # Input dimensions (stimulus + bias)

# Generative GLM weights: shape (num_states, obs_dim, input_dim)
# Each state has different weights for [stimulus, bias]
gen_weights = np.array([
    [[6, 1]],      # State 1: strong positive stimulus effect
    [[2, -3]],     # State 2: weak stimulus, negative bias
    [[2, 3]]       # State 3: weak stimulus, positive bias
])

# Generative transition matrix (in log space)
# High diagonal = states are sticky (tend to persist)
gen_log_trans_mat = np.log(np.array([
    [[0.98, 0.01, 0.01],   # From state 1
     [0.05, 0.92, 0.03],   # From state 2
     [0.03, 0.03, 0.94]]   # From state 3
]))


# =============================================================================
# 2. Create true GLM-HMM model using SSM
# =============================================================================

true_glmhmm = ssm.HMM(
    num_states,
    obs_dim,
    input_dim,
    observations="input_driven_obs",
    observation_kwargs=dict(C=num_categories),
    transitions="standard"
)
true_glmhmm.observations.params = gen_weights
true_glmhmm.transitions.params = gen_log_trans_mat


# =============================================================================
# 3. Generate synthetic data
# =============================================================================

num_sess = 20                  # Number of sessions
num_trials_per_sess = 100      # Trials per session

# Generate stimulus inputs
stim_vals = [-1, -0.5, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 0.5, 1]
inpts = np.ones((num_sess, num_trials_per_sess, input_dim))
inpts[:, :, 0] = np.random.choice(stim_vals, (num_sess, num_trials_per_sess))
inpts = list(inpts)

# Generate latent states and choices for each session
true_latents, true_choices = [], []
for sess in range(num_sess):
    true_z, true_y = true_glmhmm.sample(num_trials_per_sess, input=inpts[sess])
    true_latents.append(true_z)
    true_choices.append(true_y)

# Calculate true log-likelihood
true_ll = true_glmhmm.log_probability(true_choices, inputs=inpts)


# =============================================================================
# 5. Initialize a new GLM-HMM for comparison
# =============================================================================

new_glmhmm = ssm.HMM(
    num_states,
    obs_dim,
    input_dim,
    observations="input_driven_obs",
    observation_kwargs=dict(C=num_categories),
    transitions="standard"
)

# Initialize with data
new_glmhmm.initialize(datas=true_choices, inputs=inpts)
ssm_ll_init = new_glmhmm.log_likelihood(true_choices, inputs=inpts)

# nemos input and init params
X = np.concatenate([inp[:, :1] for inp in inpts])
y = np.concatenate([c[:, 0] for c in true_choices])
masks = [np.ones_like(tc, dtype=bool) for tc in true_choices]
pi0 = new_glmhmm.init_state_distn.initial_state_distn
Ps = new_glmhmm.transitions.transition_matrices(true_choices[0], inpts[0], masks[0], None)
# Extract GLM parameters from SSM
weights = new_glmhmm.observations.Wk[:, 0].T
coef = weights[:1, :]      # Stimulus coefficients
intercept = weights[1, :]  # Bias terms

# Set up NeMoS observation model
obs = nmo.observation_models.BernoulliObservations()
inverse_link_function = obs.default_inverse_link_function

is_new_session = []
for i in range(len(true_choices)):
    new_sess = np.zeros_like(true_choices[i][:, 0], dtype=bool)
    new_sess[0] = True  # Mark first trial of each session
    is_new_session.append(new_sess)
is_new_session = np.concatenate(is_new_session)
predicted_rate_given_state_init = compute_rate_per_state(
    X, (coef, intercept), inverse_link_function
)
glm_params = (coef, intercept)
log_likelihood_func, negative_log_likelihood_func = prepare_likelihood_func(
    False, obs.log_likelihood, obs._negative_log_likelihood
)

log_like_nemos_stack_init = log_likelihood_func(1 - y, predicted_rate_given_state_init)
log_alphas_init, log_normalization_init = forward_pass(
    np.log(pi0),
    np.log(Ps[0]),
    log_like_nemos_stack_init,
    is_new_session
)
nemos_ll_init = np.sum(log_normalization_init)
print("nemos init ll", nemos_ll_init)
print("ssm ll init", ssm_ll_init)



def partial_hmm_negative_log_likelihood(
        weights, design_matrix, observations, posterior_prob
):
    return hmm_negative_log_likelihood(
        weights,
        X=design_matrix,
        y=observations,
        posteriors=posterior_prob,
        inverse_link_function=obs.default_inverse_link_function,
        negative_log_likelihood_func=negative_log_likelihood_func,
    )


glm = nmo.glm.GLM(observation_model=obs, solver_name="LBFGS", solver_kwargs={"tol":1e-12, "maxiter": 5000})

glm.instantiate_solver(partial_hmm_negative_log_likelihood)


(
        posteriors,
        joint_posterior,
        log_initial_prob,
        log_transition_matrix,
        nemos_glm_params,
        state,
    ) = em_glm_hmm(
            X,
            (1-y).astype(float),
            initial_prob=pi0,
            transition_prob=Ps[0],
            glm_params=(coef, intercept),
            inverse_link_function=obs.default_inverse_link_function,
            likelihood_func=log_likelihood_func,
            m_step_fn_glm_params=glm._solver_run,
            is_new_session=is_new_session,
            maxiter=1000,
            tol=1e-10,
        )

nemos_initial_prob = np.exp(log_initial_prob)
nemos_transition_prob = np.exp(log_transition_matrix)
predicted_rate_given_state = compute_rate_per_state(
    X, nemos_glm_params, inverse_link_function
)

log_like_nemos_stack = log_likelihood_func(1 - y, predicted_rate_given_state)


log_alphas, log_normalization = forward_pass(
    log_initial_prob,
    log_transition_matrix,
    log_like_nemos_stack,
    is_new_session
)



nemos_fit_ll = jnp.sum(log_normalization)
ssm_fit_ll = new_glmhmm.fit(true_choices, inputs=inpts, method="em", num_iters=1000, tolerance=10**-10)

print("SSM fit ll:", ssm_fit_ll[-1])
print("nemos fit ll:", nemos_fit_ll)


# Permute states
new_glmhmm.permute(find_permutation(true_latents[0], new_glmhmm.most_likely_states(true_choices[0], input=inpts[0])))
recovered_weights = new_glmhmm.observations.params

fig = plot_weights_recovered(recovered_weights, gen_weights, "SSM", num_states, input_dim)
fig.show()

stack_nemos_weights = np.c_[nemos_glm_params[0][0], nemos_glm_params[1]]
stack_nemos_weights = np.expand_dims(stack_nemos_weights, 1)
fig = plot_weights_recovered(stack_nemos_weights, gen_weights, "NEMOS", num_states, input_dim)

fig.show()