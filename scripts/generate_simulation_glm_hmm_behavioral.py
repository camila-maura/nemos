"""
Simulate a single unit GLM with hidden Markov states.

This script generates synthetic behavioral data where each animal's activity
depends on both external features and a latent state that evolves over time
according to a Markov process.

Based on Venditto's script for behavioral simulation from
https://github.com/Brody-Lab/venditto_glm-hmm/blob/main/glmhmm_example_fit.m
and on the behavioral variables considered in Ashwood et al. (2020).
"""

# IMPORTS
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import nemos as nmo
from nemos.glm_hmm.algorithm_configs import (
    prepare_estep_log_likelihood,
    prepare_mstep_nll_objective_param,
)
from nemos.glm_hmm.expectation_maximization import (
    em_glm_hmm,
    forward_pass,
    hmm_negative_log_likelihood,
    prepare_likelihood_func,
    compute_rate_per_state
)
seed = 0  # Random seed for reproducibility
np.random.seed(seed)
jax.config.update("jax_enable_x64", True)

# SESSION PARAMETERS
n_sess = 20                                     # Number of sessions to simulate
n_trials_per_sess = 100                         # Number of trials in a session
n_timepoints = n_sess * n_trials_per_sess       # Total number of timepoints
new_sess = np.zeros(n_timepoints, dtype=int)    # Indicator for session starts
new_sess[::n_trials_per_sess] = 1               # Set 1 at the beginning of each session
# GLM-HMM parameters
n_states = 3                                    # Number of latent states

# We will consider a design matrix with the following behavioral variables
X_labels = ["bias", "stimulus"]

n_features = len(X_labels) - 1  # Number of features in design matrix
                                # Substracting one because Nemos does not consider intercept as feature

# Projection weights, using parameters from Ashwood et al. (2020)
true_projection_weights = np.array(
    [6, 2, 2], dtype=float  # Stimulus
)
true_intercept = np.array(
    [1, -3, 3],dtype=float  # Bias - Intercept
)

# Initial state probabilities
# Couldnt find exact numbers from fast check in Ashwood et al. - revise again
true_initial_prob = jnp.array([0.95, 0.025, 0.025])

# Transition matrix
true_transition_prob = np.array(
    [[0.98, 0.01, 0.01], [0.05, 0.92, 0.03], [0.03, 0.03, 0.94]]
)
# Stimuli shown to the "mice"
stim_vals = [
    -1,
    -0.5,
    -0.25,
    -0.125,
    -0.0625,
    0,
    0.0625,
    0.125,
    0.25,
    0.5,
    1,
]

# Generate random sequence of stimuli for simulation
X = np.random.choice(stim_vals, n_timepoints)
X = X.reshape(n_timepoints, n_features)

# Simulation
# Initialize storage
true_latent_states = np.zeros((n_timepoints, n_states), dtype=int)  # n_timepoints, n_states
choice_probas = np.zeros((n_timepoints,))
true_choices = np.zeros((n_timepoints,))

# Simulate states and observations
initial_state = np.random.choice(n_states,
                                 p=true_initial_prob)
true_latent_states[0, initial_state] = 1                            # latent at timepoint 0

# Initialize GLM
glm = nmo.glm.GLM(observation_model="Bernoulli")

# Set initial weights and simulate first timepoint
glm.coef_ = true_projection_weights[initial_state].reshape(n_features,)
glm.intercept_ = true_intercept[initial_state].reshape(1,)

# Set key for replication
key = jax.random.PRNGKey(seed)
print(glm.coef_)
print(X.shape)

# Simulate first count and proba
res_sim = glm.simulate(key, X[:1])
print(res_sim)
true_choices[0] = res_sim[0][0]
choice_probas[0] = res_sim[1][0]

# Simulate remaining timepoints
print("Simulating data...")
for t in range(1, n_timepoints):
    # Sample next state
    key, subkey = jax.random.split(key)
    prev_state_vec = true_latent_states[t - 1]
    transition_probs = true_transition_prob.T @ prev_state_vec
    next_state = jax.random.choice(subkey, jnp.arange(n_states), p=transition_probs)
    true_latent_states[t, next_state] = 1
    
    # Update weights and simulate
    glm.coef_ = true_projection_weights[next_state].reshape(n_features,)
    glm.intercept_ = true_intercept[next_state].reshape(1,)
    key, subkey = jax.random.split(key) # Is this necessary again?
    res = glm.simulate(subkey, X[t : t + 1])

    true_choices[t] = res[0][0]
    choice_probas[t] = res[1][0]
    print(f"Simulated timepoint {t+1}/{n_timepoints}", end="\r")
print("\nSimulation complete.")

# Utils
def plot_glm_weights(
    n_features,
    n_states,
    true_projection_weights,
    learned_intercept,
    learned_coef,
    initialization_setting,
):
    ## Plot
    fig = plt.figure(figsize=(4, 3), dpi=80, facecolor="w", edgecolor="k")
    cols = ["#ff7f00", "#4daf4a", "#377eb8"]
    recovered_weights = np.zeros_like(true_projection_weights)

    recovered_weights[:1] = learned_intercept
    recovered_weights[1:] = learned_coef
    
    print(true_projection_weights[:,1])

    for k in range(n_states):
        if k == 0:
            plt.plot(
                range(n_features),
                true_projection_weights[:, k],
                marker="o",
                color=cols[k],
                linestyle="-",
                lw=1.5,
                label="generative",
            )
            plt.plot(
                range(n_features),
                recovered_weights[:, k],
                color=cols[k],
                lw=1.5,
                label="recovered",
                linestyle="--",
            )
        else:
            plt.plot(
                range(n_features),
                true_projection_weights[:, k],
                marker="o",
                color=cols[k],
                linestyle="-",
                lw=1.5,
                label="",
            )

            plt.plot(
                range(n_features),
                recovered_weights[:, k],
                color=cols[k],
                lw=1.5,
                label="",
                linestyle="--",
            )
    plt.yticks(fontsize=10)
    plt.ylabel("GLM weight", fontsize=15)
    plt.xlabel("covariate", fontsize=15)
    plt.xticks([0, 1], X_labels, fontsize=12, rotation=45)
    plt.axhline(y=0, color="k", alpha=0.5, ls="--")
    plt.legend()
    plt.title(f"Weight recovery - {initialization_setting}", fontsize=15)
    plt.show()
    return None

# Store likelihoods
log_likelihoods = {}

###
# 1. FIT SIMULATED DATA WITH A TINY DEVIATION FROM TRUE PARAMETERS
###
initialization_setting = "small_perturbation"
print(f"Fitting data with {initialization_setting} initialization...")

# add small noise to initial prob
initial_prob_initial_guess = true_initial_prob + np.random.uniform(0, 0.01)
initial_prob_initial_guess /= initial_prob_initial_guess.sum()  # Normalize
print("--Initial probability guess: \n", initial_prob_initial_guess)
print("--Check it sums to 1", initial_prob_initial_guess.sum())

# add small noise to projection weights
projection_weights_initial_guess = (
    true_projection_weights + np.random.randn(*true_projection_weights.shape) * 1e-8
)

intercept_initial_guess = (
    true_intercept + np.random.randn(*true_intercept.shape) * 1e-8
)
print("Initial projection weights guess: \n", projection_weights_initial_guess)
print(projection_weights_initial_guess.shape)

# High proba in diagonal - low elsewhere
transition_prob_initial_guess = np.ones(true_transition_prob.shape) * 0.05
transition_prob_initial_guess[np.diag_indices(true_transition_prob.shape[1])] = 0.9
print("Initial transition probability guess \n", transition_prob_initial_guess)
print("--Check it sums to 1", transition_prob_initial_guess.sum(axis=1))

# FIRST FIT : SAME AS TRUE
# SETUP NEMOS
# Observation model
obs = nmo.observation_models.BernoulliObservations()
inverse_link_function = obs.default_inverse_link_function
# Likelihood function & wrapper
log_likelihood_func, negative_log_likelihood_func = prepare_likelihood_func(
    False, obs.log_likelihood, obs._negative_log_likelihood
)
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
# glm object
glm = nmo.glm.GLM(observation_model=obs, solver_name="LBFGS", solver_kwargs={"tol":1e-12, "maxiter": 5000})
glm.instantiate_solver(partial_hmm_negative_log_likelihood)

# reshape params
true_projection_weights = true_projection_weights.reshape(n_features, n_states)
true_intercept = true_intercept.reshape(n_states)

# fit
(
        posteriors,
        joint_posterior,
        log_initial_prob,
        log_transition_matrix,
        nemos_glm_params,
        state,
    ) = em_glm_hmm(
            X,
            (true_choices).astype(float),
            initial_prob=initial_prob_initial_guess,
            transition_prob=transition_prob_initial_guess,
            glm_params=(projection_weights_initial_guess.reshape(n_features, n_states),
                        intercept_initial_guess.reshape(n_states)),
            inverse_link_function=obs.default_inverse_link_function,
            likelihood_func=log_likelihood_func,
            m_step_fn_glm_params=glm._solver_run,
            is_new_session=new_sess,
            maxiter=5000,
            tol=1e-12,
        )

print(nemos_glm_params)

nemos_initial_prob = log_initial_prob
nemos_transition_prob = log_transition_matrix
predicted_rate_given_state = compute_rate_per_state(
    X, nemos_glm_params, inverse_link_function
)

log_like_nemos_stack = log_likelihood_func(1 - true_choices, predicted_rate_given_state)
log_alphas, log_normalization = forward_pass(
    log_initial_prob,
    log_transition_matrix,
    log_like_nemos_stack,
    new_sess
)
# Stack array for plotting
stacked_arr = np.stack([true_intercept, true_projection_weights.reshape(n_states)])

plot_glm_weights(
        n_features+1,
        n_states,
        stacked_arr,
        nemos_glm_params[1],
        nemos_glm_params[0],
        initialization_setting,
    )

# Likelihood
_,log_normalization = forward_pass(
    log_initial_prob,
    log_transition_matrix,
    log_like_nemos_stack,
    new_sess
)
print("\n Fitting complete.")
###
# 2. FIT SIMULATED DATA WITH ABSOLUTELY RANDOM INITIALIZATION
###
initialization_setting = "random_init"
print(f"Fitting data with {initialization_setting} initialization...")
# PARAMETERS
# Sampling from dirichlet [1,1,1] - all possible triplets that sum to 1 are equally likely
initial_prob_initial_guess = np.random.dirichlet(np.ones(n_states))
print("--Initial probability guess: \n ", initial_prob_initial_guess)
print("--Check it sums to 1", initial_prob_initial_guess.sum())
# Normalization is not necessary anymore since dirichlet already sums to 1

# Random projection weights
projection_weights_initial_guess = true_projection_weights + np.random.randn(
    *true_projection_weights.shape
)
print("Initial projection weights guess: \n", projection_weights_initial_guess)

# Random transition matrix
transition_prob_initial_guess = np.random.dirichlet(np.ones(n_states), size=3)
print("Initial transition probability guess \n", transition_prob_initial_guess)
print("--Check it sums to 1", transition_prob_initial_guess.sum(axis=1))


print("\n Fitting complete.")


# This is no better than starting from a slightly perturbed version of the true parameters
def compare_likelihoods(log_likelihoods):
    print(f"Log likelihoods from different initializations: \n {log_likelihoods}")
    return None


###
# 3. FIT SIMULATED DATA WITH "TILTED" INITIALIZATION
# As in Iris paper
###

###
# 4. FIT SIMULATED DATA WITH K-MEANS ALGORITHM

###
###
# 5. COMPARE LIKELIHOODS
###
compare_likelihoods(log_likelihoods)
# Okay absolutely random init in this case seems to work better than slightly perturbed true params, and the difference is pretty small...
# - Maybe the simulation is too "easy"? Also the correlation with the third state is pretty low in both cases...
# - Maybe there's a bug somewhere in the simulation?
# - I am not getting the right likelihood? -> According to documentation, I am getting the total log likelihood of the sequence given the model parameters. Cant quite figure out what is off but for sure the likelihood result is different from the ssm implementation.
# - I also should calculate the true log likelihood of the data given the true parameters to have a better idea of how well we are doing.
# - Also should plot the most likely sequence of states vs. true states

# Move to implementation of K means initialization and then come back to these issues.

# %%

# Notes for notebook
# NOTE: add an admonition on the dirichlet distribution on the notebook
