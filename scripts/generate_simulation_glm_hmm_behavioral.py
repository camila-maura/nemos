"""
Simulate a single unit GLM with hidden Markov states.

This script generates synthetic behavioral data where each animal's activity
depends on both external features and a latent state that evolves over time
according to a Markov process.

Based on Venditto's script for behavioral simulation from
https://github.com/Brody-Lab/venditto_glm-hmm/blob/main/glmhmm_example_fit.m
and on the behavioral variables considered in Ashwood et al. (2020).
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# %%
###
# IMPORTS
###
import nemos as nmo
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import sys

from nemos.glm_hmm.expectation_maximization import (
    em_glm_hmm,
    forward_backward,
    hmm_negative_log_likelihood,
    prepare_likelihood_func,
)

###
# SESSION PARAMETERS
###

seed = 1                 # Random seed for reproducibility
np.random.seed(seed)

# For 100 sessions > this works fine, but 20 sessions does not yield good recovery
n_sess = 20                 # Number of sessions to simulate
n_trials_per_sess = 100     # Number of trials in a session
n_timepoints = n_sess * n_trials_per_sess # Total number of timepoints
new_sess = np.zeros(n_timepoints, dtype=int) # Indicator for session starts
new_sess[::n_trials_per_sess] = 1            # Set 1 at the beginning of each session

###
# GLM-HMM PARAMETERS
###
n_states = 3  # Number of latent states

# We will consider a design matrix with the
# following behavioral variables
# x = [bias, stimulus, last_choice, win_stay_lose_shift]
X_labels = ["bias", "stimulus"]  # , "last_choice", "win-stay-lose-shift"]

n_features = len(X_labels)  # Number of features in design matrix

# Projection weights, using parameters from Ashwood et al. (2020)
# results for example IBL mouse (Fig. 2)
# NOTE: in the ssh tutorial, they only use Bias and Stimulus. Since computing previous choice and WSLS might take a bit, I will only use those two for now. It is pending to add the other two elements in the projection weights and design matrix.
true_projection_weights = np.array(
    [
        [1, -3, 3],  # Bias - Intercept
        [6, 2, 2],  # Stimulus
        # [0, -.1, .1],          # Previous choice
        # [0, 0, 0],             # Win stay Lose switch
    ]
)

# Initial state probabilities
# Couldnt find exact numbers from fast check in Ashwood et al. - revise again
true_initial_prob = jnp.array([0.95, 0.025, 0.025])

# Transition matrix
true_transition_prob = np.array(
    [[0.98, 0.01, 0.01], [0.05, 0.92, 0.03], [0.03, 0.03, 0.94]]
)

###
# STIMULI
###
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
]  # STimuli shown to the "mice"

# Generate random sequence of stimuli for simulation
X = np.ones((n_timepoints, n_features-1)) # not including bias (thats why features -1)
X = np.random.choice(stim_vals, n_timepoints)

###
# SIMULATION
###

# Initialize storage
true_latent_states = np.zeros((n_timepoints, n_states), dtype=int)
choice_probas = np.zeros((n_timepoints,))
true_choices = np.zeros((n_timepoints,))

# Simulate states and observations
initial_state = np.random.choice(n_states, p=true_initial_prob)
true_latent_states[0, initial_state] = 1

# Initialize GLM
glm = nmo.glm.GLM(observation_model="Bernoulli")

# Set initial weights and simulate first timepoint
#glm.coef_ = true_projection_weights[..., initial_state].reshape(
#    true_projection_weights.shape[0],1)

glm.intercept_ =  true_projection_weights[0,initial_state].reshape(1)
glm.coef_ = true_projection_weights[1,initial_state].reshape(1)

key = jax.random.PRNGKey(seed)

# Simulate first count and proba
res_sim = glm.simulate(key,X[1].reshape(1,1))

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
    # print(next_state)
    # print(true_transition_prob)

    true_latent_states[t, next_state] = 1

    # Update weights and simulate
    glm.intercept_ =  true_projection_weights[0,next_state].reshape(1)
    glm.coef_ = true_projection_weights[1,next_state].reshape(1)
    
    key, subkey = jax.random.split(key)

    res = glm.simulate(subkey, X[t : t + 1].reshape(1,1))

    true_choices[t] = res[0][0]
    choice_probas[t] = res[1][0]
    print(f"Simulated timepoint {t+1}/{n_timepoints}", end="\r")
print("\nSimulation complete.")

print("Computing true likelihood of the data given the true parameters...")

###
# UTILS
###


def plot_glm_weights(
    n_features,
    n_states,
    true_projection_weights,
    learned_coef,
    learned_intercept,
    initialization_setting,
):
    ## Plot
    fig = plt.figure(figsize=(4, 3), dpi=80, facecolor="w", edgecolor="k")
    cols = ["#ff7f00", "#4daf4a", "#377eb8"]
    recovered_weights = np.zeros_like(true_projection_weights)

    recovered_weights[:1] = learned_intercept
    recovered_weights[1:] = learned_coef

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


def fit_glm_hmm_with_em(
    X,
    true_choices,
    true_latent_states,
    true_projection_weights,
    initial_prob_initial_guess,
    transition_prob_initial_guess,
    projection_weights_initial_guess,
    initialization_setting,
):
    is_population_glm = true_projection_weights.ndim > 2


    observation_model = nmo.observation_models.BernoulliObservations()
    likelihood_func, negative_log_likelihood_func = prepare_likelihood_func(
        is_population_glm,
        observation_model.log_likelihood,
        observation_model._negative_log_likelihood,
    )
    inverse_link_function = observation_model.default_inverse_link_function

    def partial_hmm_negative_log_likelihood(
        weights, design_matrix, observations, posterior_prob
    ):
        return hmm_negative_log_likelihood(
            weights,
            X=design_matrix,
            y=observations,
            posteriors=posterior_prob,
            inverse_link_function=inverse_link_function,
            negative_log_likelihood_func=negative_log_likelihood_func,
        )

    # use the BaseRegressor initialize_solver
    regularization = "UnRegularized"
    solver_name = "LBFGS"
    
    glm = nmo.glm.GLM(
        observation_model=observation_model, 
        regularizer=regularization, solver_name=solver_name
    )
    
    print("XXXXXX", projection_weights_initial_guess[1:])
    glm.instantiate_solver(partial_hmm_negative_log_likelihood)
    solver_run = glm._solver_run

    (
        posteriors,
        joint_posterior,
        learned_initial_prob,
        learned_transition,
        (learned_coef, learned_intercept),
        final_state,
    ) = em_glm_hmm(
        X,
        jnp.squeeze(true_choices),
        initial_prob=initial_prob_initial_guess,
        transition_prob=transition_prob_initial_guess,
        glm_params=(
            projection_weights_initial_guess[1:],
            projection_weights_initial_guess[:1],
        ),
        inverse_link_function=inverse_link_function,
        likelihood_func=likelihood_func,
        solver_run=solver_run,
        tol=10**-5,
    )
    (
        _,
        _,
        _,
        log_likelihood_em,
        _,
        _,
    ) = forward_backward(
        X,  
        true_choices,
        learned_initial_prob,
        learned_transition,
        (learned_coef, learned_intercept),
        log_likelihood_func=likelihood_func,
        inverse_link_function=observation_model.default_inverse_link_function,
    )

    # find state mapping
    corr_matrix = np.corrcoef(true_latent_states.T, posteriors.T)[
        : true_latent_states.shape[1], true_latent_states.shape[1] :
    ]
    max_corr = np.max(corr_matrix, axis=1)
    print("\nMAX CORR", max_corr)  # State recovery is quite low let's check why

    plot_glm_weights(
        n_features,
        n_states,
        true_projection_weights,
        learned_coef,
        learned_intercept,
        initialization_setting,
    )
    
    fig = plt.figure(figsize=(5, 2.5), dpi=80, facecolor='w', edgecolor='k')
    #sess_id = 0  # session id; can choose any index between 0 and num_sess-1
    cols = ['#ff7f00', '#4daf4a', '#377eb8']
    print(posteriors.shape)
    for k in range(n_states):
        plt.plot(posteriors[0:100, k], label="Recovered", ls='--',lw=2,
                color=cols[k])
        plt.plot(true_latent_states[0:100, k], label="Generative", ls='-', lw=1,
                color=cols[k])
    plt.ylim((-0.01, 1.01))
    plt.yticks([0, 0.5, 1], fontsize = 10)
    plt.xlabel("trial #", fontsize = 15)
    plt.ylabel("p(state)", fontsize = 15)
    plt.legend()
    plt.show()
    
    return log_likelihood_em


# Store likelihoods
log_likelihoods = {}

###
# Import simulation -> when i import, it does NOT work. 
###
'''
# Saved and imported from ssm notebook
npzfile = np.load("scripts/ssm_input.npz")

# True initial prob, literally copied
true_initial_prob = jnp.array([0.95, 0.025, 0.025]) 

# True transition prob, copied
true_transition_prob = np.array(
    [[0.98, 0.01, 0.01], 
      [0.05, 0.92, 0.03], 
      [0.03, 0.03, 0.94]]
)

# True weights
true_projection_weights = np.array([
    [1, -3, 3],             # Bias - Intercept
    [6, 2, 2],              # Stimulus
])

# Simulation imported from ssm
true_latent_states = npzfile["true_latents"]
true_choices = npzfile["true_choices"]
X = npzfile["inputs"]
print(X)
'''
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
print("Initial projection weights guess: \n", projection_weights_initial_guess)

# High proba in diagonal - low elsewhere
transition_prob_initial_guess = np.ones(true_transition_prob.shape) * 0.05
transition_prob_initial_guess[np.diag_indices(true_transition_prob.shape[1])] = 0.9
print("Initial transition probability guess \n", transition_prob_initial_guess)
print("--Check it sums to 1", transition_prob_initial_guess.sum(axis=1))

log_likelihoods[initialization_setting] = fit_glm_hmm_with_em(
        X.reshape(n_timepoints,n_features-1),
        true_choices,
        true_latent_states,
        true_projection_weights,
        initial_prob_initial_guess,
        transition_prob_initial_guess,
        projection_weights_initial_guess,
        initialization_setting
)
print("\n Fitting complete.")


###
# 2. FIT SIMULATED DATA WITH RANDOM INITIALIZATION
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

log_likelihoods[initialization_setting] = fit_glm_hmm_with_em(
        X.reshape(n_timepoints,n_features-1),
        true_choices,
        true_latent_states,
        true_projection_weights,
        initial_prob_initial_guess,
        transition_prob_initial_guess,
        projection_weights_initial_guess,
        initialization_setting
)
print("\n Fitting complete.")


# This is no better than starting from a slightly perturbed version of the true parameters
def compare_likelihoods(log_likelihoods):
    print(f"Log likelihoods from different initializations: \n {log_likelihoods}")
    return None

###
# 5. COMPARE LIKELIHOODS
###
compare_likelihoods(log_likelihoods)


# Okay absolutely random init in this case seems to work better than slightly perturbed true params, and the difference is pretty small... Although unexpected, the difference in goodness of fit increases with sample size - better initial guesses get better results when the sample size is larger.
# For some reason, ssm implementation seems to work better for a sample size of 20 *100, whilst here the results are awful.
# I want to get the summed likelihood as opposed to mean - pending
# Pending I also should calculate the true log likelihood of the data given the true parameters
# Move to implementation of K means initialization and then come back to these issues.



# Notes for notebook
# NOTE: add an admonition on the dirichlet distribution on the notebook
