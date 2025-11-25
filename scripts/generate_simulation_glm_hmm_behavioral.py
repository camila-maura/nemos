"""
Simulate a single unit GLM with hidden Markov states.

This script generates synthetic behavioral data where each animal's activity
depends on both external features and a latent state that evolves over time
according to a Markov process.

Based on Venditto's script for behavioral simulation from
https://github.com/Brody-Lab/venditto_glm-hmm/blob/main/glmhmm_example_fit.m 
and on the behavioral variables considered in Ashwood et al. (2020).
"""
###
# IMPORTS
###

import nemos as nmo
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from nemos.glm_hmm.expectation_maximization import (
    GLMHMMState,
    backward_pass,
    check_log_likelihood_increment,
    compute_xi,
    em_glm_hmm,
    forward_backward,
    forward_pass,
    hmm_negative_log_likelihood,
    max_sum,
    prepare_likelihood_func,
    run_m_step,
)

### 
# SESSION PARAMETERS
###

seed = 123                  # Random seed for reproducibility

n_sess = 30                 # Number of sessions to simulate
n_trials_per_sess = 100     # Number of trials in a session
n_timepoints = n_sess * n_trials_per_sess # Total number of timepoints
new_sess = np.zeros(n_timepoints, dtype=int) # Indicator for session starts
new_sess[::n_trials_per_sess] = 1            # Set 1 at the beginning of each session

###
# GLM-HMM PARAMETERS
###
n_states = 3                # Number of latent states

# We will consider a design matrix with the
# following behavioral variables
# x = [bias, stimulus, last_choice, win_stay_lose_shift]
X_labels = ["bias", "stimulus"] #, "last_choice", "win-stay-lose-shift"]

n_features = len(X_labels) # Number of features in design matrix

# Projection weights, using parameters from Ashwood et al. (2020) 
# results for example IBL mouse (Fig. 2)
# NOTE: in the ssh tutorial, they only use Bias and Stimulus. Since computing previous choice and WSLS might take a bit, I will only use those two for now. It is pending to add the other two elements in the projection weights and design matrix.
true_projection_weights = np.array([
    [1, -3, 3],             # Bias - Intercept
    [6, 2, 2],              # Stimulus
    #[0, -.1, .1],          # Previous choice
    #[0, 0, 0],             # Win stay Lose switch
])

# Initial state probabilities
# Couldnt find exact numbers from fast check in Ashwood et al. - revise again
true_initial_prob = jnp.array([0.95, 0.025, 0.025]) 

# Transition matrix
true_transition_prob = np.array(
    [[0.98, 0.01, 0.01], 
      [0.05, 0.92, 0.03], 
      [0.03, 0.03, 0.94]]
)

# TODO plot params

###
# STIMULI
###
stim_vals = [-1, -0.5, -0.25, -0.125, 
             -0.0625, 0, 0.0625, 0.125, 0.25, 0.5, 1]   # STimuli shown to the "mice"

# Generate random sequence of stimuli for simulation
X = np.ones((n_timepoints, n_features))
X[:,1] = np.random.choice(stim_vals, n_timepoints)

###
# SIMULATION
###

# Initialize storage
true_latent_states = np.zeros((n_timepoints, n_states), dtype=int)
choice_probas = np.zeros((n_timepoints,))
true_choices = np.zeros((n_timepoints,))

# Simulate states and observations
np.random.seed(seed)
initial_state = np.random.choice(n_states, p=true_initial_prob)
true_latent_states[0, initial_state] = 1

# Initialize GLM
glm = nmo.glm.GLM(observation_model="Bernoulli")
glm.intercept_ = jnp.zeros((1,))

# Set initial weights and simulate first timepoint
glm.coef_ = true_projection_weights[..., initial_state].reshape(
    true_projection_weights.shape[0],1)

# Set key for replication
key = jax.random.PRNGKey(seed)

# Simulate first count and proba
res_sim = glm.simulate(key, X[:1])

true_choices[0] = res_sim[0][0].item()
choice_probas[0] = res_sim[1][0].item()

# Simulate remaining timepoints
for t in range(1, n_timepoints):
    # Sample next state
    key, subkey = jax.random.split(key)

    prev_state_vec = true_latent_states[t - 1]

    print(f"PREVIOUS STATE VECTOR {prev_state_vec} ")
    print(f"TRANSITION PROBAS {true_transition_prob}")
    transition_probs = true_transition_prob.T @ prev_state_vec
    print(f"TRANSITION PROBS {transition_probs}")

    next_state = jax.random.choice(subkey, jnp.arange(n_states), p=transition_probs)
    print(next_state)
    print(true_transition_prob)

    true_latent_states[t, next_state] = 1

    # Update weights and simulate
    glm.coef_ = true_projection_weights[..., next_state]
    key, subkey = jax.random.split(key)
    res = glm.simulate(subkey, X[t : t + 1])

    true_choices[t] = np.array(res[0])
    choice_probas[t] = np.array(res[1])

# Calculate the true log likelihood of the data
# PENDING to calculate

###
# FIT SIMULATED DATA WITH RANDOMIZED INITIAL PARAMETERS
###

# Initial parameters for fitting - first lets try with ABSOLUTELY RANDOM initialization
# This can help convey that EM is really bad with bad initializations.

is_population_glm = true_projection_weights.ndim > 2

observation_model = nmo.observation_models.BernoulliObservations()
likelihood_func, negative_log_likelihood_func = prepare_likelihood_func(
    is_population_glm,
    observation_model.log_likelihood,
    observation_model._negative_log_likelihood,
    is_log=True,
)
inverse_link_function = observation_model.default_inverse_link_function


def partial_hmm_negative_log_likelihood(
            weights, design_matrix, observations, posterior_prob):
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

solver_name = "ProximalGradient" if "Lasso" in regularization else "LBFGS"
glm = nmo.glm.GLM(
    observation_model=observation_model, regularizer=regularization, solver_name=solver_name
)
glm.instantiate_solver(partial_hmm_negative_log_likelihood)
solver_run = glm._solver_run

# add small noise to initial prob & projection weights
initial_prob_initial_guess = true_initial_prob + np.random.uniform(0, 0.1)
initial_prob_initial_guess /= initial_prob_initial_guess.sum() # Normalize

projection_weights_initial_guess = true_projection_weights + np.random.randn(*true_projection_weights.shape)

# High proba in diagonal - low elsewhere
transition_prob_initial_guess =  np.ones(true_transition_prob.shape) * 0.05
transition_prob_initial_guess[np.diag_indices(true_transition_prob.shape[1])] = 0.9

(
    posteriors,
    joint_posterior,
    learned_initial_prob,
    learned_transition,
    (learned_coef, learned_intercept),
    _,
) = em_glm_hmm(
    X[:, 1:],
    jnp.squeeze(true_choices),
    initial_prob=initial_prob_initial_guess,
    transition_prob=transition_prob_initial_guess,
    glm_params=(projection_weights_initial_guess[1:], projection_weights_initial_guess[:1]),
    inverse_link_function=inverse_link_function,
    likelihood_func=likelihood_func,
    solver_run=solver_run,
    tol=10**-10,
)
(
    _,
    _,
    _,
    log_likelihood_em,
    _,
    _,
) = forward_backward(
    X[:, 1:],  # drop intercept
    true_choices,
    learned_initial_prob,
    learned_transition,
    (learned_coef, learned_intercept),
    likelihood_func=likelihood_func,
    inverse_link_function=observation_model.default_inverse_link_function,
)

# find state mapping
corr_matrix = np.corrcoef(true_latent_states.T, posteriors.T)[
    : true_latent_states.shape[1], true_latent_states.shape[1] :
]
max_corr = np.max(corr_matrix, axis=1)
print("\nMAX CORR", max_corr) # State recovery is quite low let's check why

## Plot
fig = plt.figure(figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
cols = ['#ff7f00', '#4daf4a', '#377eb8']
recovered_weights = np.zeros_like(true_projection_weights)

recovered_weights[:1] = learned_intercept
recovered_weights[1:] = learned_coef
print(true_projection_weights)
print(true_projection_weights[:, 0].shape, true_projection_weights[:, 0])
print(true_projection_weights[0][:].shape)

for k in range(n_states):
    if k ==0:
        plt.plot(
            range(n_features),
            true_projection_weights[:,k], 
            marker='o',
            color=cols[k], 
            linestyle='-',
            lw=1.5, 
            label="generative"
        )
        plt.plot(
            range(n_features), 
            recovered_weights[:,k], 
            color=cols[k],
            lw=1.5,  
            label = "recovered", 
            linestyle = '--'
        )
    else:
        plt.plot(
            range(n_features), 
            true_projection_weights[:,k], 
            marker='o',
            color=cols[k], 
            linestyle='-',
            lw=1.5, 
            label=""
        )
        
        plt.plot(
            range(n_features), 
            recovered_weights[:,k], 
            color=cols[k],
            lw=1.5,  
            label = '', 
            linestyle = '--'
        )
plt.yticks(fontsize=10)
plt.ylabel("GLM weight", fontsize=15)
plt.xlabel("covariate", fontsize=15)
plt.xticks([0, 1], X_labels, fontsize=12, rotation=45)
plt.axhline(y=0, color="k", alpha=0.5, ls="--")
plt.legend()
plt.title("Weight recovery", fontsize=15)
plt.show()

## This looks pretty bad 
# - maybe my initial parameters are too far off? 
# - maybe theres a problem with the order - i havent permuted them?
