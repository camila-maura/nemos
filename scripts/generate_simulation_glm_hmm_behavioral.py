"""
Simulate a single unit GLM with hidden Markov states.

This script generates synthetic behavioral data where each animal's activity
depends on both external features and a latent state that evolves over time
according to a Markov process.

Based on Venditto's script for behavioral simulation from
https://github.com/Brody-Lab/venditto_glm-hmm/blob/main/glmhmm_example_fit.m 
and on the behavioral variables considered in Ashwood et al. (2020).
"""
# Imports
import nemos as nmo
import numpy as np
import jax.numpy as jnp

# Assuming that a glm-hmm generated the data
# Number of sessions to simulate
n_sess = 30 

# Number of trials in a session
n_trials_per_sess = 100

# Total number of timepoints
total_n_trials = n_sess * n_trials_per_sess

# Indicator for session starts
new_sess = np.zeros(total_n_trials, dtype=int)
# Set 1 at the beginning of each session
new_sess[::n_trials_per_sess] = 1


# Number of latent states
n_states = 3

# We will consider a design matrix with the
# following behavioral variables
# x = [bias, stimulus, last_choice, win_stay_lose_shift]
X_labels = ["bias", "stimulus"]#, "last_choice", "win-stay-lose-shift"]

# Number of features in design matrix
n_features = len(X_labels)

# Projection weights, using parameters from Ashwood et al. (2020) 
# results for example IBL mouse (Fig. 2)

# NOTE: in the ssh tutorial, they only use Bias and Stimulus. Since computing previous choice and WSLS might take a bit, I will only use those two for now. It is pending to add the other two elements in the projection weights and design matrix.
projection_weights = np.array([
    [1, -3, 3],     # Bias
    [6, 2, 2],      # Stimulus
    #[0, -.1, .1],   # Previous choice
    #[0, 0, 0],      # Win stay Lose switch
])

# Initial state probabilities
initial_prob = jnp.array([0.95, 0.025, 0.025]) # Couldnt find exact numbers from fast check in Ashwood et al. - revise again

# Initialize GLM
glm = nmo.glm.GLM(observation_model="Bernoulli")
glm.intercept_ = jnp.zeros((1,))

# Initialize storage
latent_states = np.zeros((total_n_trials, n_states), dtype=int)
rates = np.zeros((total_n_trials,))
counts = np.zeros((total_n_trials,))

# Simuli shown to the mice
stim_vals = [-1, -0.5, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 0.5, 1]

# Generate random sequence of stimuli for simulation


for i in range(n_sess):



