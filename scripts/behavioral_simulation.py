
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import nemos as nmo

seed = 0  # Random seed for reproducibility
np.random.seed(seed)
jax.config.update("jax_enable_x64", True)

def behavioral_sim():                                                                       
    n_sess = 20  # Number of sessions to simulate
    n_trials_per_sess = 100  # Number of trials in a session
    n_timepoints = n_sess * n_trials_per_sess  # Total number of timepoints
    new_sess = np.zeros(n_timepoints, dtype=int)  # Indicator for session starts
    new_sess[::n_trials_per_sess] = 1  # Set 1 at the beginning of each session
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
    X = np.ones((n_timepoints, n_features))
    X[:, 1] = np.random.choice(stim_vals, n_timepoints)

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
    glm.intercept_ = jnp.zeros((1,))

    # Set initial weights and simulate first timepoint
    glm.coef_ = true_projection_weights[..., initial_state].reshape(
        true_projection_weights.shape[0], 1
    )

    # Set key for replication
    key = jax.random.PRNGKey(seed)

    # Simulate first count and proba
    res_sim = glm.simulate(key, X[:1])

    true_choices[0] = res_sim[0][0].item()
    choice_probas[0] = res_sim[1][0].item()

    # Simulate remaining timepoints
    print("Simulating data...")
    for t in range(1, n_timepoints):
        # Sample next state
        key, subkey = jax.random.split(key)

        prev_state_vec = true_latent_states[t - 1]

        # print(f"PREVIOUS STATE VECTOR {prev_state_vec} ")
        # print(f"TRANSITION PROBAS {true_transition_prob}")
        transition_probs = true_transition_prob.T @ prev_state_vec
        # print(f"TRANSITION PROBS {transition_probs}")

        next_state = jax.random.choice(subkey, jnp.arange(n_states), p=transition_probs)
        # print(next_state)
        # print(true_transition_prob)

        true_latent_states[t, next_state] = 1

        # Update weights and simulate
        glm.coef_ = true_projection_weights[..., next_state]
        key, subkey = jax.random.split(key)
        res = glm.simulate(subkey, X[t : t + 1])

        true_choices[t] = res[0][0]
        choice_probas[t] = res[1][0]
        print(f"Simulated timepoint {t+1}/{n_timepoints}", end="\r")
    print("\nSimulation complete.")
    