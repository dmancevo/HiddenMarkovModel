import numpy as np

class HMM:
    def __init__(self, k_states):
        self.k_states = k_states
        alphas = [2 for _ in range(k_states)]
        self.log_pi = np.log(np.random.dirichlet(alphas)) # Initial state log probabilities.
        self.log_A = np.empty((k_states, k_states)) # Transition log probabilities.
        for k in range(k_states):
            self.log_A[k,:] = np.log(np.random.dirichlet(alphas))

    def fit(self, X):
        '''
        Fit Hidden Markov Model.

        X float np.array (n_timesteps).

        Implementing Baum-Welch without underflowing requires the following bit of
        logarithmic magic:

        log(a + b) = log(a * (1+b/a)) = log(a) + log(1+exp(log(b) - log(a)))
        '''
        T = X.shape[0]
        self.X = X

        # Initialize emission probabilities.
        self.log_B = np.empty((self.k_states, T))
        alphas = [2 for _ in range(T)]
        for k in range(self.k_states):
            self.log_B[k,:] = np.log(np.random.dirichlet(alphas))

        # f(log(p1), log(p2)) = log( sum(p1 + p2) )
        log_sum_p = np.frompyfunc(lambda a,b: a + np.log(1+np.exp(b-a)), 2, 1)

        # Forward.
        log_alpha =  np.empty((self.k_states, T))
        log_alpha[:,0] = self.log_pi + self.log_B[:,0]
        for t in range(1,T):
            log_alpha_A = (log_alpha[:,t-1] + self.log_A.T).T # log(alpha_i(t-1) * a_ij)
            log_alpha[:,t] = self.log_B[:,t] + log_sum_p.reduce(log_alpha_A, axis=0)

        # Backward.
        log_beta = np.empty((self.k_states, T))
        log_beta[:,T-1] = 0
        for t in range(T-2,-1,-1):
            log_beta_A_B = (log_beta[:,t+1] + self.log_A.T + self.log_B[:,t+1]).T # log(beta_j(t+1) * a_ij * b_j(t+1))
            log_beta[:,t] = log_sum_p.reduce(log_beta_A_B, axis=0)

        # Update

        # log( (alpha_i(t) * beta_i(t)) / P(Y | params))
        log_gamma = log_alpha + log_beta \
                    - log_sum_p.reduce(log_alpha + log_beta, axis=0).astype(np.float64)

        # log(P(X_t=i, X_(t+1)=j | Y, params)) = log( (alpha_i(t) * a_ij * beta_i(t+1) * b_ij) / P(Y | params) )
        log_epsilon = np.empty((self.k_states, self.k_states, T-1))
        for t in range(T-1):
            log_epsilon[:,:,t] = (log_alpha[:,t] + log_beta[:,t+1] + self.log_B[:,t+1] + self.log_A.T).T
        log_epsilon = (log_epsilon.T - log_sum_p.reduce(log_epsilon, axis=2)).T.astype(np.float64)

        self.log_pi = log_gamma[:,0]
        # log(sum_t(epsilon_ij(t)) / sum_t(gamma_i(t)))
        self.log_A = (log_sum_p.reduce(log_epsilon, axis=2).astype(np.float64).T\
                        - log_sum_p.reduce(log_gamma[:,1:], axis=1).astype(np.float64)).T
        # log( gamma_i(t) / sum_t(gamma_i(t)) )
        self.log_B = (log_gamma[:,1:].T - log_sum_p.reduce(log_gamma[:,1:], axis=1)\
                        .astype(np.float64)).T
