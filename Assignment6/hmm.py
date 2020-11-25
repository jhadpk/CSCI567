from __future__ import print_function
import json
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_{t-1} = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: A dictionary mapping each observation symbol to its index 
        - state_dict: A dictionary mapping each state to its index
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array where alpha[i, t-1] = P(Z_t = s_i, X_{1:t}=x_{1:t})
                 (note that this is alpha[i, t-1] instead of alpha[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        output = self.obs_dict[Osequence[0]]
        for i in range(S):
            alpha[i][0] = self.pi[i] * self.B[i][output]
        for t in range(1, L):
            xt = self.obs_dict[Osequence[t]]
            for s in range(S):
                sigma_a_alpha = 0
                for s_dash in range(S):
                    sigma_a_alpha = sigma_a_alpha + self.A[s_dash][s] * alpha[s_dash][t - 1]
                alpha[s][t] = self.B[s][xt] * sigma_a_alpha
        return alpha




    def backward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array where beta[i, t-1] = P(X_{t+1:T}=x_{t+1:T} | Z_t = s_i)
                    (note that this is beta[i, t-1] instead of beta[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        T = L-1
        for s in range(S):
            beta[s][T] = 1
        for tr in range(0, T):
            t = T - 1 - tr
            for s in range(S):
                sigma_a_b_beta = 0
                for s_dash in range(S):
                    sigma_a_b_beta = sigma_a_b_beta + self.A[s][s_dash] * self.B[s_dash][self.obs_dict[Osequence[t + 1]]] * beta[s_dash][t + 1]
                beta[s][t] = sigma_a_b_beta
        return beta


    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(X_{1:T}=x_{1:T})
        """
        T = len(Osequence) - 1
        alpha = self.forward(Osequence)
        prob = 0
        for s in range(len(self.pi)):
            prob += alpha[s][T]
        return prob


    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - gamma: (num_state*L) A numpy array where gamma[i, t-1] = P(Z_t = s_i | X_{1:T}=x_{1:T})
        (note that this is gamma[i, t-1] instead of gamma[i, t])
        """
        beta = self.backward(Osequence)
        alpha = self.forward(Osequence)
        S = len(self.pi)
        T = len(Osequence)
        prob = np.zeros([S, T])
        denominator = 0
        for s in range(S):
            denominator += alpha[s][T - 1]
        for i in range(T):
            for s in range(S):
                numerator = alpha[s][i] * beta[s][i]
                prob[s][i] = numerator / denominator
        return prob


    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array where prob[i, j, t-1] = 
                    P(Z_t = s_i, Z_{t+1} = s_j | X_{1:T}=x_{1:T})
        """
        beta = self.backward(Osequence)
        alpha = self.forward(Osequence)
        S = len(self.pi)
        T = len(Osequence)
        prob = np.zeros([S, S, T - 1])
        denominator = 0
        for s in range(S):
            denominator += alpha[s][T - 1]
        for i in range(T - 1):
            for s in range(S):
                for s_dash in range(S):
                    prob[s, s_dash, i] = self.A[s, s_dash] * self.B[s_dash, self.obs_dict[Osequence[i + 1]]] * beta[s_dash, i + 1] * alpha[s, i] / denominator
        return prob


    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden states (return actual states instead of their indices;
                    you might find the given function self.find_key useful)
        """
        S = len(self.pi)
        T = len(Osequence)
        big_delta = np.zeros([S, T], dtype="int")
        delta = np.zeros([S, T])
        xt = self.obs_dict[Osequence[0]]
        for s in range(S):
            delta[s][0] = self.pi[s] * self.B[s][xt]

        for t in range(1, T):
            xt = self.obs_dict[Osequence[t]]
            for s in range(S):
                max_a_smalldelta = -1
                argmax_a_smalldelta = -1
                for s_dash in range(S):
                    delta_temp = self.A[s_dash][s] * delta[s_dash][t - 1]
                    if delta_temp > max_a_smalldelta:
                        max_a_smalldelta = delta_temp
                        argmax_a_smalldelta = s_dash
                delta[s][t] = self.B[s][xt] * max_a_smalldelta
                big_delta[s][t] = argmax_a_smalldelta
        z_star = []
        z = np.argmax(delta[:, T - 1])
        z_star.append(z)
        for t in range(T - 1, 0, -1):
            z = big_delta[z][t]
            z_star.append(z)
        z_star = z_star[::-1]
        state_sequence = [0] * len(z_star)

        for state, observation in self.state_dict.items():
            for i in range(len(z_star)):
                if observation == z_star[i]:
                    state_sequence[i] = state
        return state_sequence


    #DO NOT MODIFY CODE BELOW
    def find_key(self, obs_dict, idx):
        for item in obs_dict:
            if obs_dict[item] == idx:
                return item

    def find_item(self, Osequence):
        O = []
        for item in Osequence:
            O.append(self.obs_dict[item])
        return O
