import numpy as np
import pandas as pd

obs_map = {'Cold':0, 'Hot':1}
obs = np.array([1,1,0,1,0,0,1,0,1,1,0,0,0,1])

inv_obs_map = dict((v, k) for k, v in obs_map.items())
obs_seq = [inv_obs_map[v] for v in list(obs)]

print("Simulated Observations:\n", pd.DataFrame(np.column_stack([obs, obs_seq]), columns = ['Obs_code', 'Obs_seq']))

states = ['Cold', 'Hot']
hidden_states = ['Snow', 'Rain', 'Sunshine']
pi = [0, 0.2, 0.8]
state_space = pd.Series(pi, index = hidden_states, name = 'States')
a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)
a_df.loc[hidden_states[0]] = [0.3, 0.3, 0.4]
a_df.loc[hidden_states[1]] = [0.1, 0.45, 0.45]
a_df.loc[hidden_states[2]] = [0.2, 0.3, 0.5]
print('\nTransition Matrix:\n', a_df)
a = a_df.values

observable_states = states
b_df = pd.DataFrame(columns=observable_states, index=hidden_states)
b_df.loc[hidden_states[0]] = [1,0]
b_df.loc[hidden_states[1]] = [0.8, 0.2]
b_df.loc[hidden_states[2]] = [0.3, 0.7]
print('\nObservable Layer Matrix:\n', b_df)
b = b_df.values

def viterbi(pi, a, b, obs):
    nStates = np.shape(b)[0]
    T = np.shape(obs)[0]
    
    path = np.zeros(T, dtype = int)
    delta = np.zeros((nStates, T))
    phi = np.zeros((nStates, T))
    
    delta[:, 0] = pi * b[:, obs[0]] 
    phi[:, 0] = 0
    
    print('\nStart Walk Forward\n')
    for t in range(1,T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t-1] * a[:, s]) * b[s, obs[t]]
            phi[s, t] = np.argmax(delta[:, t-1] * a[:, s])
            print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s, t]))
    print('-'*50)
    print('Start Backtrace\n')
    path[T-1] = np.argmax(delta[:, T-1])
    print('path[{}] = {}'.format(T-1, path[T-1]))
    for t in range(T-2, -1, -1):
        path[t] = phi[path[t+1], t+1]
        print('path[{}] = {}'.format(t, path[t]))
        
    return path, delta, phi

path, delta, phi = viterbi(pi, a, b, obs)
state_map = {0:'Snow', 1:'Rain', 2:'Sunshine'}
state_path = [state_map[v] for v in path]
print(pd.DataFrame().assign(Observation=obs_seq).assign(Best_Path=state_path))
