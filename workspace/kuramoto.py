## MODULES ##
# Written by Cayden Murray

import numpy as np
from scipy.integrate import odeint
from sklearn.decomposition import PCA
import statsmodels.formula.api as smf


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from funcs import load_data




## DATA AND TIMEPOINTS ##


data = load_data("data/connectomes_cobre_scale_444.npy")
t = np.linspace(0,100,100)




## MODEL ##


def kuramoto_model(theta,t,omega,K):
    dtheta_dt = np.zeros(N)
    for i in range(K.shape[0]):


        theta_sum = np.sum(K[i,:]*np.sin(theta - theta[i]))
        dtheta_indv = omega[i] + theta_sum
        dtheta_dt[i] = dtheta_indv
   
    return dtheta_dt


def compute_global_order_parameter(solutions):
    R = np.abs(np.sum(np.exp(1j * solutions), axis=1)) / N


    return R




## SIMULATE PATIENTS ##


N = data.shape[1]
'''
main_dict = {'solutions':[],'R':[],'C':[]}
for patient in range(0,50):
    omega = np.random.normal(0,1,N)
    theta_0 = 2*np.pi*np.random.normal(0,1,N)
    K = data[patient,:,:]


    solutions = odeint(kuramoto_model,theta_0,t,args=(omega,K))
    main_dict['solutions'].append(solutions)
    main_dict['R'].append(compute_global_order_parameter(solutions))


    print(f'Finished patient {patient}')


for (key,value) in main_dict.items():
    np.save(f'{key}.npy',np.array(main_dict[key]))
'''


## VISUALIZE GLOBAL PARAMETER ##


R_final = np.load('R.npy') / 444
fig,ax = plt.subplots(figsize=(15,10))
palette = {'control': 'blue', 'Sch': 'red'}


labels_list = []
with open('subjects.txt','r') as f:
    for val,line in enumerate(f):
        if line.split('xxx')[0] == 'cont':
            labels_list.append('cont')


        else:
            labels_list.append('sch')


for i in range(R_final.shape[0]):
    if i in [0, 1, 3]:
        sns.lineplot(ax=ax, data=R_final[i, :], label='control', color=palette['control'])
    else:
        sns.lineplot(ax=ax, data=R_final[i, :], label='Sch', color=palette['Sch'])


handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax.legend(unique_labels.values(), unique_labels.keys())


#plt.show()




## STATS ##


df = pd.DataFrame(R_final[:6,:])
labels_list = labels_list[:6]
df['group'] = labels_list
df_long = df.melt(id_vars=['group'], var_name='timepoint', value_name='order_param')
df_long['group'] = df_long['group'].astype('category')
model = smf.ols('order_param ~ timepoint * group', data=df_long).fit()


print(model.summary())
