import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import time

import preprocessing as pre

start = time.time()
print("+------------------+")
print("| Loading datasets |")
print("+------------------+")

df_left, df_right = pre.preprocess('../data/df.pickle')

print("+------------------+")
print("| Done             |")
print("+------------------+")

# ここからデータ分析
# 非線形最小二乗法でパラメータ推定
from scipy import stats, special, optimize, integrate
import models as md

# パラメータ推定
print("+------------------+")
print("| Estimating       |")
print("+------------------+")
node_left, edge_left, timestamp_left = pre.return_nm(df_left)
node_right, edge_right, timestamp_right = pre.return_nm(df_right)





# Npを推定する


initial_params = md.initial_params(edge_right, node_right)
params, _ = optimize.curve_fit(md.model_beta, xdata=edge_right, ydata=node_right, p0=[0.75,0.5])

kappa = []
N_fit = []
M_fit = []

for i in range(len(edge_right)):
    kappa.append(md.calc_kappa(max(node_right), edge_right[i], params[0], params[1]))

for i in range(len(edge_right)):
    N_fit.append(md.calc_N(max(node_right), kappa[i]))

for i in range(len(edge_right)):
    M_fit.append(md.calc_M(max(node_right), kappa[i]))





alpha = round(params[0], 2)
beta = round(params[1], 2)


print("alpha: ", params[0])
print("beta: ", params[1])
# print("kappa: ", kappa)


print("+------------------+")
print("| Initial values   |")
print("+------------------+")
# print("a: ", initial_params[0])
print("alpha: ", initial_params[0])
print("beta: ", initial_params[1])
print("+------------------+")

sample_x = np.linspace(1, max(node_right), 1000)

def fit(func, x, param_init):
    X = x[0]
    Y = x[1]
    popt, pcov = optimize.curve_fit(func, X, Y, p0=param_init)
    perr = np.sqrt(np.diag(pcov))
    y=func(sample_x, *popt)
    return y, popt, perr


# plot
print("+------------------+")
print("| Plotting         |")
print("+------------------+")
res = fit(md.model_beta, [edge_right,node_right], [0.75, 0.5])
fig, ax = plt.subplots()
ax.scatter(node_right, edge_right, label='N-M (right)')
ax.plot(sample_x, res[0],label='Model beta distribution version')
# 推定値を載せる
# fig.text(0.15, 0.75, r'$\hat{N_p}$: ' + str(Np), size=12, transform=fig.transFigure, ha="left", va="top")
fig.text(0.15, 0.7, r'$\alpha$: ' + str(alpha), size=12, transform=fig.transFigure, ha="left", va="top")
fig.text(0.15, 0.65, r'$\beta$: ' + str(beta), size=12, transform=fig.transFigure, ha="left", va="top")
plt.xlabel('N')
plt.ylabel('M')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

elapsed_time = time.time() - start
elapsed_time = round(elapsed_time, 2)
print("+------------------+")
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
print("+------------------+")