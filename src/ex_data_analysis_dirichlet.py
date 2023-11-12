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

print("+------------------+")
print("| Loading datasets |")
print("+------------------+")

df_left, df_right = pre.preprocess('data/df.pickle')

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
start = time.time()
initial_params = md.initial_params_for_dirichlet(edge_right, node_right)
params, _ = optimize.curve_fit(md.model_dirichlet, xdata=edge_right, ydata=node_right, p0=initial_params)

Np = params[1]
kappa = []
N_fit = []
M_fit = []

for i in range(len(edge_right)):
    kappa.append(md.calc_kappa(Np, edge_right[i], params[2], params[3]))

for i in range(len(edge_right)):
    N_fit.append(md.calc_N(Np, kappa[i]))

for i in range(len(edge_right)):
    M_fit.append(md.calc_M(Np, kappa[i]))

elapsed_time = time.time() - start
elapsed_time = round(elapsed_time, 2)

Np = round(Np, 2)
alpha = round(params[2], 2)
# beta = round(params[3], 2)

# Npの値を整形
print("Np: ", params[1])
print("alpha: ", params[2])
# print("beta: ", params[3])
# print("kappa: ", kappa)
print("+------------------+")
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
print("+------------------+")

print("+------------------+")
print("| Initial values   |")
print("+------------------+")
print("a: ", initial_params[0])
print("Np: ", initial_params[1])
print("alpha: ", initial_params[2])
# print("beta: ", initial_params[3])
print("+------------------+")

# plot
print("+------------------+")
print("| Plotting         |")
print("+------------------+")
fig, ax = plt.subplots()
ax.scatter(node_right, edge_right, label='N-M (right)')
ax.scatter(N_fit, M_fit, label='Model beta distribution version')
# 推定値を載せる
fig.text(0.15, 0.75, r'$\hat{N_p}$: ' + str(Np), size=12, transform=fig.transFigure, ha="left", va="top")
fig.text(0.15, 0.7, r'$\alpha$: ' + str(alpha), size=12, transform=fig.transFigure, ha="left", va="top")
# fig.text(0.15, 0.65, r'$\beta$: ' + str(beta), size=12, transform=fig.transFigure, ha="left", va="top")
plt.xlabel('N')
plt.ylabel('M')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()