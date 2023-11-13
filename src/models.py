import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import pickle
from scipy import stats, special, optimize, integrate
import random



def model_beta(x, alpha, beta):
    Np = 744
    f = lambda a: ((a**(alpha-1)) * ((1-a)**(beta-1)) / special.beta(alpha, beta) ) * (1 - (((alpha + beta) / alpha) * 2*a*x[0])/(Np**2-Np))**(Np - 1)
    ff, _ = integrate.quad(f, 0, 1)  # 積分の値と推定誤差を別々に受け取る
    func = Np * (1 - ff) 
    return func



def model_dirichlet(x, a, Np, alpha):
    # ディリクレ分布の確率密度関数
    rng = np.random.default_rng()
    alpha = rng.uniform(0, 1, int(Np))
    a = rng.dirichlet(alpha, size=int(Np))
    func = Np * (1 - (a * (1 - (2*a*x)/(Np**2-Np)))**(Np - 1))
    return func

# 初期値を決定するアルゴリズム
def initial_params(edge, node):
    # まずはNpを推定する
    # Npの初期値を決定する
    # Npの初期値は、ノード数の平均値とする
    Np = max(node)

    # Npの初期値を用いて、alphaとbetaの初期値を決定する
    # alphaとbetaの初期値は、Npの値を用いて、ノード数の平均値とする
    alpha = random.uniform(0, 1)
    alpha = round(alpha, 2)
    beta = random.uniform(0, 1)
    beta = round(beta, 2)

    # aの初期値を決定する
    # aの初期値は、alphaとbetaの初期値を用いて、ベータ関数から求める
    a = special.beta(alpha, beta)

    return a, Np, alpha, beta

def calc_N(Np, kappa):
    return Np * (1 - (2 / (kappa * Np))*(1 - (1 - (kappa / 2))**Np))

def calc_M(Np, kappa):
    return (kappa * Np * (Np - 1)) / 8

def calc_kappa(Np, M, alpha, beta):
    return (((alpha + beta) / alpha)**2) * (2 * M / (Np * (Np - 1)))

def initial_params_for_dirichlet(edge, node):
    # まずはNpを推定する
    # Npの初期値を決定する
    # Npの初期値は、ノード数の平均値とする
    Np = max(node)

    # Npの初期値を用いて、alphaとbetaの初期値を決定する
    # alphaとbetaの初期値は、Npの値を用いて、ノード数の平均値とする
    rng = np.random.default_rng()
    alpha = rng.uniform(0, 1, Np)
    a = rng.dirichlet(alpha, size=Np)
    # aをarrayに変換
    a = np.array(a)

    return a, Np, alpha