import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import beta

# データを生成する関数 (仮の関数)
def generate_data(x, true_Np, true_alpha, true_beta, true_a):
    true_ff = beta(true_alpha, true_beta).cdf(true_a)
    true_func = true_Np * (1 - true_ff * ((1 - ((true_alpha + true_beta) / true_alpha) * 2*true_a*x)/(true_Np**2-true_Np))**(true_Np - 1))
    noise = np.random.normal(0, 0.1, size=len(x))  # ノイズを追加
    return true_func + noise

# モデル関数
def model_beta(x, a, Np, alpha, beta):
    ff = beta(alpha, beta).cdf(a)  # 累積分布関数を使用
    func = Np * (1 - ff * ((1 - ((alpha + beta) / alpha) * 2*a*x)/(Np**2-Np))**(Np - 1))
    return func

# データを生成
np.random.seed(42)  # 乱数の再現性を確保
x_data = np.linspace(0.1, 1, 50)
true_Np, true_alpha, true_beta, true_a = 2, 2, 5, 0.5
y_data = generate_data(x_data, true_Np, true_alpha, true_beta, true_a)

# フィッティングのための初期推定値
initial_guess = [0.5, 1.5, 1.5, 3.5]

# curve_fitを使用してパラメータを推定
params, covariance = curve_fit(model_beta, x_data, y_data, p0=initial_guess)

# 推定されたパラメータ
estimated_a, estimated_Np, estimated_alpha, estimated_beta = params
print("Estimated Parameters:")
print("a:", estimated_a)
print("Np:", estimated_Np)
print("alpha:", estimated_alpha)
print("beta:", estimated_beta)
