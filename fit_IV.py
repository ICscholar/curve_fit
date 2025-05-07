import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# 读取数据（保持正向和反向扫描数据分离）
V_forward = pd.read_csv('data_firstHalf.csv', header=None).iloc[:, 0].values
I_forward = pd.read_csv('data_firstHalf.csv', header=None).iloc[:, 1].values
V_reverse = pd.read_csv('data_secHalf.csv', header=None).iloc[:, 0].values
I_reverse = pd.read_csv('data_secHalf.csv', header=None).iloc[:, 1].values

# 固定参数
SCAN_TIME = 20  # 单程扫描时间（正向或反向）
eta = 4.0
dt = SCAN_TIME / len(V_forward)  # 假设正向反向点数相同

def IV_model(V_forward, V_reverse, eta, lambda_, tau, alpha, beta, gama, sigma):
    # 正向扫描
    w_forward = np.zeros(len(V_forward))
    I_forward_sim = np.zeros(len(V_forward))
    w_forward[0] = 0
    for i in range(1, len(V_forward)):
        dw = (lambda_ * np.sinh(eta * V_forward[i]) - w_forward[i-1]/tau) * dt
        w_forward[i] = w_forward[i-1] + dw
        I_forward_sim[i] = (1-w_forward[i])*alpha*(1-np.exp(-beta*V_forward[i])) + w_forward[i]*gama*np.sinh(sigma*V_forward[i])
    
    # 反向扫描（从正向的最终w值开始）
    w_reverse = np.zeros(len(V_reverse))
    I_reverse_sim = np.zeros(len(V_reverse))
    w_reverse[0] = w_forward[-1]  # 从正向结束时的w值开始
    for i in range(1, len(V_reverse)):
        dw = (lambda_ * np.sinh(eta * V_reverse[i]) - w_reverse[i-1]/tau) * dt
        w_reverse[i] = w_reverse[i-1] + dw
        I_reverse_sim[i] = (1-w_reverse[i])*alpha*(1-np.exp(-beta*V_reverse[i])) + w_reverse[i]*gama*np.sinh(sigma*V_reverse[i])
    
    return np.concatenate([I_forward_sim, I_reverse_sim])

def residuals(params, V_forward, V_reverse, eta, I_forward, I_reverse):
    lambda_, tau, alpha, beta, gama, sigma = params
    I_sim = IV_model(V_forward, V_reverse, eta, lambda_, tau, alpha, beta, gama, sigma)
    I_actual = np.concatenate([I_forward, I_reverse])
    return I_sim - I_actual

# 初始参数和边界
initial_guess = [1.0, 1.0, 1e-6, 1.0, 1e-6, 1.0]
bounds_lower = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
bounds_upper = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]

# 优化
result = least_squares(
    residuals,
    initial_guess,
    args=(V_forward, V_reverse, eta, I_forward, I_reverse),
    bounds=(bounds_lower, bounds_upper),
    # ===== 误差控制参数 =====
    ftol=1e-10,     # 函数值相对变化容差（默认1e-8）
    xtol=1e-10,     # 参数变化容差（默认1e-8）
    gtol=1e-10,     # 梯度范数容差（默认1e-8）
    # ===== 高级控制 =====
    max_nfev=1_000_000,  # 最大函数评估次数
    x_scale='jac',       # 参数缩放方式（'auto'/'jac'/array）
    loss='soft_l1',      # 损失函数类型
    f_scale=1.0,         # 损失函数缩放因子
    diff_step=1e-3,      # 有限差分步长
    tr_solver='exact',   # 信赖域求解器
    verbose=2            # 详细输出（0-2）
)
# 结果
opt_params = result.x
print("Optimized parameters:")
for name, value in zip(['lambda_', 'tau', 'alpha', 'beta', 'gama', 'sigma'], opt_params):
    print(f"'{name}': {value:.4e},")

# 修改后的结果提取和绘图部分
I_fit_full = IV_model(V_forward, V_reverse, eta, *opt_params)
I_fit_forward = I_fit_full[:len(V_forward)]  # 正向拟合结果
I_fit_reverse = I_fit_full[len(V_forward):]  # 反向拟合结果

# 确保维度匹配
assert len(V_forward) == len(I_fit_forward), "正向维度不匹配"
assert len(V_reverse) == len(I_fit_reverse), "反向维度不匹配"

plt.figure(figsize=(12, 6))
plt.plot(V_forward, I_forward, 'b.', label='Forward Scan (Exp)', markersize=4)
plt.plot(V_reverse, I_reverse, 'g.', label='Reverse Scan (Exp)', markersize=4)
plt.plot(V_forward, I_fit_forward, 'r-', label='Forward Fit', linewidth=2)
plt.plot(V_reverse, I_fit_reverse, 'm-', label='Reverse Fit', linewidth=2)
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.title('IV Curve Hysteresis Fitting')
plt.legend()
plt.grid(True)
plt.show()

# Optimized parameters:
# lambda_: 2.0150e+03
# tau: 1.9435e+01
# alpha: 1.6158e-05
# beta: 1.0566e+00
# gama: 2.2796e-03
# sigma: 5.9212e-03