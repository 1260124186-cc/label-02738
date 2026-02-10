"""
贝尔曼方程求解器的测试文件
"""

import numpy as np
from bellman_solver import Parameters, BellmanSolver, run_simulation, test_immediate_reward


def test_boundary_conditions():
    """测试边界条件"""
    print("=" * 60)
    print("测试边界条件")
    print("=" * 60)

    params = Parameters()
    solver = BellmanSolver(params)

    # 测试 y_t 在边界 [0, eta] 的情况
    x_t = np.array([0.5])
    p_t = 5.0

    print("\n测试 y_t 在边界值的即时收益:")
    for y_t in [0, params.eta * 0.5, params.eta]:
        R = solver.immediate_reward(x_t, np.array([y_t]), p_t)[0]
        print(f"  y_t = {y_t:.4f}: R = {R:.4f}")


def test_candidate_points():
    """测试三个候选最优点"""
    print("\n" + "=" * 60)
    print("测试三个候选最优点: {0, eta*x_t, eta}")
    print("=" * 60)

    params = Parameters()
    solver = BellmanSolver(params)

    x_t = 0.8
    p_t = 5.0
    eta = params.eta

    candidates = [0, eta * x_t, eta]

    print(f"\n当 x_t = {x_t}, p_t = {p_t}:")
    for y in candidates:
        R = solver.immediate_reward(np.array([x_t]), np.array([y]), p_t)[0]
        delta_inv = x_t - y / eta
        if delta_inv > 0.001:
            action = "卖出"
        elif delta_inv < -0.001:
            action = "买入"
        else:
            action = "持有"
        print(f"  y_t = {y:.4f} ({action}): R = {R:.4f}")


def test_different_parameters():
    """测试不同参数设置"""
    print("\n" + "=" * 60)
    print("测试不同参数设置")
    print("=" * 60)

    np.random.seed(123)

    # 测试不同的 alpha, beta 组合
    test_configs = [
        {"alpha": 0.9, "beta": 0.9, "desc": "对称交易成本"},
        {"alpha": 0.8, "beta": 0.95, "desc": "买入成本高"},
        {"alpha": 0.95, "beta": 0.8, "desc": "卖出成本高"},
    ]

    for config in test_configs:
        params = Parameters(alpha=config["alpha"], beta=config["beta"])
        solver = BellmanSolver(params)

        observed_prices = np.random.normal(params.p_mean, params.p_std, params.T)
        observed_prices = np.maximum(observed_prices, 0.01)

        V, optimal_y, w_functions = solver.solve_bellman(observed_prices)

        # 计算初始状态的值
        key = (round(params.x_1, 4), round(observed_prices[0], 4))
        if key in V[1]:
            v1 = V[1][key]
        else:
            v1 = list(V[1].values())[0] if V[1] else 0

        print(f"\n{config['desc']} (alpha={config['alpha']}, beta={config['beta']}):")
        print(f"  V_1(x_1, p_1) = {v1:.4f}")


def test_price_sensitivity():
    """测试价格敏感性"""
    print("\n" + "=" * 60)
    print("测试价格敏感性")
    print("=" * 60)

    params = Parameters()
    solver = BellmanSolver(params)

    x_t = np.array([0.5])
    y_t = np.array([0.3])  # 卖出情况

    print("\n卖出情况 (x_t=0.5, y_t=0.3) 在不同价格下的收益:")
    for p in [2, 4, 6, 8, 10]:
        R = solver.immediate_reward(x_t, y_t, float(p))[0]
        print(f"  p_t = {p}: R = {R:.4f}")


def test_vectorization():
    """测试向量化计算"""
    print("\n" + "=" * 60)
    print("测试向量化计算")
    print("=" * 60)

    params = Parameters()
    solver = BellmanSolver(params)

    # 批量计算
    x_t = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    y_t = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    p_t = 5.0

    R = solver.immediate_reward(x_t, y_t, p_t)

    print("\n批量计算即时收益:")
    for i in range(len(x_t)):
        print(f"  x_t={x_t[i]:.1f}, y_t={y_t[i]:.1f}: R={R[i]:.4f}")


def run_all_tests():
    """运行所有测试"""
    test_immediate_reward()
    test_boundary_conditions()
    test_candidate_points()
    test_different_parameters()
    test_price_sensitivity()
    test_vectorization()

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
