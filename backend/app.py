"""
贝尔曼方程求解器 - Flask API 服务
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from bellman_solver import Parameters, BellmanSolver

app = Flask(__name__)
CORS(app)


@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({'status': 'ok', 'service': 'bellman-solver'})


@app.route('/api/solve', methods=['POST'])
def solve_bellman():
    """
    求解贝尔曼方程

    请求体参数:
    - alpha: 买入折扣因子 (默认 0.9)
    - beta: 卖出折扣因子 (默认 0.9)
    - eta: 库存损耗率 (默认 0.95)
    - delta: 时间折扣因子 (默认 0.9)
    - T: 时间期数 (默认 12)
    - x_1: 初始库存 (默认 1.0)
    - p_mean: 价格均值 (默认 5.0)
    - p_std: 价格标准差 (默认 2.0)
    - seed: 随机种子 (可选)
    """
    data = request.get_json() or {}

    # 解析参数
    params = Parameters(
        alpha=data.get('alpha', 0.9),
        beta=data.get('beta', 0.9),
        eta=data.get('eta', 0.95),
        delta=data.get('delta', 0.9),
        T=data.get('T', 12),
        x_1=data.get('x_1', 1.0),
        p_mean=data.get('p_mean', 5.0),
        p_std=data.get('p_std', 2.0),
        n_price_samples=data.get('n_price_samples', 100)
    )

    # 设置随机种子
    seed = data.get('seed', 42)
    np.random.seed(seed)

    # 创建求解器
    solver = BellmanSolver(params)

    # 生成价格序列
    observed_prices = np.random.normal(params.p_mean, params.p_std, params.T)
    observed_prices = np.maximum(observed_prices, 0.01)

    # 求解
    V, optimal_y, w_functions = solver.solve_bellman(observed_prices)

    # 构建最优路径
    path = []
    x_t = params.x_1
    total_reward = 0

    for t in range(1, params.T + 1):
        p_t = observed_prices[t - 1]
        key = (round(x_t, 4), round(p_t, 4))

        if key not in V[t]:
            closest_key = min(V[t].keys(), key=lambda k: abs(k[0] - x_t))
            key = closest_key

        y_star = optimal_y[t].get(key, 0)
        v_t = V[t].get(key, 0)

        R_t = solver.immediate_reward(
            np.array([x_t]), np.array([y_star]), p_t
        )[0]
        total_reward += R_t * (params.delta ** (t - 1))

        delta_inv = x_t - y_star / params.eta
        if delta_inv > 0.001:
            action = "sell"
        elif delta_inv < -0.001:
            action = "buy"
        else:
            action = "hold"

        w_s = w_functions['w_S'][t].get(key, 0)
        w_b = w_functions['w_B'][t].get(key, 0)

        path.append({
            't': t,
            'x_t': round(x_t, 4),
            'p_t': round(p_t, 4),
            'y_star': round(y_star, 4),
            'V_t': round(v_t, 4),
            'R_t': round(R_t, 4),
            'action': action,
            'w_S': round(w_s, 4),
            'w_B': round(w_b, 4)
        })

        x_t = y_star

    return jsonify({
        'parameters': {
            'alpha': params.alpha,
            'beta': params.beta,
            'eta': params.eta,
            'delta': params.delta,
            'T': params.T,
            'x_1': params.x_1,
            'p_mean': params.p_mean,
            'p_std': params.p_std
        },
        'observed_prices': [round(p, 4) for p in observed_prices.tolist()],
        'optimal_path': path,
        'total_discounted_reward': round(total_reward, 4)
    })


@app.route('/api/immediate_reward', methods=['POST'])
def compute_immediate_reward():
    """
    计算即时收益

    请求体参数:
    - x_t: 初始库存
    - y_t: 结束库存
    - p_t: 价格
    - alpha, beta, eta: 模型参数
    """
    data = request.get_json() or {}

    params = Parameters(
        alpha=data.get('alpha', 0.9),
        beta=data.get('beta', 0.9),
        eta=data.get('eta', 0.95)
    )

    solver = BellmanSolver(params)

    x_t = np.array([data.get('x_t', 1.0)])
    y_t = np.array([data.get('y_t', 0.5)])
    p_t = data.get('p_t', 5.0)

    R_sell = solver.immediate_reward_sell(x_t, y_t, p_t)[0]
    R_buy = solver.immediate_reward_buy(x_t, y_t, p_t)[0]
    R_total = solver.immediate_reward(x_t, y_t, p_t)[0]

    return jsonify({
        'x_t': x_t[0],
        'y_t': y_t[0],
        'p_t': p_t,
        'y_t_over_eta': round(y_t[0] / params.eta, 4),
        'R_sell': round(R_sell, 4),
        'R_buy': round(R_buy, 4),
        'R_total': round(R_total, 4)
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
