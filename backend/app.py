"""
贝尔曼方程求解器 - Flask API 服务
"""

import logging
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from bellman_solver import Parameters, BellmanSolver

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


class ValidationError(Exception):
    """参数验证错误"""
    pass


def validate_params(data: dict) -> dict:
    """
    验证并解析输入参数

    Raises:
        ValidationError: 参数不合法时抛出
    """
    errors = []

    # 验证 alpha
    alpha = data.get('alpha', 0.9)
    if not isinstance(alpha, (int, float)) or alpha <= 0 or alpha > 1:
        errors.append("alpha 必须在 (0, 1] 范围内")

    # 验证 beta
    beta = data.get('beta', 0.9)
    if not isinstance(beta, (int, float)) or beta <= 0 or beta > 1:
        errors.append("beta 必须在 (0, 1] 范围内")

    # 验证 eta
    eta = data.get('eta', 0.95)
    if not isinstance(eta, (int, float)) or eta <= 0 or eta > 1:
        errors.append("eta 必须在 (0, 1] 范围内")

    # 验证 delta
    delta = data.get('delta', 0.9)
    if not isinstance(delta, (int, float)) or delta <= 0 or delta > 1:
        errors.append("delta 必须在 (0, 1] 范围内")

    # 验证 T
    T = data.get('T', 12)
    if not isinstance(T, int) or T <= 0:
        errors.append("T 必须为正整数")

    # 验证 x_1
    x_1 = data.get('x_1', 1.0)
    if not isinstance(x_1, (int, float)) or x_1 < 0:
        errors.append("x_1 必须为非负数")

    # 验证 p_mean
    p_mean = data.get('p_mean', 5.0)
    if not isinstance(p_mean, (int, float)) or p_mean <= 0:
        errors.append("p_mean 必须为正数")

    # 验证 p_std
    p_std = data.get('p_std', 2.0)
    if not isinstance(p_std, (int, float)) or p_std <= 0:
        errors.append("p_std 必须为正数")

    if errors:
        raise ValidationError("; ".join(errors))

    return {
        'alpha': float(alpha),
        'beta': float(beta),
        'eta': float(eta),
        'delta': float(delta),
        'T': int(T),
        'x_1': float(x_1),
        'p_mean': float(p_mean),
        'p_std': float(p_std),
        'n_price_samples': int(data.get('n_price_samples', 100))
    }


@app.errorhandler(ValidationError)
def handle_validation_error(e):
    """处理验证错误"""
    logger.warning(f"Validation error: {str(e)}")
    return jsonify({'error': 'validation_error', 'message': str(e)}), 400


@app.errorhandler(Exception)
def handle_exception(e):
    """处理未知错误"""
    logger.error(f"Unexpected error: {str(e)}", exc_info=True)
    return jsonify({'error': 'internal_error', 'message': '服务器内部错误'}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({'status': 'ok', 'service': 'bellman-solver'})


@app.route('/api/solve', methods=['POST'])
def solve_bellman():
    """
    求解贝尔曼方程

    请求体参数:
    - alpha: 买入折扣因子 (默认 0.9, 范围 (0, 1])
    - beta: 卖出折扣因子 (默认 0.9, 范围 (0, 1])
    - eta: 库存损耗率 (默认 0.95, 范围 (0, 1])
    - delta: 时间折扣因子 (默认 0.9, 范围 (0, 1])
    - T: 时间期数 (默认 12, 正整数)
    - x_1: 初始库存 (默认 1.0, 非负数)
    - p_mean: 价格均值 (默认 5.0, 正数)
    - p_std: 价格标准差 (默认 2.0, 正数)
    - seed: 随机种子 (可选)
    """
    data = request.get_json() or {}
    logger.info(f"Received solve request with params: {data}")

    # 验证参数
    validated = validate_params(data)

    params = Parameters(**validated)

    # 设置随机种子
    seed = data.get('seed', 42)
    np.random.seed(seed)
    logger.info(f"Using random seed: {seed}")

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

        # w^S 和 w^B 现在以 (y_t, p_t) 为键
        w_key = (round(y_star, 4), round(p_t, 4))
        w_s = w_functions['w_S'][t].get(w_key, 0)
        w_b = w_functions['w_B'][t].get(w_key, 0)

        # 如果没有精确匹配，找最近的 y_t 值
        if w_key not in w_functions['w_S'][t]:
            closest_w_key = min(w_functions['w_S'][t].keys(),
                               key=lambda k: abs(k[0] - y_star))
            w_s = w_functions['w_S'][t].get(closest_w_key, 0)
            w_b = w_functions['w_B'][t].get(closest_w_key, 0)

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

    logger.info(f"Solve completed. Total discounted reward: {total_reward:.4f}")

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
    - x_t: 初始库存 (非负数)
    - y_t: 结束库存 (非负数)
    - p_t: 价格 (正数)
    - alpha, beta, eta: 模型参数
    """
    data = request.get_json() or {}
    logger.info(f"Received immediate_reward request: {data}")

    # 验证参数
    errors = []

    alpha = data.get('alpha', 0.9)
    if not isinstance(alpha, (int, float)) or alpha <= 0 or alpha > 1:
        errors.append("alpha 必须在 (0, 1] 范围内")

    beta = data.get('beta', 0.9)
    if not isinstance(beta, (int, float)) or beta <= 0 or beta > 1:
        errors.append("beta 必须在 (0, 1] 范围内")

    eta = data.get('eta', 0.95)
    if not isinstance(eta, (int, float)) or eta <= 0 or eta > 1:
        errors.append("eta 必须在 (0, 1] 范围内")

    x_t_val = data.get('x_t', 1.0)
    if not isinstance(x_t_val, (int, float)) or x_t_val < 0:
        errors.append("x_t 必须为非负数")

    y_t_val = data.get('y_t', 0.5)
    if not isinstance(y_t_val, (int, float)) or y_t_val < 0:
        errors.append("y_t 必须为非负数")

    p_t = data.get('p_t', 5.0)
    if not isinstance(p_t, (int, float)) or p_t <= 0:
        errors.append("p_t 必须为正数")

    if errors:
        raise ValidationError("; ".join(errors))

    params = Parameters(
        alpha=float(alpha),
        beta=float(beta),
        eta=float(eta)
    )

    solver = BellmanSolver(params)

    x_t = np.array([float(x_t_val)])
    y_t = np.array([float(y_t_val)])

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
    logger.info("Starting Bellman Solver API server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
