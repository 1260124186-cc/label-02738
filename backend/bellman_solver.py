"""
贝尔曼方程求解器

求解动态规划问题:
V_t(x_t, p_t) = max_{y_t in [0, eta]} { R(y_t/eta - x_t, p_t) + delta * E_t[V_{t+1}(y_t, p_{t+1})] }
V_{T+1} = 0

其中 R 是即时收益函数，分为卖出和买入两种情况。
"""

import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class Parameters:
    """模型参数"""
    alpha: float = 0.9      # 买入折扣因子
    beta: float = 0.9       # 卖出折扣因子
    eta: float = 0.95       # 库存损耗率
    delta: float = 0.9      # 时间折扣因子
    T: int = 12             # 时间期数
    x_1: float = 1.0        # 初始库存
    p_mean: float = 5.0     # 价格均值
    p_std: float = 2.0      # 价格标准差
    n_price_samples: int = 100  # 价格采样数量（用于期望计算）


class BellmanSolver:
    """贝尔曼方程求解器"""

    def __init__(self, params: Parameters):
        self.params = params
        # 预生成价格样本用于计算期望
        self.price_samples = np.random.normal(
            params.p_mean, params.p_std, params.n_price_samples
        )
        # 确保价格为正
        self.price_samples = np.maximum(self.price_samples, 0.01)

    def immediate_reward_sell(self, x_t: np.ndarray, y_t: np.ndarray, p_t: float) -> np.ndarray:
        """
        卖出情况的即时收益: p_t * beta * (x_t - y_t/eta)
        当 x_t >= y_t/eta 时（库存减少，卖出）
        """
        delta_inventory = x_t - y_t / self.params.eta
        return p_t * self.params.beta * np.maximum(delta_inventory, 0)

    def immediate_reward_buy(self, x_t: np.ndarray, y_t: np.ndarray, p_t: float) -> np.ndarray:
        """
        买入情况的即时收益: -(p_t/alpha) * (y_t/eta - x_t)
        当 y_t/eta > x_t 时（库存增加，买入）
        """
        delta_inventory = y_t / self.params.eta - x_t
        return -(p_t / self.params.alpha) * np.maximum(delta_inventory, 0)

    def immediate_reward(self, x_t: np.ndarray, y_t: np.ndarray, p_t: float) -> np.ndarray:
        """
        总即时收益 R(y_t/eta - x_t, p_t)
        """
        return self.immediate_reward_sell(x_t, y_t, p_t) + self.immediate_reward_buy(x_t, y_t, p_t)

    def compute_expected_value(self, V_next: Dict[float, float], y_t: float) -> float:
        """
        计算 E_t[V_{t+1}(y_t, p_{t+1})]
        对下一期价格求期望
        """
        expected_values = []
        for p_next in self.price_samples:
            # 下一期的状态 x_{t+1} = y_t（当期结束库存成为下期初始库存）
            key = (y_t, round(p_next, 4))
            if key in V_next:
                expected_values.append(V_next[key])
            else:
                # 如果没有精确匹配，使用插值或最近值
                expected_values.append(0)
        return np.mean(expected_values)


    def solve_bellman(self, observed_prices: np.ndarray) -> Tuple[Dict, Dict, Dict]:
        """
        后向递归求解贝尔曼方程

        Args:
            observed_prices: 观察到的价格序列 [p_1, p_2, ..., p_T]

        Returns:
            V: 值函数字典 {t: {(x, p): value}}
            optimal_y: 最优决策字典 {t: {(x, p): y*}}
            w_functions: w^S 和 w^B 函数字典
        """
        T = self.params.T
        eta = self.params.eta

        # 存储结果
        V = {t: {} for t in range(1, T + 2)}
        optimal_y = {t: {} for t in range(1, T + 1)}
        w_S = {t: {} for t in range(1, T + 1)}
        w_B = {t: {} for t in range(1, T + 1)}

        # 终端条件: V_{T+1} = 0
        # 不需要显式存储，默认为0

        # 后向递归
        for t in range(T, 0, -1):
            p_t = observed_prices[t - 1]  # 当期观察到的价格

            # 对于每个可能的初始库存 x_t
            # 这里我们追踪从 x_1 开始可能到达的状态
            if t == 1:
                x_values = [self.params.x_1]
            else:
                # 可能的 x_t 值来自上一期的 y_{t-1}
                x_values = np.linspace(0, eta, 50)

            for x_t in x_values:
                # 候选最优点: y_t in {0, eta * x_t, eta}
                candidates = np.array([0, eta * x_t, eta])
                candidates = np.clip(candidates, 0, eta)  # 确保在 [0, eta] 范围内

                best_value = -np.inf
                best_y = 0

                for y_t in candidates:
                    # 计算即时收益
                    R_t = self.immediate_reward(
                        np.array([x_t]), np.array([y_t]), p_t
                    )[0]

                    # 计算期望未来价值
                    if t == T:
                        EV_next = 0  # 终端条件
                    else:
                        EV_next = self._compute_expected_value_vectorized(
                            V[t + 1], y_t, observed_prices
                        )

                    total_value = R_t + self.params.delta * EV_next

                    if total_value > best_value:
                        best_value = total_value
                        best_y = y_t

                # 存储结果
                key = (round(x_t, 4), round(p_t, 4))
                V[t][key] = best_value
                optimal_y[t][key] = best_y

                # 计算 w^S 和 w^B
                if t < T:
                    EV_next = self._compute_expected_value_vectorized(
                        V[t + 1], best_y, observed_prices
                    )
                else:
                    EV_next = 0

                # w^S: 卖出情况
                w_S[t][key] = (-p_t * best_y * self.params.beta / eta +
                              self.params.delta * EV_next)

                # w^B: 买入情况
                w_B[t][key] = (-p_t * best_y / (self.params.alpha * eta) +
                              self.params.delta * EV_next)

        return V, optimal_y, {'w_S': w_S, 'w_B': w_B}

    def _compute_expected_value_vectorized(
        self, V_next: Dict, y_t: float, observed_prices: np.ndarray
    ) -> float:
        """
        向量化计算期望值
        使用蒙特卡洛方法对价格分布求期望
        """
        # 下一期的 x_{t+1} = y_t
        x_next = y_t

        # 对价格样本求期望
        values = []
        for p_sample in self.price_samples:
            # 查找最近的状态值
            key = (round(x_next, 4), round(p_sample, 4))
            if key in V_next:
                values.append(V_next[key])
            else:
                # 使用插值：找最近的 x 值
                closest_value = self._find_closest_value(V_next, x_next, p_sample)
                values.append(closest_value)

        return np.mean(values) if values else 0

    def _find_closest_value(self, V_dict: Dict, x: float, p: float) -> float:
        """找到最接近的状态值"""
        if not V_dict:
            return 0

        min_dist = np.inf
        closest_val = 0

        for (x_key, p_key), val in V_dict.items():
            dist = abs(x_key - x) + 0.1 * abs(p_key - p)
            if dist < min_dist:
                min_dist = dist
                closest_val = val

        return closest_val



def run_simulation():
    """运行模拟并展示结果"""
    np.random.seed(42)  # 设置随机种子以便复现

    # 初始化参数
    params = Parameters()
    solver = BellmanSolver(params)

    # 生成观察到的价格序列
    observed_prices = np.random.normal(params.p_mean, params.p_std, params.T)
    observed_prices = np.maximum(observed_prices, 0.01)  # 确保价格为正

    print("=" * 60)
    print("贝尔曼方程求解器")
    print("=" * 60)
    print(f"\n模型参数:")
    print(f"  alpha = {params.alpha}, beta = {params.beta}")
    print(f"  eta = {params.eta}, delta = {params.delta}")
    print(f"  T = {params.T}, x_1 = {params.x_1}")
    print(f"  价格分布: N({params.p_mean}, {params.p_std}^2)")

    print(f"\n观察到的价格序列:")
    for t, p in enumerate(observed_prices, 1):
        print(f"  t={t}: p_t = {p:.4f}")

    # 求解贝尔曼方程
    V, optimal_y, w_functions = solver.solve_bellman(observed_prices)

    # 展示结果
    print("\n" + "=" * 60)
    print("求解结果")
    print("=" * 60)

    # 从初始状态开始的最优路径
    print("\n最优决策路径 (从 x_1 = 1.0 开始):")
    print("-" * 50)

    x_t = params.x_1
    total_reward = 0

    for t in range(1, params.T + 1):
        p_t = observed_prices[t - 1]
        key = (round(x_t, 4), round(p_t, 4))

        # 找到最接近的键
        if key not in V[t]:
            closest_key = min(V[t].keys(), key=lambda k: abs(k[0] - x_t))
            key = closest_key

        y_star = optimal_y[t].get(key, 0)
        v_t = V[t].get(key, 0)

        # 计算即时收益
        R_t = solver.immediate_reward(
            np.array([x_t]), np.array([y_star]), p_t
        )[0]
        total_reward += R_t * (params.delta ** (t - 1))

        # 判断动作类型
        delta_inv = x_t - y_star / params.eta
        if delta_inv > 0.001:
            action = "卖出"
        elif delta_inv < -0.001:
            action = "买入"
        else:
            action = "持有"

        print(f"  t={t}: x_t={x_t:.4f}, p_t={p_t:.4f}, y*={y_star:.4f}, "
              f"V_t={v_t:.4f}, R_t={R_t:.4f}, 动作={action}")

        # 更新下一期状态
        x_t = y_star

    print(f"\n总折扣收益: {total_reward:.4f}")

    # 展示 w^S 和 w^B 函数
    print("\n" + "=" * 60)
    print("w^S 和 w^B 函数值 (在最优路径上)")
    print("=" * 60)

    x_t = params.x_1
    for t in range(1, params.T + 1):
        p_t = observed_prices[t - 1]
        key = (round(x_t, 4), round(p_t, 4))

        if key not in w_functions['w_S'][t]:
            closest_key = min(w_functions['w_S'][t].keys(),
                            key=lambda k: abs(k[0] - x_t))
            key = closest_key

        w_s = w_functions['w_S'][t].get(key, 0)
        w_b = w_functions['w_B'][t].get(key, 0)
        y_star = optimal_y[t].get(key, 0)

        print(f"  t={t}: w^S={w_s:.4f}, w^B={w_b:.4f}")
        x_t = y_star

    return V, optimal_y, w_functions, observed_prices


def test_immediate_reward():
    """测试即时收益函数"""
    print("\n" + "=" * 60)
    print("测试即时收益函数")
    print("=" * 60)

    params = Parameters()
    solver = BellmanSolver(params)

    # 测试用例
    test_cases = [
        (1.0, 0.5, 5.0, "卖出情况: x_t > y_t/eta"),
        (0.5, 0.95, 5.0, "买入情况: x_t < y_t/eta"),
        (0.5, 0.475, 5.0, "持有情况: x_t ≈ y_t/eta"),
    ]

    for x_t, y_t, p_t, desc in test_cases:
        R_sell = solver.immediate_reward_sell(np.array([x_t]), np.array([y_t]), p_t)[0]
        R_buy = solver.immediate_reward_buy(np.array([x_t]), np.array([y_t]), p_t)[0]
        R_total = solver.immediate_reward(np.array([x_t]), np.array([y_t]), p_t)[0]

        print(f"\n{desc}")
        print(f"  x_t={x_t}, y_t={y_t}, p_t={p_t}")
        print(f"  y_t/eta = {y_t/params.eta:.4f}")
        print(f"  R_sell = {R_sell:.4f}, R_buy = {R_buy:.4f}, R_total = {R_total:.4f}")


if __name__ == "__main__":
    # 运行测试
    test_immediate_reward()

    # 运行主模拟
    print("\n")
    V, optimal_y, w_functions, prices = run_simulation()
