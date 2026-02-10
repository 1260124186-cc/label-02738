"""
贝尔曼方程求解器

求解动态规划问题:
V_t(x_t, p_t) = max_{y_t in [0, eta]} { R(y_t/eta - x_t, p_t) + delta * E_t[V_{t+1}(y_t, p_{t+1})] }
V_{T+1} = 0

其中 R 是即时收益函数，分为卖出和买入两种情况。
"""

import logging
import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        self._interpolation_count = 0  # 追踪插值次数
        # 预生成价格样本用于计算期望
        self.price_samples = np.random.normal(
            params.p_mean, params.p_std, params.n_price_samples
        )
        # 确保价格为正
        self.price_samples = np.maximum(self.price_samples, 0.01)
        logger.info(f"BellmanSolver initialized with params: alpha={params.alpha}, "
                   f"beta={params.beta}, eta={params.eta}, T={params.T}")

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
        missing_count = 0
        for p_next in self.price_samples:
            key = (y_t, round(p_next, 4))
            if key in V_next:
                expected_values.append(V_next[key])
            else:
                missing_count += 1
                # 使用插值找最近值
                closest_value = self._find_closest_value(V_next, y_t, p_next)
                expected_values.append(closest_value)

        if missing_count > 0:
            logger.debug(f"compute_expected_value: {missing_count}/{len(self.price_samples)} "
                        f"samples required interpolation for y_t={y_t:.4f}")

        return np.mean(expected_values)

    def compute_w_S(self, y_t: float, p_t: float, EV_next: float) -> float:
        """
        计算 w^S(y_t, p_t) - 卖出情况的价值函数（去掉常数项和max）

        公式: w^S_t(y_t, p_t) = -p_t * y_t * (beta/eta) + delta * E_t[V_{t+1}(y_t, p_{t+1})]

        Args:
            y_t: 决策变量（结束库存）
            p_t: 当期价格
            EV_next: 期望未来价值 E_t[V_{t+1}(y_t, p_{t+1})]
        """
        return -p_t * y_t * (self.params.beta / self.params.eta) + self.params.delta * EV_next

    def compute_w_B(self, y_t: float, p_t: float, EV_next: float) -> float:
        """
        计算 w^B(y_t, p_t) - 买入情况的价值函数（去掉常数项和max）

        公式: w^B_t(y_t, p_t) = -p_t * y_t * (1/(alpha*eta)) + delta * E_t[V_{t+1}(y_t, p_{t+1})]

        Args:
            y_t: 决策变量（结束库存）
            p_t: 当期价格
            EV_next: 期望未来价值 E_t[V_{t+1}(y_t, p_{t+1})]
        """
        return -p_t * y_t / (self.params.alpha * self.params.eta) + self.params.delta * EV_next

    def solve_bellman(self, observed_prices: np.ndarray) -> Tuple[Dict, Dict, Dict]:
        """
        后向递归求解贝尔曼方程

        Args:
            observed_prices: 观察到的价格序列 [p_1, p_2, ..., p_T]

        Returns:
            V: 值函数字典 {t: {(x, p): value}}
            optimal_y: 最优决策字典 {t: {(x, p): y*}}
            w_functions: w^S 和 w^B 函数字典，存储在不同 y_t 值上的函数值
                - w_S[t][(y_t, p_t)] = w^S_t(y_t, p_t)
                - w_B[t][(y_t, p_t)] = w^B_t(y_t, p_t)
        """
        logger.info(f"Starting Bellman equation solve for T={self.params.T} periods")
        self._interpolation_count = 0

        T = self.params.T
        eta = self.params.eta

        # 存储结果
        V = {t: {} for t in range(1, T + 2)}
        optimal_y = {t: {} for t in range(1, T + 1)}
        # w^S 和 w^B 是关于 (y_t, p_t) 的函数
        w_S = {t: {} for t in range(1, T + 1)}
        w_B = {t: {} for t in range(1, T + 1)}

        # 后向递归
        for t in range(T, 0, -1):
            p_t = observed_prices[t - 1]
            logger.debug(f"Processing period t={t}, p_t={p_t:.4f}")

            if t == 1:
                x_values = [self.params.x_1]
            else:
                x_values = np.linspace(0, eta, 50)

            # y_t 的可能取值范围 [0, eta]
            y_values = np.linspace(0, eta, 50)

            # 先计算所有 y_t 值对应的 w^S 和 w^B
            for y_t in y_values:
                if t == T:
                    EV_next = 0
                else:
                    EV_next = self._compute_expected_value_vectorized(
                        V[t + 1], y_t, observed_prices
                    )

                y_key = (round(y_t, 4), round(p_t, 4))
                w_S[t][y_key] = self.compute_w_S(y_t, p_t, EV_next)
                w_B[t][y_key] = self.compute_w_B(y_t, p_t, EV_next)

            for x_t in x_values:
                candidates = np.array([0, eta * x_t, eta])
                candidates = np.clip(candidates, 0, eta)

                best_value = -np.inf
                best_y = 0

                for y_t in candidates:
                    R_t = self.immediate_reward(
                        np.array([x_t]), np.array([y_t]), p_t
                    )[0]

                    if t == T:
                        EV_next = 0
                    else:
                        EV_next = self._compute_expected_value_vectorized(
                            V[t + 1], y_t, observed_prices
                        )

                    total_value = R_t + self.params.delta * EV_next

                    if total_value > best_value:
                        best_value = total_value
                        best_y = y_t

                key = (round(x_t, 4), round(p_t, 4))
                V[t][key] = best_value
                optimal_y[t][key] = best_y

        logger.info(f"Bellman solve completed. Total interpolations: {self._interpolation_count}")
        return V, optimal_y, {'w_S': w_S, 'w_B': w_B}

    def _compute_expected_value_vectorized(
        self, V_next: Dict, y_t: float, observed_prices: np.ndarray
    ) -> float:
        """向量化计算期望值"""
        x_next = y_t
        values = []

        for p_sample in self.price_samples:
            key = (round(x_next, 4), round(p_sample, 4))
            if key in V_next:
                values.append(V_next[key])
            else:
                closest_value = self._find_closest_value(V_next, x_next, p_sample)
                values.append(closest_value)
                self._interpolation_count += 1

        return np.mean(values) if values else 0

    def _find_closest_value(self, V_dict: Dict, x: float, p: float) -> float:
        """找到最接近的状态值"""
        if not V_dict:
            logger.warning(f"Empty V_dict when finding closest value for x={x:.4f}, p={p:.4f}")
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
    np.random.seed(42)

    params = Parameters()
    solver = BellmanSolver(params)

    observed_prices = np.random.normal(params.p_mean, params.p_std, params.T)
    observed_prices = np.maximum(observed_prices, 0.01)

    logger.info("=" * 60)
    logger.info("贝尔曼方程求解器")
    logger.info("=" * 60)
    logger.info(f"模型参数: alpha={params.alpha}, beta={params.beta}, "
               f"eta={params.eta}, delta={params.delta}, T={params.T}")

    V, optimal_y, w_functions = solver.solve_bellman(observed_prices)

    x_t = params.x_1
    total_reward = 0

    logger.info("最优决策路径:")
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
            action = "卖出"
        elif delta_inv < -0.001:
            action = "买入"
        else:
            action = "持有"

        logger.info(f"  t={t}: x_t={x_t:.4f}, p_t={p_t:.4f}, y*={y_star:.4f}, "
                   f"V_t={v_t:.4f}, R_t={R_t:.4f}, 动作={action}")
        x_t = y_star

    logger.info(f"总折扣收益: {total_reward:.4f}")
    return V, optimal_y, w_functions, observed_prices


def test_immediate_reward():
    """测试即时收益函数"""
    logger.info("测试即时收益函数")

    params = Parameters()
    solver = BellmanSolver(params)

    test_cases = [
        (1.0, 0.5, 5.0, "卖出情况"),
        (0.5, 0.95, 5.0, "买入情况"),
        (0.5, 0.475, 5.0, "持有情况"),
    ]

    for x_t, y_t, p_t, desc in test_cases:
        R_total = solver.immediate_reward(np.array([x_t]), np.array([y_t]), p_t)[0]
        logger.info(f"{desc}: x_t={x_t}, y_t={y_t}, R={R_total:.4f}")


if __name__ == "__main__":
    test_immediate_reward()
    run_simulation()
