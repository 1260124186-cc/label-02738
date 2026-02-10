# 贝尔曼方程求解器

## How to Run

```bash
# 构建并启动服务
docker-compose up --build -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

## Services

| 服务 | 端口 | 描述 |
|------|------|------|
| backend | 8084 | 贝尔曼方程求解 API |

API 端点:
- `GET /api/health` - 健康检查
- `POST /api/solve` - 求解贝尔曼方程
- `POST /api/immediate_reward` - 计算即时收益

## 测试账号

本项目为计算服务，无需登录账号。

## 题目内容
我需要你编写程序来计算一个贝尔曼方程，用python 写并测试

$$
\begin{aligned}
V_{t}(x_{t}, p_{t}) & = \max _{y_{t} \in [0, \eta]} \left\{ R\left( \frac{y_{t}}{\eta}-x_{t}, P_{t} \right) + \delta \mathbb{E}_{t}(V_{t}(y_{t}, p_{t+1})) \right\}

\\

V_{T+1}(x_{T+1}, p_{T+1}) & = 0
\end{aligned}
$$

where

$$
R\left( \frac{y_{t}}{\eta}-x_{t}, p_{t} \right) = p_{t}\beta\left( x_{t} - \frac{y_{t}}{\eta} \right)^{+} + \left( -(p_{t} /\alpha) \left( \frac{y_{t}}{\eta}-x_{t} \right)^{+} \right)
$$

and $\mathbb{E}_{t}$ is shorthand of $\mathbb{E} [\cdot | p_{t}]$，也就是说，对第二项（价格）求平均。

$R$ 代表这个时刻的 immediate payoff. $x_{t}$ 是没做决定时的库存，$y_{t}$ 是这个时刻最后的库存（在做了决定和 每次eta 的固定损失后）, $p_{t}$ 是价格，在R中的第一项中，库存减少意味着卖出，第二项意味着买入，$\alpha, \beta$ 都是小于等于1的常数。$\delta \mathbb{E}_{t}(V_{t}(y_{t}, p_{t+1}))$ 代表下一时刻的期望, $\delta$ 是折扣因子


然后在计算出

$$
\begin{aligned}
w_{t}^{S}(y_{t}, p_{t})  & =  \left( -P_{t} \cdot y_{t} \cdot  \frac{\beta}{\eta }\right)  + \delta \mathbb{E}_{t}(V_{t}(y_{t}, p_{t+1}))  \\
w_{t}^{B}(y_{t}, p_{t})  & =  \left( -P_{t} \cdot y_{t} \cdot  \frac{1}{ \alpha \eta }\right)  + \delta \mathbb{E}_{t}(V_{t}(y_{t}, p_{t+1}))
\end{aligned}
$$

第一个情况对应着卖出，去掉常数项和max，第二个是只买入.
所以在计算 immediate payoff，你可能需要分两种情况，这样后面计算这两个函数会方便一点。


假设
$\alpha = \beta = 0.9$, $x_{1}= 1$, $T = 12$, $\eta = 0.95$. $p_{t}$ 是正态分布, 平均为5方差为2，$\delta = 0.9$.

你需要先生成一组p_t 来知道你在 $t$ 时刻观察到的数据，然后计算 $R$ 和 $\delta \mathbb{E}_{t}(V_{t}(y_{t}, p_{t+1}))$. 因为下一时刻的价格未知，我们又知道真实分布，所以求平均 make sense。

这样你就知道max里的真实函数了，已经证明最大值在三个地方，一个是0, 一个是eta, 一个是eta x_t，你只需比较

你可能需要向量化来加速运算,注意可读性，
---

## 项目介绍

本项目实现了一个动态规划问题的贝尔曼方程求解器，用于库存管理决策优化。

### 核心功能

- 后向递归求解贝尔曼方程
- 计算最优库存决策路径
- 支持自定义模型参数
- 提供 RESTful API 接口

### 技术栈

- Python 3.11
- Flask (Web 框架)
- NumPy (数值计算)
- Gunicorn (WSGI 服务器)
- Docker (容器化)

### API 使用示例

```bash
# 求解贝尔曼方程
curl -X POST http://localhost:8084/api/solve \
  -H "Content-Type: application/json" \
  -d '{"alpha": 0.9, "beta": 0.9, "T": 12}'

# 计算即时收益
curl -X POST http://localhost:8084/api/immediate_reward \
  -H "Content-Type: application/json" \
  -d '{"x_t": 1.0, "y_t": 0.5, "p_t": 5.0}'
```


## 自测验证

### 1. 健康检查

```bash
curl http://localhost:8084/api/health
```

预期输出：
```json
{"service":"bellman-solver","status":"ok"}
```

### 2. 求解贝尔曼方程

```bash
curl -X POST http://localhost:8084/api/solve \
  -H "Content-Type: application/json" \
  -d '{"T": 5, "seed": 42}'
```

预期输出：
```json
{
  "observed_prices": [2.1693, 4.1587, 4.3146, 3.3954, 4.6774],
  "optimal_path": [
    {"t": 1, "x_t": 1.0, "p_t": 2.1693, "y_star": 0.95, "V_t": 3.2001, "R_t": 0.0, "action": "hold", "w_S": 1.2478, "w_B": 0.7898},
    {"t": 2, "x_t": 0.95, "p_t": 4.1587, "y_star": 0.0, "V_t": 3.5557, "R_t": 3.5557, "action": "sell", "w_S": 0.0, "w_B": 0.0},
    {"t": 3, "x_t": 0.0, "p_t": 4.3146, "y_star": 0.0, "V_t": 0.0, "R_t": 0.0, "action": "hold", "w_S": 0.0, "w_B": 0.0},
    {"t": 4, "x_t": 0.0, "p_t": 3.3954, "y_star": 0.0, "V_t": 0.0, "R_t": 0.0, "action": "hold", "w_S": 0.0, "w_B": 0.0},
    {"t": 5, "x_t": 0.0, "p_t": 4.6774, "y_star": 0.0, "V_t": 0.0, "R_t": 0.0, "action": "hold", "w_S": 0.0, "w_B": 0.0}
  ],
  "parameters": {"T": 5, "alpha": 0.9, "beta": 0.9, "delta": 0.9, "eta": 0.95, "p_mean": 5.0, "p_std": 2.0, "x_1": 1.0},
  "total_discounted_reward": 3.2001
}
```

### 3. 计算即时收益

```bash
curl -X POST http://localhost:8084/api/immediate_reward \
  -H "Content-Type: application/json" \
  -d '{"x_t": 1.0, "y_t": 0.5, "p_t": 5.0}'
```

预期输出：
```json
{
  "R_buy": 0.0,
  "R_sell": 2.1316,
  "R_total": 2.1316,
  "p_t": 5.0,
  "x_t": 1.0,
  "y_t": 0.5,
  "y_t_over_eta": 0.5263
}
```

说明：`x_t=1.0 > y_t/eta=0.5263`，库存减少，为卖出操作，收益为正。
