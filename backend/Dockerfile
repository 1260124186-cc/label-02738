# 使用官方 Python 镜像，支持多平台 (ARM64 和 AMD64)
FROM --platform=$BUILDPLATFORM python:3.11-slim AS builder

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 最终镜像
FROM python:3.11-slim

WORKDIR /app

# 从 builder 复制已安装的包
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/gunicorn /usr/local/bin/gunicorn

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 5000

# 使用 gunicorn 运行
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
