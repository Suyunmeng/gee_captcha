# 第一阶段：构建依赖
FROM python:3.10-slim as builder

WORKDIR /app
COPY requirements.txt .

# 合并清理步骤到单个 RUN 指令
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc python3-dev wget \
 && pip3 install --user --no-cache-dir -r requirements.txt \
 && apt-get purge -y --auto-remove gcc python3-dev \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
 
RUN wget -4 -q -O /app/PP-HGNetV2-B4.onnx https://huggingface.co/luguoyixiazi/model_save/resolve/main/PP-HGNetV2-B4.onnx

# 第二阶段：运行环境
FROM python:3.10-slim

# 合并运行时依赖安装步骤
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

# 精确复制 Python 依赖
COPY --from=builder /root/.local/lib/python3.10/site-packages /root/.local/lib/python3.10/site-packages
COPY --from=builder /root/.local/bin /root/.local/bin
COPY . .
RUN mkdir /app/model
COPY --from=builder /app/PP-HGNetV2-B4.onnx /app/model/PP-HGNetV2-B4.onnx

# 环境变量与清理
ENV PATH=/root/.local/bin:$PATH \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONDONTWRITEBYTECODE=1

# 清理 Python 缓存和临时文件
RUN find /app -name "*.pyc" -delete \
 && find /app -name "__pycache__" -delete \
 && find /root/.local -name "*.pyc" -delete \
 && find /root/.local -type d -name "__pycache__" -exec rm -rf {} +

CMD ["python3", "main.py"]