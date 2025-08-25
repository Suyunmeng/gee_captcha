FROM python:3.9-slim

WORKDIR /app

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_without_train.txt .

RUN pip install --no-cache-dir -r requirements_without_train.txt

RUN mkdir -p model img_saved img_2_val

RUN echo "Downloading..." && \
    wget -O model/d-fine-n.onnx https://huggingface.co/luguoyixiazi/model_save/resolve/main/d-fine-n.onnx && \
    echo "Download Success."

COPY . .

ENV use_pdl=0
ENV use_dfine=1
ENV PORT=7860

EXPOSE 7860

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--reload"]