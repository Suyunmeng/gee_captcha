import os
import json
import time
import httpx
import shutil
import random
import logging
import logging.config
from crack import Crack
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from fastapi import FastAPI,Query, HTTPException
from predict import predict_onnx,predict_onnx_pdl,predict_onnx_dfine
from crop_image import crop_image_v3,save_path,save_fail_path,save_pass_path,validate_path

PORT = int(os.getenv('PORT', 9645))
platform = os.name
# --- 日志配置字典 ---
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": f"uvicorn.{'_logging' if platform== 'nt' else 'logging'}.DefaultFormatter",
            "fmt": "%(levelprefix)s %(asctime)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    },
    "loggers": {
        # 将根日志记录器的级别设置为 INFO
        "": {"handlers": ["default"], "level": "INFO"},
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
    },
}
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)
def get_available_hosts() -> set[str]:
    """获取本机所有可用的IPv4地址。"""
    import socket
    hosts = {"127.0.0.1"}
    try:
        hostname = socket.gethostname()
        addr_info = socket.getaddrinfo(hostname, None, socket.AF_INET)
        hosts.update({info[4][0] for info in addr_info})
    except socket.gaierror:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                hosts.add(s.getsockname()[0])
        except OSError:
            pass
    return hosts

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("="*50)
    logger.info("启动服务中...")
    # 从 uvicorn 配置中获取 host 和 port
    server = app.servers[0] if app.servers else None
    host = server.config.host if server else "0.0.0.0"
    port = server.config.port if server else PORT
    if host == "0.0.0.0":
        available_hosts = get_available_hosts()
        logger.info(f"服务地址(依需求选用，docker中使用宿主机host:{port}):")
        for h in sorted(list(available_hosts)):
            logger.info(f"  - http://{h}:{port}")
    else:
        logger.info(f"服务地址: http://{host}:{port}")
    logger.info(f"可用服务路径如下:")
    for route in app.routes:
        logger.info(f"    -{route.methods} {route.path}")
    logger.info("="*50)
    
    yield
    logger.info("="*50)
    logger.info("服务关闭")
    logger.info("="*50)

app = FastAPI(title="极验V3图标点选+九宫格", lifespan=lifespan)

def prepare(gt: str, challenge: str) -> tuple[Crack, bytes, str, str]:
    """获取信息。"""
    logging.info(f"开始获取:\ngt:{gt}\nchallenge:{challenge}")
    crack = Crack(gt, challenge)
    crack.gettype()
    crack.get_c_s()
    time.sleep(random.uniform(0.4,0.6))
    crack.ajax()
    pic_content,pic_name,pic_type = crack.get_pic()
    return crack,pic_content,pic_name,pic_type

def do_pass_nine(pic_content: bytes, use_v3_model: bool, point: Optional[str]) -> list[str]:
    """处理九宫格验证码，返回坐标点列表。"""
    crop_image_v3(pic_content)
    if use_v3_model:
        result_list = predict_onnx_pdl(validate_path)
    else:
        with open(f"{validate_path}/cropped_9.jpg", "rb") as rb:
            icon_image = rb.read()
        with open(f"{validate_path}/nine.jpg", "rb") as rb:
            bg_image = rb.read()
        result_list = predict_onnx(icon_image, bg_image, point)
    return [f"{col}_{row}" for row, col in result_list]

def do_pass_icon(pic:Any, draw_result: bool) -> list[str]:
    """处理图标点选验证码，返回坐标点列表。"""
    result_list = predict_onnx_dfine(pic,draw_result)
    print(result_list)
    return [f"{round(x / 333 * 10000)}_{round(y / 333 * 10000)}" for x, y in result_list]

def save_image_for_train(pic_name,pic_type,passed):
    shutil.move(os.path.join(validate_path,pic_name),os.path.join(save_path,pic_name))
    if passed:
        path_2_save = os.path.join(save_pass_path,pic_name.split('.')[0])
    else:
        path_2_save = os.path.join(save_fail_path,pic_name.split('.')[0])
    os.makedirs(path_2_save,exist_ok=True)
    for pic in os.listdir(validate_path):
        if pic_type == "nine" and pic.startswith('cropped'):
            shutil.move(os.path.join(validate_path,pic),os.path.join(path_2_save,pic))
        if pic_type == "icon" and pic.startswith('icon'):
            shutil.move(os.path.join(validate_path,pic),os.path.join(path_2_save,pic))


def handle_pass_request(gt: str, challenge: str, save_result: bool, **kwargs) -> JSONResponse:
    """
    统一处理所有验证码请求的核心函数。
    """
    start_time = time.monotonic()
    try:
        # 1. 准备
        crack, pic_content, pic_name, pic_type = prepare(gt, challenge)
        
        # 2. 识别
        
        if pic_type == "nine":
            point_list = do_pass_nine(
                pic_content,
                use_v3_model=kwargs.get("use_v3_model", True),
                point=kwargs.get("point",None)
            )
        elif pic_type == "icon":
            point_list = do_pass_icon(pic_content, save_result)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown picture type: {pic_type}")

        # 3. 验证
        elapsed = time.monotonic() - start_time
        wait_time = max(0, 4.0 - elapsed)
        time.sleep(wait_time)

        response_str = crack.verify(point_list)
        result = json.loads(response_str)

        # 4. 后处理
        passed = 'validate' in result.get('data', {})
        if save_result:
            save_image_for_train(pic_name, pic_type, passed)
        else:
            os.remove(os.path.join(validate_path,pic_name))

        total_time = time.monotonic() - start_time
        logging.info(
            f"请求完成,耗时: {total_time:.2f}s (等待 {wait_time:.2f}s). "
            f"结果: {result}"
        )
        return JSONResponse(content=result)

    except Exception as e:
        logging.error(f"服务错误: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "An internal server error occurred.", "detail": str(e)}
        )

    

@app.get("/pass_nine")
def pass_nine(gt: str = Query(...), 
            challenge: str = Query(...), 
            point: str = Query(default=None), 
            use_v3_model = Query(default=True),
            save_result = Query(default=False)
           ):
    return handle_pass_request(
        gt, challenge, save_result,
        use_v3_model=use_v3_model, point=point
    )

@app.get("/pass_icon")
def pass_icon(gt: str = Query(...), 
            challenge: str = Query(...),
            save_result = Query(default=False)
            ):
    return handle_pass_request(gt, challenge, save_result)

@app.get("/pass_uni")
def pass_uni(gt: str = Query(...), 
            challenge: str = Query(...),
            save_result = Query(default=False)
            ):
    return handle_pass_request(gt, challenge, save_result)

@app.get("/pass_hutao")
def pass_hutao(gt: str = Query(...), 
            challenge: str = Query(...),
            save_result = Query(default=False)):
    try:
        # 调用原函数获取返回值
        response = handle_pass_request(gt, challenge, save_result)
        # 获取原始状态码和内容
        original_status_code = response.status_code
        original_content = json.loads(response.body.decode("utf-8"))
        if original_status_code == 200 and original_content.get("status",False)=="success" and "validate" in original_content.get("data",{}):
            rebuild_content = {"code":0,"data":{"gt":gt,"challenge":challenge,"validate":original_content["data"]["validate"]}}
        else:
            rebuild_content = {"code":1,"data":{"gt":gt,"challenge":challenge,"validate":original_content}}
        return JSONResponse(content=rebuild_content, status_code=original_status_code)

    except Exception as e:
        logging.error(f"修改路由错误: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "An internal server error occurred.", "detail": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,port=PORT)