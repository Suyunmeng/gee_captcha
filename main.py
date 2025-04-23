import json
import time
import random
from crack import Crack
from crop_image import crop_image_v3,save_path,save_fail_path,save_pass_path,validate_path
import httpx
from fastapi import FastAPI,Query
from fastapi.responses import JSONResponse
import shutil
import os

port = 7860
# api
app = FastAPI()


@app.get("/pass_nine")
def get_pic(gt: str = Query(...), 
            challenge: str = Query(...), 
            point: str = Query(default=None), 
            use_v3_model = Query(default=True),
            save_result = Query(default=False)
           ):
    print(f"开始获取:\ngt:{gt}\nchallenge:{challenge}")
    t = time.time()

    crack = Crack(gt, challenge)

    crack.gettype()

    crack.get_c_s()

    time.sleep(random.uniform(0.4,0.6))

    crack.ajax()

    pic_content,pic_name = crack.get_pic()

    crop_image_v3(pic_content)
    
    if use_v3_model:
        result_list = predict_onnx_pdl(validate_path)
    else:
        with open(f"{validate_path}/cropped_9.jpg", "rb") as rb:
            icon_image = rb.read()
        with open(f"{validate_path}/nine.jpg", "rb") as rb:
            bg_image = rb.read()
        result_list = predict_onnx(icon_image, bg_image, point)
        
    point_list = [f"{col}_{row}" for row, col in result_list]
    wait_time = max(0,4.0 - (time.time() - t))
    time.sleep(wait_time)
    result = json.loads(crack.verify(point_list))
    if save_result:
        shutil.move(os.path.join(validate_path,pic_name),os.path.join(save_path,pic_name))
        if 'validate' in result['data']:
            path_2_save = os.path.join(save_pass_path,pic_name.split('.')[0])
        else:
            path_2_save = os.path.join(save_fail_path,pic_name.split('.')[0])
        os.makedirs(path_2_save,exist_ok=True)
        for pic in os.listdir(validate_path):
            if pic.startswith('cropped'):
                shutil.move(os.path.join(validate_path,pic),os.path.join(path_2_save,pic))
    total_time = time.time() - t
    print(f"总计耗时(含等待{wait_time}s): {total_time}\n{result}")
    return JSONResponse(content=result)



if __name__ == "__main__":
    from predict import predict_onnx,predict_onnx_pdl
    import uvicorn
    print(f"{' '*10}api: http://0.0.0.0:{port}/pass_nine{' '*10}")
    print(f"{' '*10}api所需参数：gt、challenge、point(可选){' '*10}")
    uvicorn.run(app,host="0.0.0.0",port=port)
