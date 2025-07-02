# 九宫格+点选测试代码

## **本项目仅供学习交流使用，请勿用于商业用途，否则后果自负。**

## **本项目仅供学习交流使用，请勿用于商业用途，否则后果自负。**

## **本项目仅供学习交流使用，请勿用于商业用途，否则后果自负。**

## 参考项目

模型及V4数据集：https://github.com/taisuii/ClassificationCaptchaOcr

点选检测模型https://github.com/Peterande/D-FINE

api：https://github.com/ravizhan/geetest-v3-click-crack

## 运行步骤

### 1.安装依赖（本地必选，使用docker跳至[5-b](#docker)）

（可选）a.如果要训练paddle的话还得安装paddlex及图像分类模块，安装看项目https://github.com/PaddlePaddle/PaddleX
（可选）b.d-fine训练看项目https://github.com/Peterande/D-FINE
（* 必选！）c.模型需要在项目目录下新建一个model文件夹，然后把模型文件放进去，具体命名可以是resnet18.onnx或者PP-HGNetV2-B4.onnx，默认使用PP-HGNetV2-B4模型，如果用resnet则use_v3_model设置为False，因为模型的输入输出不一样，可以自行修改，d-fine的模型同样放置此路径用于通过点选

```
pip install -r requirements.txt
```

仅推理
```
pip install -r requirements_without_train.txt
```

### 2.自行准备数据集，V3和V4有区别（可选），点选可以自己生成，要有旋转、重叠、换色

##### a. 训练resnet18（可选）

- 数据集详情参考上面标注的项目，但是上面项目是V4数据集，V3没有demo，自行发挥吧，用V4练V3不改代码正确率有点感人
- 主要是V4的尺寸和V3有差别，V4的api直接给两张图，一张是目标图，一张是九宫格，V3放在一起要切目标，且V3目标图清晰度很低，V4九宫格切了之后是100 * 86的图（去掉黑边），但是V3九宫格切的是112 * 112，不确定V4九宫格内容在V3基础上做了什么变换，反正改预处理就完事了

##### b. 训练PP-HGNetV2-B4（可选）

在paddle上随便找的，数据集格式如下，如果拿V4练V3，建议是多整点变换

```
   dataset
   ├─images   #所有图片存放路径
   ├─label.txt #标签路径，每一行数据格式为 <序号>+<空格>+<类别>，如15 地球仪
   ├─train.txt #训练图片，每一行数据格式为 <图片路径>+<空格>+<类别>，如images/001.jpg 0
   └─验证集和测试集同上
```

##### b. 训练d-fine（可选）

数据集格式如d-fine中标识，如果不修改源码则num_classes需+1，采用coco格式即可，我用的320*320，dataloader注释掉了RandomZoomOut、RandomHorizontalFlip、RandomIoUCrop（这些我全写在数据集生成中了）

##### c. 如果要切V3的九宫格图用crop_image.py的crop_image_v3，切V4则使用crop_image，自行编写切图脚本

### 3.训练模型（可选）

- 训练resnet18运行 `python train.py`
- 如果训练PP-HGNetV2-B4运行`python train_paddle.py`
- 训练d-fine参照原项目，一个模型拿下，比ddddocr+相似性检测资源开销小点

### 4-a.PP-HGNetV2-B4模型和resnet模型转换为onnx（可选）

- 运行 `python convert.py`（自行进去修改需要转换的模型，一般是选loss小的）
- paddle模型转换要装paddle2onnx，详情参见https://www.paddlepaddle.org.cn/documentation/docs/guides/advanced/model_to_onnx_cn.html

### 4-b.d-fine转换为onnx（可选）

- 依原项目转换
- 推理时图像预处理应于训练时一致，d-fine仓库中onnx推理的预处理和训练不一致……
  
### 5-a.启动fastapi服务（必须要有训练完成的onnx格式模型）

运行 `python main.py`（默认用的paddle的onnx模型，如果要用resnet18可以自己改注释）或者`uvicorn main:app --host 0.0.0.0 --port 9645 --reload`

由于轨迹问题，可能会出现验证正确但是结果失败，所以建议增加retry次数，训练后的paddle模型正确率在99.9%以上

### 5-b.使用docker启动服务 

镜像地址为<span id="docker">luguoyixiazi/test_nine:25.7.2</span>

运行时只需指定绑定的port和两个环境变量`use_pdl`和`use_dfine`，1为启用模型，0为不启用，默认均启用，api端口为/pass_uni，必填参数gt、challenge，单独的pass_nine和pass_icon也写了，有更多可选参数

### 6.api调用

python调用如：

```python
import httpx

def game_captcha(gt: str, challenge: str):
	res = httpx.get("http://127.0.0.1:9645/pass_uni",params={'gt':gt,'challenge':challenge},timeout=10)
	# 或者依旧使用pass_nine路径：
	# res = httpx.get("http://127.0.0.1:9645/pass_nine",params={'gt':gt,'challenge':challenge,'use_v3_model':True,"save_result":False},timeout=10)
	datas = res.json()['data']
    if datas['result'] == 'success':
        return datas['validate']
    return None # 失败返回None 成功返回validate
```

在snap hutao中的服务端口为/pass_hutao,返回值已做对齐，填写api如`(http://127.0.0.1:9645/pass_hutao?gt={0}&challenge={1})`即可

具体调用代码看使用项目，此处示例仅为API url和参数示例

#### --宣传--

欢迎大家支持我的其他项目(搭配使用)喵~~~~~~~~
