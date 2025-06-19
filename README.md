# 九宫格测试代码

## **本项目仅供学习交流使用，请勿用于商业用途，否则后果自负。**

## **本项目仅供学习交流使用，请勿用于商业用途，否则后果自负。**

## **本项目仅供学习交流使用，请勿用于商业用途，否则后果自负。**

## 参考项目

模型及V4数据集：https://github.com/taisuii/ClassificationCaptchaOcr

api：https://github.com/ravizhan/geetest-v3-click-crack

## 运行步骤

### 1.安装依赖（本地必选，使用docker跳至[5-b](#docker)）

（可选）a.如果要训练paddle的话还得安装paddlex及图像分类模块，安装看项目https://github.com/PaddlePaddle/PaddleX

（* 必选！）b.模型需要在项目目录下新建一个model文件夹，然后把模型文件放进去，具体命名可以是resnet18.onnx或者PP-HGNetV2-B4.onnx，默认使用PP-HGNetV2-B4模型，如果用resnet则use_v3_model设置为False，因为模型的输入输出不一样，可以自行修改

```
pip install -r requirements.txt
```

仅推理
```
pip install -r requirements_without_train.txt
```

### 2.自行准备数据集，V3和V4有区别（可选）

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

##### c. 如果要切V3的图用crop_image.py的crop_image_v3，切V4则使用crop_image，自行编写切图脚本

### 3.训练模型（可选）

- 训练resnet18运行 `python train.py`
- 如果训练PP-HGNetV2-B4运行`python train_paddle.py`

### 4.模型转换为onnx（可选）

- 运行 `python convert.py`（自行进去修改需要转换的模型，一般是选loss小的）
- paddle模型转换要装paddle2onnx，详情参见https://www.paddlepaddle.org.cn/documentation/docs/guides/advanced/model_to_onnx_cn.html

### 5-a.启动fastapi服务（必须要有训练完成的onnx格式模型）

运行 `python main.py`（默认用的paddle的onnx模型，如果要用resnet18可以自己改注释）

由于轨迹问题，可能会出现验证正确但是结果失败，所以建议增加retry次数，训练后的paddle模型正确率在99.9%以上
### 5-b.使用docker启动服务 

镜像地址为<span id="docker">luguoyixiazi/test_nine:25.6.20</span>

运行时只需指定绑定的port即可，api端口为/pass_nine，必填参数gt、challenge

### 6.api调用

python调用如：

```python
import httpx

def game_captcha(gt: str, challenge: str):
	res = httpx.get("http://127.0.0.1:9645/pass_nine",params={'gt':gt,'challenge':challenge,'use_v3_model':True,"save_result":False},timeout=10)
	datas = res.json()['data']
    if datas['result'] == 'success':
        return datas['validate']
    return None # 失败返回None 成功返回validate
```

具体调用代码看使用项目，此处示例仅为API url和参数示例

#### --宣传--

欢迎大家支持我的其他项目喵~~~~~~~~
