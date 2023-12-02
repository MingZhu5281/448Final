import paddle
import numpy as np
import PIL.Image as Image
from paddle.vision.datasets import DatasetFolder
from paddle.vision import transforms
import matplotlib.pyplot as plt
import os
import time


# 定义数据预处理操作，这些变换将应用于每一张图片
data_transforms = transforms.Compose([
    transforms.Resize(size=(100, 100)),  # 将图片缩放至100x100大小
    transforms.Transpose(),  # 将图片从高度x宽度x通道数(HWC)格式转换为通道数x高度x宽度(CHW)格式
    transforms.Normalize(
        mean=[0, 0, 0],  # 设置归一化的均值
        std=[255, 255, 255],  # 设置归一化的标准差
        to_rgb=True)  # 确保图片为RGB格式
])

class Fruits360(DatasetFolder):
    def __init__(self, path):
        # 初始化父类DatasetFolder，path为图片数据集的路径
        super().__init__(path)

    def __getitem__(self, index):
        # 根据索引获取图片路径和对应的标签
        img_path, label = self.samples[index]
        # 使用Pillow库打开图片
        img = Image.open(img_path)
        # 将标签转换为numpy数组，并将其类型设置为int64
        label = np.array([label]).astype(np.int64)
        # 返回经过预处理的图片和标签
        return data_transforms(img), label

# 指定训练集和测试集的路径
train_dataset_path = r"archive\fruits-360_dataset\fruits-360\Training"
test_dataset_path = r"archive\fruits-360_dataset\fruits-360\Test"
# Adjust the class definition and initialization if necessary
train_dataset = Fruits360(train_dataset_path)
test_dataset = Fruits360(test_dataset_path)

train_loader = paddle.io.DataLoader(train_dataset, batch_size=60, shuffle=True)
test_loader = paddle.io.DataLoader(test_dataset, batch_size=30, shuffle=False)

class MyCNN(paddle.nn.Layer):
    def __init__(self):
        super(MyCNN,self).__init__()

        self.conv1 = paddle.nn.Conv2D(in_channels=3, out_channels=16, kernel_size=5, padding='SAME')
        self.pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2, padding='VALID')

        self.conv2 = paddle.nn.Conv2D(in_channels=16, out_channels=32, kernel_size=5, padding='SAME')
        self.pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2, padding='VALID')

        self.conv3 = paddle.nn.Conv2D(in_channels=32, out_channels=64, kernel_size=5, padding='SAME')
        self.pool3 = paddle.nn.MaxPool2D(kernel_size=2, stride=2, padding='VALID')

        self.conv4 = paddle.nn.Conv2D(in_channels=64, out_channels=128, kernel_size=5, padding='SAME')
        self.pool4 = paddle.nn.MaxPool2D(kernel_size=2, stride=2, padding='VALID')

        self.flatten = paddle.nn.Flatten()

        self.linear1 = paddle.nn.Linear(in_features=4608, out_features=256)
        self.drop1 = paddle.nn.Dropout(p=0.8)

        self.out = paddle.nn.Linear(in_features=256, out_features=131)


    # forward 定义执行实际运行时网络的执行逻辑
    def forward(self,x):
        # input.shape (batch_size, 3, 100, 100)
        x = self.conv1(x)
        x = paddle.nn.functional.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = paddle.nn.functional.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = paddle.nn.functional.relu(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = paddle.nn.functional.relu(x)
        x = self.pool4(x)

        # x = paddle.reshape(x, [-1, x.shape[1] * x.shape[2] * x.shape[3]]) #reshape2算子
        x = self.flatten(x) # Lite目前不支持该算子

        x = self.linear1(x)
        x = paddle.nn.functional.relu(x)
        x = self.drop1(x)

        x = self.out(x)

        return x

# 模型结构可视化
paddle.summary(MyCNN(), (1, 3, 100, 100))


# model = paddle.Model(MyCNN())
# Assuming your model is named MyCNN and you have defined it as shown in your script
model = MyCNN()

# Example input data to get the input shape
input_data = paddle.randn([60, 3, 100, 100])

# Initialize the Model with the input specification
# Initialize the Model with the input specification
model = paddle.Model(MyCNN(), inputs=[paddle.static.InputSpec(shape=[-1, 3, 100, 100], dtype='float32', name='x')])
# Continue with your existing code for model.prepare(), etc.

# 模型训练相关配置，准备损失计算方法，优化器和精度计算方法
model.prepare(paddle.optimizer.Adam(learning_rate=1e-3, parameters=model.parameters()),
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy())

# 模型训练
model.fit(train_dataset,
            epochs=5,
            batch_size=60,
            verbose=1)

# 模型预估
model.evaluate(test_dataset, batch_size=30, verbose=1)

# 保存模型参数
model.save('Hapi_MyCNN')  # save for training
model.save('Hapi_MyCNN', False)  # save for inference


def infer_img(path, model_file_path, use_gpu):

    img = Image.open(path)
    plt.imshow(img)
    plt.show()

    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
    model = paddle.jit.load(model_file_path)
    model.eval()

    infer_imgs = []
    infer_imgs.append(data_transforms(img))
    infer_imgs = np.array(infer_imgs)
    label_list = test_dataset.classes

    for i in range(len(infer_imgs)):
        data = infer_imgs[i]
        dy_x_data = np.array(data).astype('float32')
        dy_x_data = dy_x_data[np.newaxis,:, : ,:]
        img = paddle.to_tensor(dy_x_data)
        out = model(img)

        # print(paddle.nn.functional.softmax(out)[0]) # 手动在预测时加上softmax，输出score时比较直观。

        lab = np.argmax(out.numpy())  #argmax():返回最大数的索引
        print("样本: {},被预测为:{}".format(path, label_list[lab]))

    print("*********************************************")

    image_path = []

    for root, dirs, files in os.walk('work/'):
        # 遍历work/文件夹内图片
        for f in files:
            image_path.append(os.path.join(root, f))

    for i in range(len(image_path)):
        # infer_img(path=image_path[i], use_gpu=True, model_file_path="MyCNN")
        infer_img(path=image_path[i], use_gpu=True, model_file_path="Hapi_MyCNN")
        time.sleep(0.5)