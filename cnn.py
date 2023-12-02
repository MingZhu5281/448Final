import paddle
import numpy as np
import PIL.Image as Image
from paddle.vision.datasets import DatasetFolder
from paddle.vision import transforms
import matplotlib.pyplot as plt
import os
import time


# Define data preprocessing operations, these transformations will be applied to each image
data_transforms = transforms.Compose([
    transforms.Resize(size=(100, 100)),  # Resize the image to 100x100 size
    transforms.Transpose(),  # Convert the image from HWC to CHW format
    transforms.Normalize(
        mean=[0, 0, 0],  # Set the normalization mean
        std=[255, 255, 255],  # Set the normalization standard deviation
        to_rgb=True)  # Ensure the image is in RGB format
])

class Fruits360(DatasetFolder):
    def __init__(self, path):
        # Initialize the parent class DatasetFolder, path is the path to the image dataset
        super().__init__(path)

    def __getitem__(self, index):
        # Get the image path and corresponding label by index
        img_path, label = self.samples[index]
        # Open the image using PIL library
        img = Image.open(img_path)
        # Convert the label to a numpy array and set its type to int64
        label = np.array([label]).astype(np.int64)
        # Return the preprocessed image and label
        return data_transforms(img), label

# Specify the paths for the training and test datasets
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


    # forward defines the execution logic of the network during actual runtime.
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

        # x = paddle.reshape(x, [-1, x.shape[1] * x.shape[2] * x.shape[3]]) 
        x = self.flatten(x)

        x = self.linear1(x)
        x = paddle.nn.functional.relu(x)
        x = self.drop1(x)

        x = self.out(x)

        return x

# Visualize the model structure
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

# Model training-related configuration, prepare the loss computation method, optimizer, and accuracy calculation method
model.prepare(paddle.optimizer.Adam(learning_rate=1e-3, parameters=model.parameters()),
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy())

# Model training
model.fit(train_dataset,
            epochs=5,
            batch_size=60,
            verbose=1)

# Model evaluation
model.evaluate(test_dataset, batch_size=30, verbose=1)

# Save model parameters
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

        # print(paddle.nn.functional.softmax(out)[0])

        lab = np.argmax(out.numpy())  #argmax(): returns the index of the largest number.
        print("样本: {},被预测为:{}".format(path, label_list[lab]))

    print("*********************************************")

    image_path = []

    for root, dirs, files in os.walk('work/'):
        # # Traverse the images in the work/ folder
        for f in files:
            image_path.append(os.path.join(root, f))

    for i in range(len(image_path)):
        # infer_img(path=image_path[i], use_gpu=True, model_file_path="MyCNN")
        infer_img(path=image_path[i], use_gpu=True, model_file_path="Hapi_MyCNN")
        time.sleep(0.5)