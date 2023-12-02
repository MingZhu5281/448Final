import paddle
import numpy as np
import PIL.Image as Image
from paddle.vision.datasets import DatasetFolder
from paddle.vision import transforms
import matplotlib.pyplot as plt
import os
import time

# Data preprocessing
data_transforms = transforms.Compose([
    transforms.Resize(size=(100, 100)),
    transforms.Transpose(),
    transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)
])

# Dataset definition
class Fruits360(DatasetFolder):
    def __init__(self, path):
        super().__init__(path)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        img = Image.open(img_path)
        label = int(label)  # Ensure label is an integer
        return data_transforms(img), label

# Dataset paths
train_dataset_path = r"archive\fruits-360_dataset\fruits-360\Training"
test_dataset_path = r"archive\fruits-360_dataset\fruits-360\Test"

train_dataset = Fruits360(train_dataset_path)
test_dataset = Fruits360(test_dataset_path)

train_loader = paddle.io.DataLoader(train_dataset, batch_size=60, shuffle=True)
test_loader = paddle.io.DataLoader(test_dataset, batch_size=30, shuffle=False)

# RNN Model
class MyRNN(paddle.nn.Layer):
    def __init__(self):
        super(MyRNN, self).__init__()
        self.rnn = paddle.nn.LSTM(input_size=300, hidden_size=128, num_layers=2)  # Update input_size to 300
        self.linear1 = paddle.nn.Linear(in_features=128, out_features=256)
        self.drop1 = paddle.nn.Dropout(p=0.5)
        self.out = paddle.nn.Linear(in_features=256, out_features=131)

    def forward(self, x):
        batch_size = x.shape[0]
        lstm_outputs = []

        for i in range(batch_size):
            # Flatten color channels and add batch dimension
            img_tensor = x[i].unsqueeze(0)  # Shape: [1, 3, 100, 100]
            img_tensor = paddle.reshape(img_tensor, [1, 100, -1])  # Flatten color channels: Shape [1, 100, 300]
            img_tensor = paddle.transpose(img_tensor, [1, 0, 2])  # Transpose to (100, 1, 300)

            # Process through the LSTM
            lstm_out, _ = self.rnn(img_tensor)
            lstm_out = lstm_out[-1, :, :]  # Select the last output of the sequence
            lstm_outputs.append(lstm_out)

        # Aggregate outputs for the batch
        x = paddle.concat(lstm_outputs, axis=0)

        x = self.linear1(x)
        x = paddle.nn.functional.relu(x)
        x = self.drop1(x)
        x = self.out(x)
        return x

# Instantiate the RNN model
my_rnn_model = MyRNN()

# Wrap the model with paddle.Model
model = paddle.Model(my_rnn_model)

# Prepare the model
model.prepare(
    optimizer=paddle.optimizer.Adam(learning_rate=1e-3, parameters=model.parameters()),
    loss=paddle.nn.CrossEntropyLoss(),
    metrics=[paddle.metric.Accuracy()]
)

# Model training
model.fit(train_loader,
          epochs=30,
          batch_size=60,
          verbose=1)

# Model evaluation
model.evaluate(test_loader, batch_size=30, verbose=1)

# Save model
model.save('MyRNN')  # save for training
model.save('MyRNN', False)  # save for inference

# Inference function
def infer_img(path, model_file_path, use_gpu):
    img = Image.open(path)
    plt.imshow(img)
    plt.show()

    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
    model = paddle.jit.load(model_file_path)
    model.eval()

    infer_imgs = [data_transforms(img)]
    infer_imgs = np.array(infer_imgs)
    label_list = test_dataset.classes

    for data in infer_imgs:
        dy_x_data = np.array(data).astype('float32')
        dy_x_data = dy_x_data[np.newaxis, :, :, :]
        img = paddle.to_tensor(dy_x_data)
        out = model(img)
        lab = np.argmax(out.numpy())
        print(f"Sample: {path}, Predicted as: {label_list[lab]}")

# Inference on images
image_path = []
for root, dirs, files in os.walk('work/'):
    for f in files:
        image_path.append(os.path.join(root, f))

for path in image_path:
    infer_img(path=path, use_gpu=True, model_file_path="MyRNN")
    time.sleep(0.5)
