import numpy as np
import os
import glob
import torch
import matplotlib.image as mpimg
from pathlib import Path
from skimage import transform
from sklearn.model_selection import train_test_split
data_path=Path('/Users/maksimtrebusinin/Desktop/Cancer_data')
list_dir=os.listdir(data_path)[1:]

X=list()
y=[0]*4000
index=0
for i, column_name in enumerate(list_dir):
    for img_path in glob.iglob(os.path.join(data_path, column_name, '*')):
        X.append(np.asarray(mpimg.imread(img_path)))
        y[index]=i
        index+=1
#0 is Healthy
#1 is Melanoma
#2 is Bascal cell carcioma
#3 is Squamous cell carcinoma
y=np.array(y)
transfrom_X=[transform.resize(image, (159,159,3)) for image in X]

# train and test selections
X_train, X_test, y_train, y_test = train_test_split(transfrom_X, y, test_size=0.25, stratify=y,
                                                    random_state=42)
X_train_tensor=torch.from_numpy(np.array(X_train))
X_test_tensor=torch.from_numpy(np.array(X_test))
y_train_tensor=torch.from_numpy(y_train)
y_test_tensor=torch.from_numpy(y_test)

X_train_tensor=X_train_tensor.reshape(3000, 3, 159, 159)
X_test_tensor=X_test_tensor.reshape(1000, 3, 159, 159)

class CancerNet(torch.nn.Module):
    def __init__(self):
        super(CancerNet,self).__init__()

        self.conv1=torch.nn.Conv2d(in_channels=3, out_channels=30, kernel_size=4)
        self.pool1=torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.act1=torch.nn.ReLU()

        self.conv2=torch.nn.Conv2d(in_channels=30, out_channels=60, kernel_size=3, padding=1)
        self.pool2=torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.act2=torch.nn.ReLU()

        self.conv3=torch.nn.Conv2d(in_channels=60, out_channels=120, kernel_size=2)
        self.pool3=torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.act3=torch.nn.ReLU()

        self.conv4=torch.nn.Conv2d(in_channels=120, out_channels=80, kernel_size=3, padding=1)
        self.pool4=torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.act4=torch.nn.ReLU()

        self.conv5=torch.nn.Conv2d(in_channels=80, out_channels=30, kernel_size=3, padding=1)
        self.act5=torch.nn.ReLU()

        self.fc1=torch.nn.Linear(9*9*30, 200)
        self.act6=torch.nn.ReLU()
        self.fc2=torch.nn.Linear(200, 4)
        self.sm=torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.act4(x)
        x = self.conv5(x)
        x = self.pool5(x)
        x = self.act5(x)

        x = x.view(x.size(0), x.size(1)*x.size(2)*x.size(3))

        x = self.fc1(x)
        x = self.act6(x)
        x = self.fc2(x)
        return x

    def inference(self, x):
        x=self.forward(x)
        x=self.sm(x)
        return x

cancer_net=CancerNet()

loss=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(cancer_net.parameters(), lr=0.001)

def train(X_tr,y_tr, epoch, batch_size, opt, Net, loss_v):
    for ep in range(epoch):
        order=np.random.permutation(len(X_tr))
        for start_index in range(0, len(X_tr), batch_size):
            opt.zero_grad()
            batch_indexes=order[start_index:start_index+batch_size]

            X_tr_batch=X_tr[batch_indexes]
            y_tr_batch=y_tr[batch_indexes]

            pred=Net.forward(X_tr_batch)
            loss_value=loss_v(pred, y_tr_batch)
            loss_value.backward()
            opt.step()

def predict(X_test):
    preds=cancer_net.forward(X_test)
    preds=preds.argmax(dim=1)
    preds=np.array(preds)
    return preds
