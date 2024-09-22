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

