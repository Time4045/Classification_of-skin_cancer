import torch
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import cv2
import base64

class CancerNet(torch.nn.Module):
    def __init__(self):
        super(CancerNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=30, kernel_size=4)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.act1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(in_channels=30, out_channels=60, kernel_size=3, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.act2 = torch.nn.ReLU()

        self.conv3 = torch.nn.Conv2d(in_channels=60, out_channels=120, kernel_size=2)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.act3 = torch.nn.ReLU()

        self.conv4 = torch.nn.Conv2d(in_channels=120, out_channels=80, kernel_size=3, padding=1)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.act4 = torch.nn.ReLU()

        self.conv5 = torch.nn.Conv2d(in_channels=80, out_channels=30, kernel_size=3, padding=1)
        self.act5 = torch.nn.ReLU()

        # Добавляем Dropout после последней активации перед полносвязным слоем
        self.dropout1 = torch.nn.Dropout(p=0.5)  # 50% случайных обнулений

        self.fc1 = torch.nn.Linear(9 * 9 * 30, 200)
        self.act6 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(p=0.5)  # 50% Dropout на скрытом слое
        self.fc2 = torch.nn.Linear(200, 4)
        self.sm = torch.nn.Softmax(dim=1)

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
        x = self.act5(x)

        # Применяем Dropout после всех сверток и активаций
        x = self.dropout1(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))  # Преобразуем в вектор для подачи на fc

        x = self.fc1(x)
        x = self.act6(x)

        # Применяем Dropout после первого полносвязного слоя
        x = self.dropout2(x)

        x = self.fc2(x)
        return x

    def inference(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x


def predict(image_bytes):
    np_image = np.frombuffer(image_bytes, np.uint8)
    open_image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    if open_image is None:
        return "Ошибка при чтении изображения", None

    input_image_reshape = cv2.resize(open_image, (159, 159))
    image_norm = input_image_reshape / 255
    img_reshape = np.reshape(image_norm, (1, 3, 159, 159))
    img_reshape = img_reshape.astype(np.float32)
    img_reshape_tensor = torch.from_numpy(img_reshape)

    model = CancerNet()
    model.load_state_dict(torch.load('CancerNet7.pth', map_location=torch.device("cpu")))
    model.eval()
    with torch.no_grad():
        predictions = model.forward(img_reshape_tensor)
        predictions = predictions.argmax(dim=1).item()

        if predictions == 0:
            return 'Healthy (Вы здоровы! Поводов для беспокойства нет.)'
        elif predictions == 1:
            return 'Melanoma (за дополнительной информацией советуем обратиться к врачу)'
        elif predictions == 2:
            return 'Bascal cell carcinoma (за дополнительной информацией советуем обратиться к врачу)'
        else:
            return 'Squamous cell carcinoma (за дополнительной информацией советуем обратиться к врачу)'


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # Проверяем, был ли файл в запросе
        if "file" not in request.files:
            return "Файл отсутствует в запросе"

        file = request.files["file"]
        if file.filename == "":
            return "Файл не выбран"

        if file:
            # Предобработка изображения и вызов функции
            result = predict(file.read())
            return render_template("result.html", result=result)

    # Если не отправлен POST-запрос, отобразить форму загрузки
    return render_template("upload.html")


if __name__ == "__main__":
    app.run(debug=True)



