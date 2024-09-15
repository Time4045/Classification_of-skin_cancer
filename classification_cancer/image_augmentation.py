# import libriries
import os
import matplotlib.image as mpimg
import torchvision.transforms as transforms



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.RandomGrayscale(p=0.2),
    transforms.Normalize(mean=[0.01,0.02,0.04], std=[1.0,1.0,1.0]),
    transforms.ToPILImage()
])

data_path='/Users/maksimtrebusinin/Desktop/Cancer_data/Squamous cell carcinoma'
files=os.listdir(data_path)

img_num=0
for file in files[1:]:
    image=mpimg.imread(os.path.join(data_path,file))
    tr_img=transform(image)
    tr_img.save(os.path.join(data_path,'img'+str(img_num)+'.jpg'))
    img_num+=1


