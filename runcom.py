import os

# os.system('python main.py --mode train --model_path ./model_base_21.pth --optimizer rmsprop --model UNet_Xception --loss binary_cross_entropy --scheduler plateau')
os.system('python main.py --mode test --model UNet_Xception --model_path ./model_base_20.pth --optimizer rmsprop --loss binary_cross_entropy --metric dice') 
