import os

os.system('python main.py --mode train --model_path ./model_base_10.pth --optimizer rmsprop --model unet --loss binary_cross_entropy --scheduler steplr')
os.system('python main.py --mode test --model unet --model_path ./model_base_10.pth --optimizer rmsprop --loss binary_cross_entropy --metric dice') 