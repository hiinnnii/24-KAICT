from data.dataset import *
from model.model import *
from utils import *
from train import *
from sklearn.model_selection import train_test_split
from torchsummary import summary
import argparse

def main(args):

    print("Start Code")
    if args.model == 'simple':
        model = SimpleUNet()
    elif args.model == 'UNet_Xception':
        model = UNet_Xception(n_channels=1, n_classes=1)
    train_transform = Augmentation(data_type='train')
    val_transform = Augmentation(data_type='val')
    if args.mode == 'train':
        train_dataset = CLC_ClinicDBDataset(root_dir='/home/work/.hiinnnii/AIP2_unet_crack/unet/dataset/crack_segmentation_dataset', data_type='train', transform=train_transform, args=args)
        train_dataset, val_dataset = train_test_split(train_dataset, test_size =0.2, random_state = 42)
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=0)
        val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    else:
        # val_dataset = CLC_ClinicDBDataset(root_dir='dataset/cvc_clinicdb', data_type='val', transform=val_transform, args=args)
        test_dataset = CLC_ClinicDBDataset_test(root_dir='/home/work/.hiinnnii/AIP2_unet_crack/unet/dataset/crack_segmentation_dataset', data_type='test', transform=val_transform)
    
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle = False, num_workers =0)
    print("dataset loaded")
    #dataset = CLC_ClinicDBDataset(root_dir='dataset/cvc_clinicdb', transform=None)
    #dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # DataLoader에서 배치 가져오기
    #images, gt_images = next(iter(dataloader))
    
    # 가져온 배치 시각화
    #visualize_batch(images, gt_images)
    
    # 학습 시작
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == 'train' :
        infer_model = train_model(model, train_dataloader, val_dataloader, epochs=100, device=device, args=args)
        torch.save(infer_model.state_dict(), args.model_path)
        
        # Confirm saving
        if os.path.exists(args.model_path):
            print(f"Model successfully saved at {args.model_path}")
        else:
            print(f"Failed to save the model at {args.model_path}")

    elif args.mode == 'test' :
        model.load_state_dict(torch.load(args.model_path))
        print("model loaded")
        test_loss, test_score_dice, test_score_iou = test_model(model, test_dataloader, device=device, args=args)
        print(f"Final Test Loss : {test_loss:.4f}, Test {args.metric.capitalize()} Score_dice: {test_score_dice:.4f} Score_iou: {test_score_iou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a segmentation model.')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='Optimizer for training: "adam" or "sgd"')
    parser.add_argument('--model', type=str, default='simple', help='model for training')
    parser.add_argument('--loss', type=str, default='mse', help='model loss function')
    parser.add_argument('--scheduler', type=str, default='step', help='Learning rate scheduler')
    parser.add_argument('--img_size_h', type=int, default=224, help='img height')
    parser.add_argument('--img_size_w', type=int, default=224, help='img width')
    parser.add_argument('--metric', type=str, default='dice')
    parser.add_argument('--model_path', type = str, default = './model_base_1.pth')
    parser.add_argument('--mode', type=str, default='select train or test')
    args = parser.parse_args()
    main(args)
    
