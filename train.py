import torch
import torch.nn as nn
from tqdm import tqdm
from utils import *#save_loss_graph, iou_loss , visualize_predictions 

def train_model(model, train_loader, val_loader, epochs, device, args):
    model = model.to(device)
    
    if args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "binary_cross_entropy":
        criterion = nn.BCELoss()

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-5)

    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
    elif args.scheduler == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    train_losses = []
    val_losses = []
    train_scores = []
    val_scores = []

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        train_score = 0.0
        
        # Print the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1}/{epochs}], Learning Rate: {current_lr:.10f}")
        
        for images, gt_images in train_loader:
            images, gt_images = images.to(device), gt_images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            
            if args.loss == 'binary_cross_entropy':
                outputs = torch.sigmoid(outputs)
            
            loss = criterion(outputs, gt_images)
            
            if args.metric == 'iou':
                score = iou_score(outputs, gt_images)
            elif args.metric == 'dice':
                score = dice_score(outputs, gt_images)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_score += score.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_score = train_score / len(train_loader)
        train_losses.append(avg_train_loss)
        train_scores.append(avg_train_score)

        model.eval()
        val_loss = 0.0
        val_score = 0.0
        with torch.no_grad():
            for images, gt_images in val_loader:
                images, gt_images = images.to(device), gt_images.to(device)
                outputs = model(images)
                
                if args.loss == 'binary_cross_entropy':
                    outputs = torch.sigmoid(outputs)
                
                loss = criterion(outputs, gt_images)
                
                if args.metric == 'iou':
                    score = iou_score(outputs, gt_images)
                elif args.metric == 'dice':
                    score = dice_score(outputs, gt_images)

                val_loss += loss.item()
                val_score += score.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_score = val_score / len(val_loader)
        val_losses.append(avg_val_loss)
        val_scores.append(avg_val_score)

        print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, \
        Training {args.metric.capitalize()} Score: {avg_train_score:.4f}, Validation {args.metric.capitalize()} Score: {avg_val_score:.4f}")

        if args.scheduler == 'plateau':
            scheduler.step(avg_val_score)
        elif args.scheduler == 'steplr':
            scheduler.step()

        save_loss_graph(train_losses, val_losses, epoch + 1, f'output/loss_graph_epoch_{epoch + 1}.png')
        visualize_predictions(val_loader, model, device, epoch + 1)

    print('Finished Training')
    return model




def test_model(model, test_loader, device, args):
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    
    # Initialize criterion based on args.loss
    criterion = None
    if args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "binary_cross_entropy":
        criterion = nn.BCELoss()

    test_loss = 0.0
    test_score_iou = 0.0
    test_score_dice = 0.0
    print("test start")
    with torch.no_grad():  # Disable gradient calculation for testing
        for images, gt_images in tqdm(test_loader):
            images, gt_images = images.to(device), gt_images.to(device)
            outputs = model(images)

            if args.loss == 'binary_cross_entropy':
                outputs = torch.sigmoid(outputs)  # Apply sigmoid if using binary cross-entropy
            
            loss = criterion(outputs, gt_images)
            test_loss += loss.item()

            # Calculate the evaluation metric (e.g., IoU or Dice)
            #if args.metric == 'iou':
            #    score = iou_score(outputs, gt_images)
            #elif args.metric == 'dice':
            #    score = dice_score(outputs, gt_images)
            score_iou = iou_score(outputs, gt_images)
            score_dice = dice_score(outputs, gt_images)
            test_score_iou += score_iou.item()
            test_score_dice += score_dice.item()

    avg_test_loss = test_loss / len(test_loader)
    avg_test_score_iou = test_score_iou / len(test_loader)
    avg_test_score_dice = test_score_dice / len(test_loader)

    print(f"Test Loss: {avg_test_loss:.4f}, Score iou: {avg_test_score_iou:.4f} Score dice: {avg_test_score_dice:.4f}")

    return avg_test_loss, avg_test_score_iou, avg_test_score_dice