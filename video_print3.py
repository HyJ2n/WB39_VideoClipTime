import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from pycocotools.coco import COCO
import random
import time

# Bottleneck Block
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=0.5):
        super(Bottleneck, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()  # Use SiLU activation function

    def forward(self, x):
        identity = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.activation(out)

class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_bottlenecks):
        super(CSPBlock, self).__init__()
        self.downsample_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.split_conv0 = nn.Conv2d(out_channels, out_channels // 2, kernel_size=1, stride=1, padding=0)
        self.split_conv1 = nn.Conv2d(out_channels, out_channels // 2, kernel_size=1, stride=1, padding=0)
        self.blocks_conv = nn.Sequential(*[
            Bottleneck(out_channels // 2, out_channels // 2)
            for _ in range(num_bottlenecks)
        ])
        self.concat_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.activation = nn.SiLU()  # Use SiLU activation function

    def forward(self, x):
        x = self.downsample_conv(x)
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)
        x = torch.cat((x1, x0), dim=1)
        x = self.concat_conv(x)
        return self.activation(x)

class CSPDarknet53(nn.Module):
    def __init__(self):
        super(CSPDarknet53, self).__init__()
        self.stem_conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.stage1 = CSPBlock(32, 64, 1)
        self.stage2 = CSPBlock(64, 128, 2)
        self.stage3 = CSPBlock(128, 256, 8)
        self.stage4 = CSPBlock(256, 512, 8)
        self.stage5 = CSPBlock(512, 1024, 4)

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.stage1(x)
        x1 = self.stage2(x)
        x2 = self.stage3(x1)
        x3 = self.stage4(x2)
        x4 = self.stage5(x3)
        return x4, x3, x2

class PANet(nn.Module):
    def __init__(self, in_channels):
        super(PANet, self).__init__()
        self.reduce_layer1 = nn.Conv2d(in_channels[0], in_channels[1], kernel_size=1, stride=1, padding=0)
        self.reduce_layer2 = nn.Conv2d(in_channels[1], in_channels[2], kernel_size=1, stride=1, padding=0)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.panet_layer1 = nn.Conv2d(in_channels[1], in_channels[1], kernel_size=3, stride=1, padding=1)
        self.panet_layer2 = nn.Conv2d(in_channels[2], in_channels[2], kernel_size=3, stride=1, padding=1)
        self.activation = nn.SiLU()  # Use SiLU activation function

    def forward(self, x):
        x4, x3, x2 = x
        x3_in = self.reduce_layer1(x4)
        x3_in = self.upsample(x3_in)
        x3 = x3 + x3_in
        x2_in = self.reduce_layer2(x3)
        x2_in = self.upsample(x2_in)
        x2 = x2 + x2_in
        x3 = self.panet_layer1(x3)
        x2 = self.panet_layer2(x2)
        return x2, x3, x4

class YOLOv8Head(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(YOLOv8Head, self).__init__()
        self.num_classes = num_classes
        self.pred_conv = nn.Conv2d(in_channels, num_classes + 4 + 1, kernel_size=1, stride=1, padding=0)  # class + bbox + confidence
        self.activation = nn.Sigmoid()  # Use Sigmoid activation function for output

    def forward(self, x):
        out = self.pred_conv(x)
        return self.activation(out)

class YOLOv8(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv8, self).__init__()
        self.backbone = CSPDarknet53()
        self.neck = PANet([1024, 512, 256])
        self.head0 = YOLOv8Head(num_classes, 256)
        self.head1 = YOLOv8Head(num_classes, 512)
        self.head2 = YOLOv8Head(num_classes, 1024)
    
    def forward(self, x):
        x4, x3, x2 = self.backbone(x)
        x2, x3, x4 = self.neck([x4, x3, x2])
        out0 = self.head0(x2)
        out1 = self.head1(x3)
        out2 = self.head2(x4)
        #print(out0.shape)
        #print(out1.shape)
        #print(out2.shape)
        return [out0, out1, out2]
    
def bbox_iou(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
        b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
        b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    return inter_area / (b1_area + b2_area - inter_area + 1e-16)

def bbox_ciou(box1, box2):
    """
    Returns the Complete IoU (CIoU) between box1 and box2.
    """
    # Calculate the IoU
    iou = bbox_iou(box1, box2, x1y1x2y2=False)
    
    # Calculate the center distance
    center_distance = torch.sum((box1[..., :2] - box2[..., :2]) ** 2, axis=-1)
    
    # Calculate the enclosing box
    enclose_x1 = torch.min(box1[..., 0] - box1[..., 2] / 2, box2[..., 0] - box2[..., 2] / 2)
    enclose_y1 = torch.min(box1[..., 1] - box1[..., 3] / 2, box2[..., 1] - box2[..., 3] / 2)
    enclose_x2 = torch.max(box1[..., 0] + box1[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2)
    enclose_y2 = torch.max(box1[..., 1] + box1[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2)
    enclose_diagonal = torch.sum((enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2, axis=-1)
    
    # Calculate the aspect ratio
    ar_gt = box2[..., 2] / (box2[..., 3] + 1e-16)  # Add small value to avoid division by zero
    ar_pred = box1[..., 2] / (box1[..., 3] + 1e-16)  # Add small value to avoid division by zero
    ar_loss = 4 / (torch.pi ** 2) * torch.pow(torch.atan(ar_gt) - torch.atan(ar_pred), 2)
    
    # Calculate the CIoU
    ciou = iou - (center_distance / (enclose_diagonal + 1e-16)) - ar_loss
    
    # Print intermediate values for debugging
    #print(f'IOU: {iou}')
    #print(f'Center Distance: {center_distance}')
    #print(f'Enclose Diagonal: {enclose_diagonal}')
    #print(f'AR Loss: {ar_loss}')
    #print(f'CIoU: {ciou}')
    
    return ciou

def distribution_focal_loss(pred, target):
    """
    Distribution Focal Loss 계산 함수
    Args:
        pred: 모델의 예측 값, shape: (N, C)
        target: 실제 타겟 값, shape: (N,)
    Returns:
        dfl_loss: 계산된 Distribution Focal Loss
    """
    # 예측 값과 타겟 값을 비교하여 손실 계산
    loss = F.cross_entropy(pred, target, reduction='none')
    return loss.mean()

class YoloLoss(nn.Module):
    def __init__(self, num_classes=80, lambda_box=7.5, lambda_obj=1.0, lambda_noobj=0.5, lambda_cls=0.5, lambda_dfl=1.5):
        super(YoloLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls
        self.lambda_dfl = lambda_dfl
        #self.bce_loss = nn.BCELoss(reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum') # 변경된 부분
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, predictions, targets):
        total_loss = 0
        batch_size = predictions[0].shape[0]

        for i, prediction in enumerate(predictions):
            #batch_size, channels, grid_size, _ = prediction.shape
            #target = targets[i].view(batch_size, channels, grid_size, grid_size)
            target = targets[i].permute(0, 2, 3, 1)
            prediction = predictions[i].permute(0, 2, 3, 1)
            obj_mask = target[..., 4] > 0
            noobj_mask = target[..., 4] == 0

            # Debugging: print the number of objects and no objects
            #print(f'Objects: {obj_mask.sum().item()}, No Objects: {noobj_mask.sum().item()}')

            coord_loss = torch.tensor(0.0, device=prediction.device)
            dfl_loss = torch.tensor(0.0, device=prediction.device)
            if obj_mask.sum() > 0:
                pred_box = prediction[obj_mask][:, :4]
                target_box = target[obj_mask][:, :4]
                #print(f'pred_box: {pred_box}')
                #print(f'target_box: {target_box}')
                ciou = bbox_ciou(pred_box, target_box)
                coord_loss = torch.mean(1 - ciou)

                #if i == 2:
                    #print(f'coord_loss: {coord_loss}')

                # dfl_loss 계산
                pred_dfl = prediction[obj_mask][:, 4:]  # 모델의 분포 예측 값 (N, C)
                target_dfl = target[obj_mask][:, 4:]  # 실제 타겟 값 (N, C)
                dfl_loss = distribution_focal_loss(pred_dfl, target_dfl)

            obj_loss = torch.tensor(0.0, device=prediction.device)
            noobj_loss = torch.tensor(0.0, device=prediction.device)
            if obj_mask.sum() > 0:
                pred_obj = prediction[obj_mask][:, 4]
                target_obj = target[obj_mask][:, 4]
                #print(f'pred_obj: {pred_obj}')
                #print(f'target_obj: {target_obj}')
                obj_loss = self.bce_loss(pred_obj, target_obj)
                #if i == 2:
                    #print(f'obj_loss: {obj_loss}')
            if noobj_mask.sum() > 0:
                pred_noobj = prediction[noobj_mask][:, 4]
                target_noobj = target[noobj_mask][:, 4]
                #print(f'pred_noobj: {pred_noobj}')
                #print(f'target_noobj: {target_noobj}')
                noobj_loss = self.bce_loss(pred_noobj, target_noobj)
                #if i == 2:
                    #print(f'noobj_loss: {noobj_loss}')

            class_loss = torch.tensor(0.0, device=prediction.device)
            if obj_mask.sum() > 0:
                # Extract the class targets
                class_targets = target[obj_mask][:, 5:].argmax(dim=-1)
                pred_classes = prediction[obj_mask][:, 5:(5 + self.num_classes)]

                #print(f'class_targets: {class_targets}')
                #print(f'pred_classes: {pred_classes}')

                # Ensure the correct shape and range for class targets
                if class_targets.max().item() >= self.num_classes or class_targets.min().item() < 0:
                    raise ValueError(f"Class targets out of range: {class_targets}")

                # Compute class loss using CrossEntropyLoss with raw logits
                class_loss = self.ce_loss(pred_classes, class_targets)
                
                # Debugging: print the class loss
                #if i == 2:
                    #print(f'Class Loss: {class_loss.item()}')

            if obj_mask.sum() > 0:
                total_loss += (self.lambda_box * coord_loss + self.lambda_obj * obj_loss + self.lambda_cls * class_loss + self.lambda_dfl * dfl_loss)
            total_loss += self.lambda_noobj * noobj_loss

        return total_loss / batch_size
    
# Dataset and DataLoader
class COCODataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform, S=[80, 40, 20], B=2, C=91, max_bboxes=8, subset_size=10000):
        self.coco = COCO(annotation_file)
        self.img_dir = img_dir
        self.img_ids = self.coco.getImgIds()
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        self.max_bboxes = max_bboxes

        if subset_size is not None:
            self.img_ids = random.sample(self.img_ids, subset_size)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]

        if len(anns) == 0 or len(anns) > self.max_bboxes:
            return self.__getitem__((index + 1) % len(self.img_ids))

        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        original_width, original_height = image.shape[1], image.shape[0]

        bboxes = []
        class_labels = []

        for ann in anns:
            bbox = ann['bbox']
            x, y, width, height = bbox

            cx = (x + width / 2) / original_width
            cy = (y + height / 2) / original_height
            width /= original_width
            height /= original_height

            bboxes.append([cx, cy, width, height])
            class_labels.append(ann['category_id'])

        # Update transform with current image size
        current_transform = get_train_transform()
        if self.transform:
            try:
                image, bboxes, class_labels = apply_transform(image, bboxes, class_labels, current_transform)
            except Exception as e:
                print(f"Error applying transform: {e}")
                return self.__getitem__((index + 1) % len(self.img_ids))

        targets = [np.zeros((5 + self.C, S, S)) for S in self.S]

        for bbox, label in zip(bboxes, class_labels):
            cx, cy, width, height = bbox

            for i, S in enumerate(self.S):
                cell_x = int(cx * S)
                cell_y = int(cy * S)

                if 0 <= cell_x < S and 0 <= cell_y < S:
                    if targets[i][4, cell_y, cell_x] == 0:
                        targets[i][:4, cell_y, cell_x] = np.array([cx, cy, width, height])
                        targets[i][4, cell_y, cell_x] = 1
                        targets[i][5 + label, cell_y, cell_x] = 1

        targets = [torch.tensor(target, dtype=torch.float32) for target in targets]
        return image, targets
        
# Training and Evaluation Functions
def train(model, dataloader, optimizer, criterion, device, epoch, scheduler, save_path='yololv8_model.pth', warmup_steps=1000, warmup_lr=0.1):
    model.train()
    total_loss = 0
    scaler = torch.cuda.amp.GradScaler() #

    for batch_idx, (images, targets) in enumerate(dataloader):
        images, targets = images.to(device), [t.to(device) for t in targets]
        # Learning rate warm-up
        if epoch == 0 and batch_idx < warmup_steps:
            warmup_factor = warmup_lr + (1.0 - warmup_lr) * (batch_idx / warmup_steps)
            lr = optimizer.param_groups[0]['lr'] * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f'Train Epoch [{epoch+1}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')
        
        if batch_idx % 600 == 0 and batch_idx != 0:
            # 각 에포크가 끝날 때 1분 대기
            print(f'batch_idx {batch_idx} completed. Waiting for 1 minute before starting the next epoch.')
            time.sleep(300)  # 1분 동안 대기

    if scheduler:
        scheduler.step()

    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

    return total_loss / len(dataloader)

def non_max_suppression(prediction, conf_threshold=0.5, iou_threshold=0.5):
    if len(prediction) == 0:
        return []
    
    prediction = [pred for pred in prediction if pred[1] > conf_threshold]
    if len(prediction) == 0:
        return []

    prediction = sorted(prediction, key=lambda x: x[1], reverse=True)
    filtered_preds = []

    while prediction:
        best_pred = prediction.pop(0)
        filtered_preds.append(best_pred)
        prediction = [pred for pred in prediction if calculate_iou(best_pred[2:], pred[2:]) < iou_threshold]

    return filtered_preds

def calculate_iou(box1, box2):
    box1 = [float(coord) for coord in box1]
    box2 = [float(coord) for coord in box2]
    
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / (union_area + 1e-16)

def get_batch_statistics(outputs, targets, iou_threshold):
    batch_metrics = []
    for output, target in zip(outputs, targets):
        pred_boxes = []
        target_boxes = []

        output = output.permute(0, 2, 3, 1)  # Permute to (batch, grid_y, grid_x, channels)
        batch_size, grid_y, grid_x, _ = output.shape

        for i in range(batch_size):
            pred_confidences = output[i, ..., 4].contiguous().view(-1)
            pred_classes = torch.argmax(output[i, ..., 5:], dim=-1).contiguous().view(-1)
            pred_boxes_coords = output[i, ..., :4].contiguous().view(-1, 4)

            target_confidences = target[i, ..., 4].contiguous().view(-1)
            target_classes = torch.argmax(target[i, ..., 5:], dim=-1).contiguous().view(-1)
            target_boxes_coords = target[i, ..., :4].contiguous().view(-1, 4)

            for j in range(pred_boxes_coords.shape[0]):
                if pred_confidences[j] > 0.5:
                    pred_boxes.append([pred_classes[j].item(), pred_confidences[j].item()] + pred_boxes_coords[j].tolist())
                if target_confidences[j] > 0:
                    target_boxes.append([target_classes[j].item()] + target_boxes_coords[j].tolist())

        if len(pred_boxes) > 0:
            pred_boxes = non_max_suppression(pred_boxes, conf_threshold=0.5, iou_threshold=iou_threshold)

        if len(pred_boxes) == 0 or len(target_boxes) == 0:
            batch_metrics.append([[], [], []])
            continue

        true_positives = np.zeros(len(pred_boxes))

        for pred_i, pred in enumerate(pred_boxes):
            pred_box = pred[2:]
            pred_class = pred[0]

            ious = [calculate_iou(pred_box, target[1:]) for target in target_boxes if target[0] == pred_class]
            if len(ious) == 0:
                continue
            best_iou = max(ious)
            best_target_idx = ious.index(best_iou)

            if best_iou > iou_threshold:
                true_positives[pred_i] = 1
                del target_boxes[best_target_idx]

        batch_metrics.append([true_positives, [pred[1] for pred in pred_boxes], [pred[0] for pred in pred_boxes]])

    return batch_metrics

def compute_mAP(outputs, targets, iou_threshold=0.5):
    batch_metrics = get_batch_statistics(outputs, targets, iou_threshold)

    true_positives = []
    scores = []
    labels = []

    for sample_metrics in batch_metrics:
        if sample_metrics:
            true_positives.extend(sample_metrics[0])
            scores.extend(sample_metrics[1])
            labels.extend(sample_metrics[2])

    if len(true_positives) == 0:
        return 0.0

    true_positives = np.array(true_positives)
    scores = np.array(scores)
    labels = np.array(labels)

    indices = np.argsort(-scores)
    true_positives = true_positives[indices]
    scores = scores[indices]
    labels = labels[indices]

    unique_classes = np.unique(labels)

    average_precisions = []

    for c in unique_classes:
        c_true_positives = true_positives[labels == c]
        c_scores = scores[labels == c]
        c_true_positives = np.cumsum(c_true_positives)
        c_false_positives = np.cumsum(1 - c_true_positives)

        recalls = c_true_positives / (c_true_positives[-1] + 1e-16)
        precisions = c_true_positives / (c_true_positives + c_false_positives + 1e-16)

        average_precision = np.trapz(precisions, recalls)
        average_precisions.append(average_precision)

    mAP = np.mean(average_precisions) if len(average_precisions) > 0 else 0.0
    return mAP

def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    total_mAP = 0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images, targets = images.to(device), [t.to(device) for t in targets]
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            mAP = compute_mAP(outputs, targets)
            total_mAP += mAP

            print(f'Validation Epoch [{epoch+1}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}, mAP: {mAP}')

    avg_loss = total_loss / len(dataloader)
    avg_mAP = total_mAP / len(dataloader)
    return avg_loss, avg_mAP

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Function to visualize a sample
def visualize_sample(dataset, index):
    image, targets = dataset[index]
    image = image.permute(1, 2, 0).numpy() * 255  # Convert to HWC format

    fig, ax = plt.subplots(1)
    ax.imshow(image.astype(np.uint8))

    for level, target in enumerate(targets):
        target = target.numpy()
        grid_size = target.shape[1]

        for y in range(grid_size):
            for x in range(grid_size):
                if target[4, y, x] > 0:  # Object present
                    bx = target[0, y, x] * 640
                    by = target[1, y, x] * 640
                    bw = target[2, y, x] * 640
                    bh = target[3, y, x] * 640
                    rect = patches.Rectangle((bx - bw / 2, by - bh / 2), bw, bh, linewidth=1, edgecolor='b', facecolor='none')
                    ax.add_patch(rect)
                    class_id = np.argmax(target[5:, y, x])
                    plt.text(bx, by, str(class_id), color='red', fontsize=8)

    plt.show()

def visualize_predictions(model, dataset, device, num_images=5, conf_threshold=0.5):
    model.eval()
    with torch.no_grad():
        for i in range(num_images):
            image, targets = dataset[i]
            image = image.unsqueeze(0).to(device)  # Batch 차원을 추가
            outputs = model(image)
            
            # 각 출력 텐서를 (배치 크기, 채널, 높이, 너비) 형태에서 (배치 크기, 높이, 너비, 채널) 형태로 변환
            outputs = [output.permute(0, 2, 3, 1) for output in outputs]
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            ax.imshow(image.cpu().squeeze().permute(1, 2, 0).numpy())

            for output in outputs:
                output = output.cpu().squeeze()
                grid_size = output.shape[0]
                
                for y in range(grid_size):
                    for x in range(grid_size):
                        confidence = output[y, x, 4].item()
                        if confidence > conf_threshold:
                            bbox = output[y, x, :4].numpy() * 640  # Original image 크기로 변환
                            cx, cy, w, h = bbox
                            rect = plt.Rectangle((cx - w / 2, cy - h / 2), w, h, linewidth=2, edgecolor='r', facecolor='none')
                            ax.add_patch(rect)

            plt.show()

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.08631),
        A.VerticalFlip(p=0.0),
        A.ColorJitter(brightness=0.13636, contrast=0.53554, saturation=0.53554, hue=0.01148, p=1.0),
        A.Resize(width=640, height=640), #416, 416
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def apply_transform(image, bboxes, class_labels, transform):
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    return transformed['image'], transformed['bboxes'], transformed['class_labels']

# Main
def main():
    train_annotation_file = r'D:\annotations\instances_train2017.json'
    train_img_dir = r'D:\train2017'
    val_annotation_file = r'D:\annotations\instances_val2017.json'
    val_img_dir = r'D:\val2017'

    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        #transforms.RandomHorizontalFlip(),  # 랜덤 수평 뒤집기
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 색상 조정
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 정규화
    ])

    train_dataset = COCODataset(train_annotation_file, train_img_dir, transform, subset_size=118287)
    val_dataset = COCODataset(val_annotation_file, val_img_dir, transform, subset_size=5000)
    
    # Visualize a few samples
    for i in range(3):
        visualize_sample(train_dataset, i)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLOv8(num_classes=91).to(device)

    save_path = r'C:\Users\bit\VSCode\yolol\yololv8_model_new_3.pth'

    # 저장된 모델이 있으면 불러오기
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print(f'Model loaded from {save_path}')
        
    # Adjust hyperparameters based on the provided best values
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.937, weight_decay=0.0005)  # Initial learning rate and momentum
    criterion = YoloLoss(num_classes=91, lambda_box=7.5, lambda_obj=1.0, lambda_noobj=0.5, lambda_cls=0.5, lambda_dfl=1.5)  # Initial loss weights
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    num_epochs = 100
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, optimizer, criterion, device, epoch, scheduler, save_path)
        #val_loss, val_accuracy = validate(model, val_dataloader, criterion, device, epoch)
        #print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
        print(f'Epoch {epoch+1}/{num_epochs}, Train AVG Loss: {train_loss:.4f}')
        
        # 각 에포크가 끝날 때 10분 대기
        print(f'Epoch {epoch+1} completed. Waiting for 1 minute before starting the next epoch.')
        time.sleep(600)  # 10분 동안 대기

        # 에포크가 끝날 때마다 예측 결과 시각화
        #visualize_predictions(model, val_dataset, device)

if __name__ == '__main__':
    main()