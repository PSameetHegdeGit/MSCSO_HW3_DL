import torch
import numpy as np

from .models import FCN, save_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb

import torch.nn.functional as F

def print_dimensions(name):
    def hook(model, input, output):
        if "Sequential" not in str(model):
            print(model, output.shape)
    return hook



def train(args):
    from os import path
    model = FCN(base_channels=32)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(f'device: {device}')

    model.to(device)
    if args.continue_training:
        from os import path
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % args.model)))

    if args.continue_training:
        from os import path
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % args.model)))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    #invert the class distribution
    inverse_frequencies = np.array(DENSE_CLASS_DISTRIBUTION)**(-0.1)
    class_weights = torch.tensor(inverse_frequencies/np.sum(inverse_frequencies), dtype=torch.float32).to(device)
    print(f'class weights: {class_weights}')

    # Do we need to update the loss function used?
    loss = torch.nn.CrossEntropyLoss(weight=class_weights)


    dense_transforms_to_apply = dense_transforms.Compose(
        [
            dense_transforms.RandomHorizontalFlip(),
            dense_transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.2),
            dense_transforms.ToTensor()
        ]
    )

    train_data = load_dense_data('dense_data/train')
  #  train_data = load_dense_data('dense_data/train')
    valid_data = load_dense_data('dense_data/valid', transform=dense_transforms_to_apply)


    conf_matrix_train = ConfusionMatrix()
    conf_matrix_valid = ConfusionMatrix()

    for epoch in range(args.num_epoch):
        model.train()
        loss_vals, global_acc_vals, vacc_vals = [], [], []
        train_iou, valid_iou = [], []

        for img, label in train_data:
            img, label = img.to(device), label.long().to(device)
            # for name, module in model.named_modules():
            #     module.register_forward_hook(print_dimensions(name))
            logit = model(img).to(device)
            #exit()

            loss_val = loss(logit, label)

            # get iou and global_accuracy TODO: use class accuracies
            conf_matrix_train.add(logit.argmax(1), label)
            global_acc_vals.append(conf_matrix_train.global_accuracy.detach().cpu().numpy())
            train_iou.append(conf_matrix_train.iou)

            loss_vals.append(loss_val.detach().cpu().numpy())
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()



        avg_loss = sum(loss_vals) / len(loss_vals)
        avg_global_acc = sum(global_acc_vals) / len(global_acc_vals)
        avg_train_iou = sum(train_iou) / len(train_iou)

        # TODO: Log loss and accuracy for train data
        train_logger.add_scalar('cross_entropy_loss', scalar_value=avg_loss, global_step=epoch)
        train_logger.add_scalar('global_accuracy', scalar_value=avg_global_acc, global_step=epoch)
        train_logger.add_scalar('training iou', scalar_value=avg_train_iou, global_step=epoch)

        model.eval()
        for img, label in valid_data:
            img, label = img.to(device), label.to(device)

            logit = model(img)

            # get properties of conf_matrix
            conf_matrix_valid.add(logit.argmax(1), label)
            vacc_vals.append(conf_matrix_valid.global_accuracy)
            valid_iou.append(conf_matrix_valid.iou)

            log(valid_logger, img, label, logit, epoch)

        avg_global_vacc = sum(vacc_vals) / len(vacc_vals)
        avg_vac_iou = sum(valid_iou) / len(valid_iou)

        # TODO: log validation accuracy
        valid_logger.add_scalar('global_accuracy', scalar_value=avg_global_vacc, global_step=epoch)
        valid_logger.add_scalar('valid iou', scalar_value=avg_vac_iou, global_step=epoch)

        # log(valid_logger, imgs_in_epoch, labels_in_epoch, valid_logits_in_epoch, epoch)
        print('epoch %-3d \t loss = %0.3f \t acc = %0.3f \t train iou = %0.3f \t val acc = %0.3f \t val iou = %0.3f' % (epoch, avg_loss, avg_global_acc, avg_train_iou, avg_global_vacc, avg_vac_iou))

        scheduler.step()

    save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-pt', '--prob_of_transformation', type=float, default=1)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-op', '--optimizer')

    args = parser.parse_args()
    train(args)
