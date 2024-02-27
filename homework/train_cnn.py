from .models import CNNClassifier, save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb
import torch.nn.functional as F



def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'train')), tb.SummaryWriter(
        path.join(args.log_dir, 'test'))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    if args.continue_training:
        from os import path
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % args.model)))

    if args.optimizer == "adam":
        print("using adam")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  # seems to be pretty poor atm
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss = F.cross_entropy

    train_data = load_data('data/train', probability_threshold=args.prob_of_transformation, random_horizontal_flip=True,
                           random_color_jitter=True)
    valid_data = load_data('data/valid', probability_threshold=0)

    best_val_acc = float('-inf')
    counter = 0
    early_stopping = 10

    for epoch in range(args.num_epoch):
        model.train()
        loss_vals, acc_vals, vacc_vals = [], [], []
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            logit = model(img)

            loss_val = loss(logit, label)
            acc_val = accuracy(logit, label)

            loss_vals.append(loss_val.detach().cpu().numpy())
            acc_vals.append(acc_val.detach().cpu().numpy())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

        avg_loss = sum(loss_vals) / len(loss_vals)
        avg_acc = sum(acc_vals) / len(acc_vals)

        # TODO: Log loss and accuracy for train data
        train_logger.add_scalar('loss', scalar_value=avg_loss, global_step=epoch)
        train_logger.add_scalar('accuracy', scalar_value=avg_acc, global_step=epoch)

        model.eval()
        for img, label in valid_data:
            img, label = img.to(device), label.to(device)
            vacc_vals.append(accuracy(model(img), label).detach().cpu().numpy())

        avg_vacc = sum(vacc_vals) / len(vacc_vals)

        # TODO: log validation accuracy
        valid_logger.add_scalar('accuracy', scalar_value=avg_vacc, global_step=epoch)

        print('epoch %-3d \t loss = %0.3f \t acc = %0.3f \t val acc = %0.3f' % (epoch, avg_loss, avg_acc, avg_vacc))

        if avg_vacc > best_val_acc:
            best_val_acc = avg_vacc
            counter = 0
            save_model(model)
        else:
            counter += 1
            if counter >= early_stopping:
                print("best validation accuracy hit, set early stopping")
                break

        scheduler.step()


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

