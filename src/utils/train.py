import torch
import pandas as pd
import torch.backends.cudnn as cudnn
from dataset import*
from model import*
import wandb
from collections import defaultdict


all_labels = []
all_labels_df = pd.read_csv("DATASET/allAnnotations.csv", sep=';')
all_labels_df['Annotation tag'].value_counts()
with open('RESOURCES/label_map.json', 'r') as j:
    label_to_int = json.load(j)
int_to_label = {v: k for k, v in label_to_int .items()}


wandb_args = dict(
    api_key='dd3718ea0cc7a1c78b861f18150631f9fe2856c8',      
    entity='rahaf-abu-hara',        # Your W&B username
    project='Perception Project_5',
)


config = dict(
    lr=3e-4, wt=5e-4,
    epochs=30, decay_lr={10: 3e-5, 15: 3e-6, 20: 3e-7},
    momentum=0.9, batch_size=4
)

os.environ['WANDB_API_KEY'] = wandb_args['api_key']
wandb_logger = wandb.init(
    entity=wandb_args['entity'], project=wandb_args['project'],
    config=config, resume=True,
)
torch.cuda.empty_cache()


def train(data_loader, model, criterion, optimizer, loss_history):
    """
    One epoch's training.
    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """

    for phase in ['train', 'valid']:
        if phase == 'train':
                model.train()
        else:
                model.eval()
        losses = AverageMeter()  # loss

        # Batches
        for i, (images, boxes, labels) in enumerate(data_loader[phase]):

            # Move to default device
            images = images.to(device)  # (batch_size (N), 3, 600, 600)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            # zero the parameter gradients
            optimizer.zero_grad()
            # Forward prop.
            predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

            # Loss
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                # Update model
                optimizer.step()

            losses.update(loss.item(), images.size(0))
            if i % 100 == 0:
                print(losses.avg)
        loss_history[phase] = losses.avg

    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored
    return loss_history


# Data parameters
data_folder = './'  # folder with data files

# Model parameters
n_classes = len(label_to_int)  # number of different types of objects
# Learning parameters
checkpoint = "./checkpoint.pth.tar"  # path to model checkpoint, None if none
batch_size = config['batch_size']  # batch size
epochs = config['epochs']  # number of epochs to train
workers = 0  # number of workers for loading data in the DataLoader
print_freq = 100  # print training status every __ batches, 0: print only after every epoch
lr = config['lr']  # learning rate
decay_lr = config['decay_lr']  # decay learning rate after these many epochs
momentum = config['momentum']  # momentum
weight_decay = config['wt']  # weight decay
cudnn.benchmark = True


def main():
    """
    Training.
    """
    global start_epoch, epoch, checkpoint, decay_lr

    # Initialize model or load checkpoint
    if checkpoint is None:

        start_epoch = 0
        model = SSD600(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders
    train_dataset = LISADataset(data_folder,
                                split='train')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here
    valid_dataset = LISADataset(data_folder,
                                split='validation')

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here
    dataloaders = {'train': train_loader, 'valid': valid_loader}

    loss_history = defaultdict()

    # Epochs
    for epoch in range(start_epoch, epochs):
        if epoch in decay_lr.keys():
            adjust_learning_rate(optimizer, decay_lr[epoch])

        # One epoch's training
        loss_history = train(data_loader=dataloaders,
                             model=model,
                             criterion=criterion,
                             optimizer=optimizer,
                             epoch=epoch, loss_history=loss_history)

        if print_freq == 0:
            for key, value in loss_history.items():
                wandb_logger.log({
                    key+'_loss': value,
                }, step=epoch)
                print('{} Loss: {:.4f}'.format(key.capitalize(), value))
                train_log = open('./train.log', 'a')
                train_log.write('{} Loss: {:.4f}'.format(key.capitalize(), value))
                train_log.close()

        # Save checkpoint
        if True:
            save_checkpoint(epoch, model, optimizer)
    return loss_history, model


history, model = main()
wandb_logger.finish()
