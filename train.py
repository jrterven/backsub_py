import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from CDNet2014Dataset import CDNet2014Dataset, Rescale, ToTensor
from model1 import BackSubModel1
from data_utils import count_labels_distribution


def train(num_epochs, epochs_to_save, output_path, load_checkpoint, chk_num):
    """ Train model """
    '''
    LOAD DATASET
    '''
    train_data = CDNet2014Dataset(root_dir='/datasets/backsub/cdnet2014/dataset',
                                  category='baseline',
                                  train=True,
                                  transform=transforms.Compose([
                                           Rescale((240, 320)),
                                           ToTensor()
                                           ]))
    test_data = CDNet2014Dataset(root_dir='/datasets/backsub/cdnet2014/dataset',
                                 category='baseline',
                                 train=False,
                                 transform=transforms.Compose([
                                           Rescale((240, 320)),
                                           ToTensor()
                                           ]))

    train_loader = DataLoader(train_data, batch_size=100,
                              shuffle=False, num_workers=4)

    test_loader = DataLoader(test_data, batch_size=1,
                             shuffle=False, num_workers=4)

    print('Training dataset length:', len(train_data))
    print('Testing dataset length:', len(test_data))

    print('Calculating data statistics ...')
    bg_pix_count, fg_pix_count = count_labels_distribution(train_loader)
    print('Foreground pixel count:', fg_pix_count)
    print('Background pixel count:', bg_pix_count)
    foreground_weight = bg_pix_count/fg_pix_count
    print('Foreground weight:', foreground_weight)
    if torch.cuda.is_available():
        classes_weights = torch.Tensor([1.0, foreground_weight]).cuda()
    else:
        classes_weights = torch.Tensor([1.0, foreground_weight])

    train_loader = DataLoader(train_data, batch_size=1,
                              shuffle=True, num_workers=4)

    '''
    INSTANTIATE MODEL CLASS
    '''

    model = BackSubModel1()

    if torch.cuda.is_available():
        print('Using GPU:', torch.cuda.get_device_name(0))

        model.cuda()
    else:
        print('NO GPU DETECTED!')

    '''
    INSTANTIATE LOSS CLASS
    '''
    criterion = nn.NLLLoss(weight=classes_weights)

    '''
    STEP 6: INSTANTIATE OPTIMIZER CLASS
    '''
    learning_rate = 0.0001

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=0.9, weight_decay=0.0001)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    '''
    STEP 7: TRAIN THE MODEL
    '''
    if load_checkpoint:
        print('Loading checkpoint ...')
        model.load_state_dict(torch.load(load_checkpoint))
        iter = chk_num
    else:
        iter = 0

    print('Start training ...')
    for epoch in range(chk_num, num_epochs):
        exp_lr_scheduler.step()
        
        for i, sample_batch in enumerate(train_loader):
            images = sample_batch['image']
            labels = sample_batch['gt']

            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.long().squeeze(1).cuda())
            else:
                images = Variable(images)
                labels = Variable(labels.long().squeeze(1))

            optimizer.zero_grad()    # Clear gradients w.r.t. parameters
            outputs = model(images)  # Forward pass to get output/logits

            loss = criterion(outputs, labels)  # Compute Loss
            loss.backward()         # Getting gradients w.r.t. parameters
            optimizer.step()        # Updating parameters

            iter += 1

            # Print loss
            if iter % 100 == 0:
                # Print Loss and accuracy
                str_loss = 'Epoch: {0:3d}, Iteration: {1:4d}, Loss: {2:15.12f}'
                print(str_loss.format(epoch, iter, loss.data[0]))

        # Calculate accuracy after each epoch
        print('Calculating accuracy on the test set ...')
        correct = 0
        total = 0

        # Iterate through test dataset
        for test_batch in test_loader:
            images = test_batch['image']
            labels = test_batch['gt'].long()

            if torch.cuda.is_available():
                images = Variable(images.cuda())
            else:
                images = Variable(images)

            # Forward pass only to get logits/output
            outputs = model(images)

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

            # Total number of labels
            total += (labels.size(2) * labels.size(3))

            # Total correct predictions
            if torch.cuda.is_available():
                correct += (predicted.cpu() == labels.cpu()).sum()
            else:
                correct += (predicted == labels).sum()

        accuracy = 100 * correct / total

        # Print Loss and accuracy
        str_loss = 'Epoch: {0:3d}, Iteration: {1:8d}, '\
            'Loss: {2:15.12f}, Accuracy: {3:15.12f}'
        print(str_loss.format(epoch, iter, loss.data[0], accuracy))

        # Save model after every epochs_to_save epochs
        if epoch % epochs_to_save == 0:
            chk_name = os.path.join(output_path,
                                    'model1_epoch' + str(epoch) + '.pkl')
            print('Saving checkpoint ',  chk_name)
            torch.save(model.state_dict(), chk_name)


def weighted_mse_loss(input, target, weights):
    out = (input-target)**2
    out = out * weights.expand_as(out)
    loss = out.sum(0)   # or sum over whatever dimensions
    return loss


if __name__ == "__main__":
    num_epochs = 100
    epochs_to_save = 10
    output_path = '/home2/backsub_repo/checkpoints'
    load_checkpoint = []
    chk_num = 0

    train(num_epochs, epochs_to_save, output_path, load_checkpoint, chk_num)
