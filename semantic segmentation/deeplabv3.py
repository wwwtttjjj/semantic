import torch
import torch.nn as nn
import torchvision.transforms
import torchvision
import wandb
from deeplabv3plus.model import DeepLabV3Plus
from metrics import iou
from data import voc2012
import numpy
import torch.utils.data


if __name__ == '__main__':
    '''
    download dataset of Voc2012
    '''
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize([0.485, 0.465, 0.406],[0.229, 0.224, 0.225], inplace=True)
    ])
    
    Voc2012_train = voc2012.get_voc2012(require_data='train')
    Voc2012_val = voc2012.get_voc2012(require_data='trainval')
    train_iter = torch.utils.data.DataLoader(Voc2012_train, batch_size = 8, shuffle = True, drop_last = True)
    val_iter = torch.utils.data.DataLoader(Voc2012_val, batch_size = 8, shuffle = False, drop_last = True)

    '''
    set device,epoch,model and optimizer,scheduler
    '''
    device = torch.device('cuda:0')
    eopchs = 50
    model = DeepLabV3Plus(outstride=16, num_classes=21)
    optimizer = torch.optim.SGD([{'params':model.parameters(), 'lr' :1e-3},
                                ], lr = 1e-2, momentum=0.9,weight_decay=1e-4)
    loss = torch.nn.CrossEntropyLoss(reduction='none')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
    '''
    set wandb paramters
    '''
    wandb.init(project='semantic segmatation',entity='wtj')
    wandb.config = {
        'learning_rate':0.01,
        'epochs':50,
        'batch_size':8
    }

    model.to(device)
    model.train()
    accumulate = iou.SegClassIouAccumulator(num_classes=21)
    for eopch in range(eopchs):
        loss_arr = []
        for x_cpu, y_cpu in train_iter:
            optimizer.zero_grad()
            x = x_cpu.to(device)
            y = y_cpu.to(device)
            y_predict = model(x)

            train_loss = loss(y, y_predict).mean(1).mean(1).sum()
            train_loss.backward()
            loss_arr.append(train_loss.detach().cpu())
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_mean_loss = torch.tensor(loss_arr)
            for x_cpu, y_cpu in val_iter:
                x = x_cpu.to(device)
                y_predict = model(x).cpu()
                y_predict = torch.argmax(y_predict, dim = 1)
                label = y_cpu
                accumulate(label.numpy().astype(numpy.uint32), y_predict.numpy().astype(numpy.uint32))
            result = accumulate.json()
            wandb.log({
                'train_loss':train_mean_loss.mean(),'train_mAP':result['acc'],'mean_iou':result['iou'],
            })
            accumulate.reset()

        model.train()




    torch.save(model, '../.pth/semantic segmentation.pths')

