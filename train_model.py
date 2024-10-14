import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
import yaml

from source.utils import iou_score, AverageMeter
from source.network import UNetPP

def train(deep_sup, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
    model.train()
    pbar = tqdm(total=len(train_loader))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for input, target, _ in train_loader:
        input = input.to(device)
        target = target.to(device)

        if deep_sup:
            outputs = model(input)
            loss = sum(criterion(output, target) for output in outputs) / len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg)])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg)])

def validate(deep_sup, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.to(device)
            target = target.to(device)

            if deep_sup:
                outputs = model(input)
                loss = sum(criterion(output, target) for output in outputs) / len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            postfix = OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg)])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg)])

if __name__ == "__main__":
    # Load config and datasets
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    log_path = config["log_path"]
    model_path = config["model_path"]
    epochs = config["epochs"]

    # Load preprocessed datasets
    train_dataset = pd.read_pickle("train_dataset.pkl")
    val_dataset = pd.read_pickle("val_dataset.pkl")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=16, shuffle=False, drop_last=False)

    model = UNetPP(1, 3, True)
    if torch.cuda.is_available():
        model.cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)

    log = OrderedDict([('epoch', []), ('loss', []), ('iou', []), ('val_loss', []), ('val_iou', [])])
    best_iou = 0
    trigger = 0

    for epoch in range(epochs):
        print(f'Epoch [{epoch}/{epochs}]')
        train_log = train(True, train_loader, model, criterion, optimizer)
        val_log = validate(True, val_loader, model, criterion)

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f' % (
            train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])

        pd.DataFrame(log).to_csv(log_path, index=False)

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), model_path)
            best_iou = val_log['iou']
            print("=> saved best model")
