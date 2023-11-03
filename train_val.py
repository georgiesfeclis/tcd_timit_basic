import torch


def train(model, device, lossFunction, lossList, trainLoader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target, _) in enumerate(trainLoader):
        data, target = data.to(device).type(torch.float32), target.to(device).type(torch.float32)
        optimizer.zero_grad()

        output = model(data)

        loss = lossFunction(output, target)
        loss.backward()
        lossList.append(loss.item())

        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train\t Epoch: {:3} [{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(trainLoader.dataset),
                100. * batch_idx / len(trainLoader),
                loss.item())
            )
    return lossList

def validate(model, device, lossFunction, lossList, validLoader, epoch):
    model.eval()
    for batch_idx, (features, target, _) in enumerate(validLoader):
        features = features.to(device).type(torch.float32)
        target = target.to(device).type(torch.float32)

        output = model(features)
        loss = lossFunction(output, target)
        lossList.append(loss.item())

        if batch_idx % 100 == 0:
            print('Valid\t Epoch: {:3} [{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(features),
                len(validLoader.dataset),
                100. * batch_idx / len(validLoader),
                loss.item())
            )
    return lossList

