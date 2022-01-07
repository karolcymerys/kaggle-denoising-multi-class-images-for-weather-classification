import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

import helper
from model import CNNAutoEncoder

LR = 1e-3
EPOCHS = 100
BATCH_SIZE = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
dataset = ImageFolder('./dataset', transform=transform)
train_set, test_set = random_split(dataset, [1000, 125])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)


model = CNNAutoEncoder().to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)
criteria = nn.MSELoss()

loss_history = {
    'train_loss': [],
    'validation_loss': []
}
noise_factor = 0.1

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for features, _ in train_loader:
        features = features.to(device)
        input = features + noise_factor * torch.randn(*features.shape, device=device)
        optimizer.zero_grad()
        output = model.forward(input)
        loss = criteria(output, features)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * features.shape[0]

    model.eval()
    val_loss = 0.0
    for features, _ in test_loader:
        features = features.to(device)
        input = features + noise_factor * torch.randn(*features.shape, device=device)
        output = model.forward(input)
        loss = criteria(output, features)
        val_loss += loss.item() * features.shape[0]
    print(f'Epoch: {epoch+1}\t'
          f'Train loss: {train_loss/len(train_set)}\t'
          f'Validation loss: {val_loss/len(test_set)}\t')
    loss_history['train_loss'].append(train_loss/len(train_set))
    loss_history['validation_loss'].append(val_loss/len(test_set))
    if len(loss_history['validation_loss']) > 10 and \
            torch.mean(torch.tensor(loss_history['validation_loss'][-10:-1])).item() < 0.75*loss_history['validation_loss'][-1]:
        print('Training data overfitting detected. Finishing.')
        torch.save(model.state_dict(), 'model.pth')
        break

plt.plot(loss_history['train_loss'], label='Training loss')
plt.plot(loss_history['validation_loss'], label='Validation loss')
plt.legend(frameon=False)
plt.show()

model.eval()
for idx, (features, _) in enumerate(test_loader):
    features = features.to(device)
    input = features + noise_factor * torch.randn(*features.shape, device=device)
    output = model(input)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(helper.im_convert(input))
    ax[1].imshow(helper.im_convert(output))
    plt.show()
