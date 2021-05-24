import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset, BatchSampler, DataLoader

BATCH_NUM = 20
BATCH_SIZE = 16
EPOCH_NUM = 4

IMAGE_SIZE = 90000
CLASS_NUM = 10

USE_GPU = True # whether use GPU to run model

# define a random dataset
class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples

dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)

class SimpleNet(nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(IMAGE_SIZE, CLASS_NUM)

    def forward(self, image, label=None):
        return self.fc(image)

simple_net = SimpleNet()
opt = paddle.optimizer.SGD(learning_rate=1e-3,
                          parameters=simple_net.parameters())

loader = DataLoader(dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    drop_last=True,
                    num_workers=6)

for e in range(EPOCH_NUM):
    for i, (image, label) in enumerate(loader()):
        out = simple_net(image)
        loss = F.cross_entropy(out, label)
        avg_loss = paddle.mean(loss)
        avg_loss.backward()
        opt.minimize(avg_loss)
        simple_net.clear_gradients()
        print("Epoch {} batch {}: loss = {}".format(e, i, np.mean(loss.numpy())))