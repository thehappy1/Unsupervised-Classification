from fpidataset import Fpidataset
from torch.utils.data import DataLoader

dataset = Fpidataset(train=True, img_size=60, transform=None)


dataloader = DataLoader(dataset, batch_size=len(dataset), num_workers=1)


data = next(iter(dataloader))


print("mean: ", data[0].mean(), "std: ",data[0].std())

