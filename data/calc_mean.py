from fpidataset import Fpidataset
from torch.utils.data import DataLoader
from fashion import Fashion

dataset = Fashion(train=False, transform=None)

print(dataset.__getitem__(1999))

