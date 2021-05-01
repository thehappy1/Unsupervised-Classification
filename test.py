from data.fpidataset import Fpidataset
import numpy as np
train = Fpidataset(train=True,transform=None)

temp = train.df[["articleType", "targets"]]
print(np.unique(temp.targets))