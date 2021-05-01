from data.fpidataset import Fpidataset
import numpy as np
train = Fpidataset(train=True,transform=None)


print(np.unique(train.df[["articleType", "targets"]].head(100)))