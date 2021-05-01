from data.fpidataset import Fpidataset

train = Fpidataset(train=True,transform=None)


print(train.df[["articleType", "targets"]].head(100))