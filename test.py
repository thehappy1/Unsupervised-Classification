from data.fpidataset import Fpidataset

train = Fpidataset(train=True,transform=None)

print(train.df.head(100))