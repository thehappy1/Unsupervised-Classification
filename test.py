from data.fpidataset import Fpidataset

train = Fpidataset(train=True,transform=None)

temp = train.df[["articleType", "targets"]]
print(temp)