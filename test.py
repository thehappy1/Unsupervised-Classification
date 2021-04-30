from data.fpidataset import Fpidataset

train = Fpidataset(train=True,transform=None)

temp = train.df.articleType.unique()
train.df[["articleType", "targets"]]
print(temp)