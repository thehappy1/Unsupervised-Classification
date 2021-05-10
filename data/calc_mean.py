from fpidataset import Fpidataset

data = Fpidataset(train=True)

width, height = data[0]["image"].size
size = width*height
print(size)