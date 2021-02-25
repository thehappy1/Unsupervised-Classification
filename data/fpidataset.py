import os
import pandas as pd
from PIL import Image

class Fpidataset():
    # Constructor
    def __init__(self, train=True, transform=None):

        self.train = train
        self.transform = transform

        self.df = pd.read_csv('data/styles.csv', error_bad_lines=False)
        self.df['image_path'] = self.df.apply(lambda x: os.path.join("data/images", str(x.id) + ".jpg"), axis=1)
        self.df = self.df.drop([32309, 40000, 36381, 16194, 6695])

        #map articleType as number
        mapper = {}
        for i, cat in enumerate(list(self.df.articleType.unique())):
            mapper[cat] = i
        print(mapper)
        self.df['targets'] = self.df.articleType.map(mapper)
        print(self.df.head(100))

        if self.train:
            self.images, self.labels = self.get_i_items(self.df, 800, train=True)
        else:
            self.images, self.labels = self.get_i_items(self.df, 200, train=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        print("index: ",index)
        label = self.labels[index]
        image = self.images[index]

        img_size = image.size

        # transform
        if self.transform is not None:
            image = self.transform(image)

        out = {'image': image, 'target': label, 'meta': {'im_size': img_size, 'index': index}}

        return out

    def get_i_items(self, df, number_of_items, train):
        # get i items of each target

        # calculate classes with more than 1000 items
        temp = df.targets.value_counts().sort_values(ascending=False)[:10].index.tolist()
        df_temp = df[df["targets"].isin(temp)]

        images = []
        labels = []

        if train:
            for label in temp:

                train_temp = df_temp[df_temp.targets == label]
                train_temp = train_temp[:number_of_items]

                labels.extend(train_temp["targets"].to_list())

                for element in train_temp.image_path:
                    img = Image.open(element)
                    img = img.resize((60,80))
                    images.append(img)

                print("Anzahl x_train items bei ", label, " :", len(images))
                print(" ")
        else:
            for label in temp:

                test_temp = df_temp[df_temp.targets == label]
                test_temp = test_temp[800:1000]

                labels.extend(test_temp["targets"].to_list())

                for element in test_temp.image_path:
                    img = Image.open(element)
                    img = img.resize((60, 80))
                    images.append(img)

                print("Anzahl x_test items bei ", label, " :", len(images))
                print(" ")

        return images, labels

