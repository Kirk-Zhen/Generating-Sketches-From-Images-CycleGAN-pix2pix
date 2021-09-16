import pandas as pd
from os.path import join
from PIL import Image
import numpy as np


class LoadData():
    def __init__(self,
                 seed=132327,
                 path='data/stats.csv'):

        df = pd.read_csv(path)
        # filter data
        # we only accept correct sketch in low difficulties (1,2).
        acceptable_difficulties = (1, 2)
        df = df[df['Difficulty'].isin(acceptable_difficulties)]
        df = df[df['Error?'] == 0]
        df = df[df['Context?'] == 0]
        df = df[df['Ambiguous?'] == 0]
        df = df[df['WrongPose?'] == 0]
        df = df

        # Train-test split
        np.random.seed(seed)
        df_unique = df.drop_duplicates(subset='ImageNetID')
        # only 80% image is used for training
        # For the rest 20% images, only one sketch per image for testing
        msk = np.random.rand(len(df_unique)) < 0.8
        self.df_train = df.loc[df['ImageNetID'].isin(df_unique[msk]['ImageNetID'])]
        df_test = df.loc[df['ImageNetID'].isin(df_unique[~msk]['ImageNetID'])]
        # only return a single 'sketch' per image for testing
        self.df_test = df_test.drop_duplicates(subset='ImageNetID')

    def getData(self,
                mode="train",
                data_dir='./data',
                image_dir='256x256/photo/tx_000000000000',
                sketch_dir='256x256/sketch/tx_000000000000',
                ):

        if mode == "test":
            images = np.array(
                [np.array(Image.open(join(data_dir, image_dir, self.df_test.iloc[idx]['Category'].replace(' ', '_'),
                                          f'{self.df_test.iloc[idx]["ImageNetID"]}.jpg')).resize((256, 256))) for idx
                 in
                 range(self.df_test.shape[0])])
            sketches = np.array(
                [np.array(Image.open(join(data_dir, sketch_dir, self.df_test.iloc[idx]['Category'].replace(' ', '_'),
                                          f'{self.df_test.iloc[idx]["ImageNetID"]}-{self.df_test.iloc[idx]["SketchID"]}.png')).resize(
                    (256, 256))) for idx in
                    range(self.df_test.shape[0])])

            labels = np.array([self.df_test.iloc[idx]['CategoryID'] for idx in range(self.df_test.shape[0])])
        else:
            images = np.array(
                [np.array(Image.open(join(data_dir, image_dir, self.df_train.iloc[idx]['Category'].replace(' ', '_'),
                                          f'{self.df_train.iloc[idx]["ImageNetID"]}.jpg')).resize((256, 256))) for idx
                 in
                 range(self.df_train.shape[0])])
            sketches = np.array(
                [np.array(Image.open(join(data_dir, sketch_dir, self.df_train.iloc[idx]['Category'].replace(' ', '_'),
                                          f'{self.df_train.iloc[idx]["ImageNetID"]}-{self.df_train.iloc[idx]["SketchID"]}.png')).resize(
                    (256, 256))) for idx in
                    range(self.df_train.shape[0])])

            labels = np.array([self.df_train.iloc[idx]['CategoryID'] for idx in range(self.df_train.shape[0])])

        return images, sketches, labels
