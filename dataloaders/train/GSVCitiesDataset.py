import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# NOTE: Hard coded path to dataset folder 
BASE_PATH = 'datasets/'

if not Path(BASE_PATH).exists():
    raise FileNotFoundError(
        'BASE_PATH is hardcoded, please adjust to point to gsv_cities')


class GSVCitiesDataset(Dataset):
    def __init__(self,
                 cities=['London', 'Boston'],
                 img_per_place=4,
                 min_img_per_place=4,
                 random_sample_from_each_place=True,
                 transform=default_transform,
                 base_path=BASE_PATH
                 ):
        super(GSVCitiesDataset, self).__init__()
        self.base_path = base_path
        self.cities = cities

        assert img_per_place <= min_img_per_place, \
            f"img_per_place should be less than {min_img_per_place}"
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.random_sample_from_each_place = random_sample_from_each_place
        self.transform = transform

        # generate the dataframe contraining images metadata
        self.dataframe = self.__getdataframes()

        # get all unique place ids
        self.places_ids = pd.unique(self.dataframe.index)
        self.total_nb_images = len(self.dataframe)

    K = 0

    def __getdataframes(self):
        ''' 
            Return one dataframe containing
            all info about the images from all cities

            This requieres DataFrame files to be in a folder
            named Dataframes, containing a DataFrame
            for each city in self.cities
        '''
        # read the first city dataframe
        df = pd.read_csv(self.base_path + 'Dataframes/' + f'{self.cities[0]}.csv')
        df = df.sample(frac=1)  # shuffle the city dataframe

        # append other cities one by one
        for i in range(1, len(self.cities)):
            tmp_df = pd.read_csv(
                self.base_path + 'Dataframes/' + f'{self.cities[i]}.csv')

            # Now we add a prefix to place_id, so that we
            # don't confuse, say, place number 13 of NewYork
            # with place number 13 of London ==> (0000013 and 0500013)
            # We suppose that there is no city with more than
            # 99999 images and there won't be more than 99 cities
            # TODO: rename the dataset and hardcode these prefixes
            prefix = i
            tmp_df['place_id'] = tmp_df['place_id'] + (prefix * 10 ** 5)
            tmp_df = tmp_df.sample(frac=1)  # shuffle the city dataframe

            df = pd.concat([df, tmp_df], ignore_index=True)

        # keep only places depicted by at least min_img_per_place images
        res = df[df.groupby('place_id')['place_id'].transform(
            'size') >= self.min_img_per_place]
        return res.set_index('place_id')

    def __getitem__(self, index):
        place_id = self.places_ids[index]

        # get the place in form of a dataframe (each row corresponds to one image)
        place = self.dataframe.loc[place_id]
        # sample K images (rows) from this place
        # we can either sort and take the most recent k images
        # or randomly sample them
        if self.random_sample_from_each_place:
            place = place.sample(n=self.img_per_place)
        else:  # always get the same most recent images
            place = place.sort_values(
                by=['year', 'month', 'lat'], ascending=False)
            place = place[: self.img_per_place]

        imgs = []

        for i, row in place.iterrows():
            id = row.name
            city = row['city']
            GSVCitiesDataset.K += 1
            # print(GSVCitiesDataset.K)
            img_name = self.get_img_name(row)
            img_path = 'MLDL_datasets/gsv_xs/train/' + \
                       city + '/' + img_name
            try:
                with self.image_loader(img_path) as img:
                    if self.transform is not None:
                        img = self.transform(img)

                    imgs.append(img)
            except (Exception, FileNotFoundError) as e:
                print(e)
        # NOTE: contrary to image classification where __getitem__ returns only one image 
        # in GSVCities, we return a place, which is a Tesor of K images (K=self.img_per_place)
        # this will return a Tensor of shape [K, channels, height, width]. This needs to be taken into account 
        # in the Dataloader (which will yield batches of shape [BS, K, channels, height, width])
        return torch.stack(imgs), torch.tensor(place_id).repeat(self.img_per_place)

    def __len__(self):
        '''Denotes the total number of places (not images)'''
        return len(self.places_ids)

    @staticmethod
    def image_loader(path):
        return Image.open(path).convert('RGB')

    @staticmethod
    def get_img_name(row):
        # given a row from the dataframe
        # return the corresponding image name

        # now remove the two digit we added to the id
        # they are superficially added to make ids different
        # for different cities
        place_id = row.name % 10 ** 5  # row.name is the index of the row, not to be confused with image name
        place_id = str(place_id)

        # UTM1,UTM2,UTMzone,category,lat,lon,panoid,numeric,yearMonth,place_id
        utm1, utm2 = str(f"{row['UTM1']:.2f}").zfill(10), str(f"{row['UTM2']:.2f}".zfill(10))
        utmZone = str(row['UTMzone']).zfill(2)
        category = str(row['category'])
        lat, lon = str(f"{row['lat']:.5f}").zfill(9), str(f"{row['lon']:.5f}").zfill(10)
        panoid = str(row['panoid'])
        numeric = str(row['numeric']).zfill(3)
        yearMonth = str(row['yearMonth']).zfill(6)
        # place_id = str(row.name)
        city = str(row['city'])
        name = '@' + utm1 + '@' + utm2 + '@' + utmZone + '@' \
               + category + '@' + lat + '@' + lon + '@' + panoid + '@@' + numeric + \
               '@@@@' + yearMonth + '@' + place_id + '_' + city + '@.jpg'

        return name
