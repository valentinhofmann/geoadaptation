import geopandas as gpd
import pandas as pd

from helpers import *


def main():

    # Load dev data
    data = list()
    with open('dev.txt', 'r') as f:
        for l in f:
            if l.strip() == '':
                continue
            data.append(l.strip().split('\t'))

    # Load test data
    with open('test_gold.txt', 'r') as f:
        for l in f:
            if l.strip() == '':
                continue
            data.append(l.strip().split('\t'))
    data = pd.DataFrame(data, columns=['latitude', 'longitude', 'text'])
    print(data.shape)

    # Remove posts with less than 10 words
    data['length'] = data.text.apply(lambda x: len(x.strip().split()))
    data = data[data.length >= 10]
    print(data.shape)

    # Filter BCMS countries
    data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.longitude, data.latitude))
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    bcms = world[world.name.isin({'Bosnia and Herz.', 'Croatia', 'Montenegro', 'Serbia'})].dissolve()
    data = data[data.within(bcms.loc[0, 'geometry'])]
    print(data.shape)

    # Get country
    bih = world[world.name == 'Bosnia and Herz.'].geometry.iloc[0]
    hrv = world[world.name == 'Croatia'].geometry.iloc[0]
    mne = world[world.name == 'Montenegro'].geometry.iloc[0]
    srb = world[world.name == 'Serbia'].geometry.iloc[0]
    data['location'] = data.geometry.apply(lambda x: point2country(x, bih, hrv, mne, srb))
    data = data[data.location.isin({'Hrvatska', 'Srbija'})]
    data = pd.DataFrame(data)[['location', 'text']]
    print(data.shape)

    # Downsample to 1,000 posts per country
    n = 1000
    grouped = data.groupby('location', group_keys=False)
    data = grouped.apply(lambda x: x.sample(n, random_state=0))
    print(data.shape)
    print(data.location.value_counts())

    data.to_json('zeroshot_countries.json', orient='records', lines=True)


if __name__ == '__main__':
    main()
