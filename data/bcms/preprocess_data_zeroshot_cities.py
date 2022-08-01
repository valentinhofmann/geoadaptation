from collections import Counter

import pandas as pd
import requests
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim
from transformers import AutoTokenizer

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

    # Get data cities
    locator = Nominatim(user_agent='nlp_geocoding')
    data['point'] = data.apply(lambda r: '{}, {}'.format(r['latitude'], r['longitude']), axis=1)
    points = set(data.point)
    point2city = dict()
    for i, p in enumerate(points):
        location = locator.reverse(p, language='hr')
        try:
            point2city[p] = translit_cyrillic(location.raw['address']['city'])
        except KeyError:
            continue
        if i % 100 == 0:
            print(i / len(points))
    city_words = {'Grad', 'Opština', 'opština', 'Gradska', 'Općina'}
    point2city = {
        p: ' '.join(w for w in c.split() if w not in city_words) for p, c in point2city.items()
    }
    data['location'] = data.point.apply(lambda x: point2city[x] if x in point2city else '')

    # Get BERTic cities
    tok = AutoTokenizer.from_pretrained('classla/bcms-bertic', model_max_length=128)
    vocab_bertic = set(tok.vocab)
    url_bih = 'https://en.wikipedia.org/wiki/List_of_cities_in_Bosnia_and_Herzegovina'
    url_hrv = 'https://en.wikipedia.org/wiki/List_of_cities_and_towns_in_Croatia'
    url_mne = 'https://en.wikipedia.org/wiki/List_of_cities_in_Montenegro'
    url_srb = 'https://en.wikipedia.org/wiki/List_of_cities_in_Serbia'
    url2country = {
        url_bih: 'bih',
        url_hrv: 'hrv',
        url_mne: 'mne',
        url_srb: 'srb'
    }
    cities_bertic, country2cities = set(), dict()
    for url in url2country:
        soup = BeautifulSoup(requests.get(url).text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})
        table = pd.read_html(str(table))
        table = pd.DataFrame(table[0])
        table.columns = list(range(len(table.columns)))
        if url2country[url] == 'mne':
            cities_country = [clean_city(c) for c in table[1]]
        elif url2country[url] == 'srb':
            cities_country = [clean_city(c) for c in table[0]] + ['Beograd']
        else:
            cities_country = [clean_city(c) for c in table[0]]
        cities_bertic.update([c for c in cities_country if c in vocab_bertic])
        country2cities[url2country[url]] = [c for c in cities_country if c in vocab_bertic]
    print(len(cities_bertic))

    # Filter data
    theta = 100
    data = data[data.location.isin(cities_bertic)]
    print(data.location.value_counts())
    city_counter = Counter(data.location)
    cities_filtered = set([c for c, count in city_counter.most_common() if count >= theta])
    data = data[data.location.isin(cities_filtered)]
    data = data[['location', 'text']]
    print(data.shape)

    # Downsample to 100 posts per city
    n = 100
    grouped = data.groupby('location', group_keys=False)
    data = grouped.apply(lambda x: x.sample(n, random_state=123))
    print(data.shape)
    print(data.location.value_counts())

    data.to_json('zeroshot_cities.json', orient='records', lines=True)


if __name__ == '__main__':
    main()
