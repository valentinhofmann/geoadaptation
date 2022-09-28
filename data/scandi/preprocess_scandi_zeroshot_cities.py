from collections import Counter
import pandas as pd
import requests
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim
from transformers import AutoTokenizer

from helpers_scandi import *
import json

def main():

    # # locator = Nominatim(user_agent='nlp_geocoding')
    # # locator.geocode("Zagreb", language="en")
    # # loc = locator.reverse('57.70, 18.66', language = 'da')

    # Load dev data
    # data_jsons = [json.loads(x) for x in load_lines("/ceph/gglavas/data/geoadaptation/Scandi/scandi_geoloc_zero.json")]
    # data = [(x["latitude"], x["longitude"], x["text"]) for x in data_jsons]
    
    # data = pd.DataFrame(data, columns=['latitude', 'longitude', 'text'])
    # print(data.shape)

    # # Remove posts with less than 10 words
    # data['length'] = data.text.apply(lambda x: len(x.strip().split()))
    # data = data[data.length >= 10]
    # print(data.shape)

    # # Get data cities
    # locator = Nominatim(user_agent='nlp_geocoding')
    # loc = locator.reverse('57.70, 18.66', language = 'da')
    # data['point'] = data.apply(lambda r: '{}, {}'.format(r['latitude'], r['longitude']), axis=1)
    # points = set(data.point)
    # point2city = dict()
    # #
    # for i, p in enumerate(points):
    #     if i % 100 == 0:
    #         print(i)
    #     location = locator.reverse(p, language='da')
    #     try:
    #         if "city" in location.raw['address']:
    #             point2city[p] = location.raw['address']['city']
    #         elif "municipality" in location.raw['address'] and location.raw['address']['municipality'].endswith("kommun"):
    #             city = location.raw['address']['municipality'].split()[0]
    #             if city.endswith("s"):
    #                 city = city[:-1]
    #             point2city[p] = city
    #     except:
    #         continue
        
    # city_words = {'staden', 'stad', 'kommun', 'by', 'kommune', 'Staden', 'Stad', 'Kommun', 'By', 'Kommune'}
    # point2city = {
    #     p: ' '.join(w for w in c.split() if w not in city_words) for p, c in point2city.items()
    # }
    
    # data['location'] = data.point.apply(lambda x: point2city[x] if x in point2city else '')

    # serialize(point2city,  "/ceph/gglavas/data/geoadaptation/Scandi/p2c.pkl")
    # data.to_pickle("/ceph/gglavas/data/geoadaptation/Scandi/locations_pandas_df.pkl")
    
    # exit()

    p2c = deserialize("/ceph/gglavas/data/geoadaptation/Scandi/p2c.pkl")
    data = pd.read_pickle("/ceph/gglavas/data/geoadaptation/Scandi/locations_pandas_df.pkl")

    # Get BERTic cities
    tok = AutoTokenizer.from_pretrained('vesteinn/ScandiBERT', model_max_length=128)
    vocab_bertic = set(tok.vocab)

    url_sv = 'https://en.wikipedia.org/wiki/List_of_cities_in_Sweden'
    url_no = 'https://en.wikipedia.org/wiki/List_of_cities_in_Norway'
    url_da = 'https://en.wikipedia.org/wiki/List_of_cities_in_Denmark'
    
    url2country = {
        url_sv: 'sv',
        url_no: 'no',
        url_da: 'da',
    }
 
    cities_bertic, country2cities = set(), dict()

    for url in url2country:
        soup = BeautifulSoup(requests.get(url).text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})
        table = pd.read_html(str(table))
        table = pd.DataFrame(table[0])
        table.columns = list(range(len(table.columns)))
        if url2country[url] == 'da':
            cities_country = [clean_city(c) for c in table[1]]
            cities_country.append("København")
        else:
            cities_country = [clean_city(c) for c in table[0]]
            if url2country[url] == "sv":
                cities_country.append("Göteborg")
        
        cities_country = {c : tok.tokenize(c)[0] for c in cities_country if len(tok.tokenize(c)) == 1}
        cities_bertic.update([c for c in cities_country if cities_country[c] in vocab_bertic])
        country2cities[url2country[url]] = cities_bertic #[cities_country[c] for c in cities_country if c in vocab_bertic]
    
    #data['firsttok'] = data.location.apply(lambda x: tok.tokenize(x)[0] if x else '')
    print(len(cities_bertic))

    # Filter data
    theta = 80
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

    data.to_json('/ceph/gglavas/data/geoadaptation/Scandi/zeroshot_cities_more.json', orient='records', lines=True, force_ascii=False)

if __name__ == '__main__':
    main()
