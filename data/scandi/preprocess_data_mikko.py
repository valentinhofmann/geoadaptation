import codecs
import csv
import os
import json
from helpers_scandi import *
import random

base_path = "/ceph/gglavas/data/geoadaptation/Scandi"

#######
# ##### Mikko's CSV to JSON conversion + cleaning and filtering
#######
# sources = ["SV_v2_all_geo_columns.csv"] # ["NO_v2_all_geo_columns.csv" "DA_v2_all_geo_columns.csv" , "SV_v2_all_geo_columns.csv"]

# entries = []
# for s in sources:
#     print("Loading lines from csv...")
#     lines = get_csv_lines(os.path.join(base_path, s))[1:]
#     print(len(lines))
    
#     cnt = 0
#     for l in lines:
#         cnt += 1
#         if cnt % 10000 == 0: 
#             print(cnt)

#         try:
#             if is_retweet(l[-1]):
#                 continue
#             l[-1] = replace_links(l[-1])
#             l[-1] = replace_usernames(l[-1])

#             if len(l[-1]) < 10 or len(l[-1].split()) < 2:
#                 continue

#             if l[9] != "None" and l[10] != "None" and l[9].strip() and l[10].strip():

#                 long1, lat1 = l[9].replace("[", "").replace("]", "").strip().split(", ")
#                 long2, lat2 = l[10].replace("[", "").replace("]", "").strip().split(", ")
        
#                 long = (float(long1) + float(long2)) / 2
#                 lat = (float(lat1) + float(lat2)) / 2

#                 if long > 5 and long < 25 and lat > 53 and lat < 72:
#                     entries.append(json.dumps({"latitude" : '%.2f' % lat, "longitude" : '%.2f' % long, "text" : l[-1]}, ensure_ascii=False))
#         except:
#             continue    
#     write_list(os.path.join(base_path, "SV_all.json"), entries)

#####
##### balancing, creating adaptation, training and val (for geolocation fine-tuning) and zero-shot evaluation data
#####

all = { "no" : load_lines(os.path.join(base_path, "NO_all.json")), 
        "sv" : load_lines(os.path.join(base_path, "SV_all.json")), 
        "da" : load_lines(os.path.join(base_path, "DA_all.json"))}

min_len = min([len(all[k]) for k in all])

geoadapt = {}
geoloc_train = {}
geoloc_dev = {}
geoloc_test = {}
geoloc_zero = {}

for k in all:
    random.shuffle(all[k])
    all[k] = all[k][: min_len]
    
    geoadapt[k] = all[k][: 200000]
    geoloc_train[k] = all[k][200000 : 250000]
    geoloc_dev[k] = all[k][250000 : 275000]
    geoloc_test[k] = all[k][275000 : 300000]
    geoloc_zero[k] = all[k][300000 : ]

geoadapt_small = []
geoadapt_big = []
gl_train = []
gl_dev = []
gl_test = []
gl_zero = []

for k in all:
    geoadapt_small.extend(geoadapt[k][:100000])
    geoadapt_big.extend(geoadapt[k])
    gl_train.extend(geoloc_train[k])
    gl_dev.extend(geoloc_dev[k])
    gl_test.extend(geoloc_test[k])
    gl_zero.extend(geoloc_zero[k])

random.shuffle(geoadapt_small)
random.shuffle(geoadapt_big)
random.shuffle(gl_train)
random.shuffle(gl_dev)
random.shuffle(gl_test)
random.shuffle(gl_zero)    
    
write_list(os.path.join(base_path, "scandi_geoadapt_small.json"), geoadapt_small)
write_list(os.path.join(base_path, "scandi_geoadapt_big.json"), geoadapt_big)
write_list(os.path.join(base_path, "scandi_geoloc_train.json"), gl_train)
write_list(os.path.join(base_path, "scandi_geoloc_dev.json"), gl_dev)
write_list(os.path.join(base_path, "scandi_geoloc_test.json"), gl_test)
write_list(os.path.join(base_path, "scandi_geoloc_zero.json"), gl_zero)