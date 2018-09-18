import json

items = json.loads(open('20180830.data.sr.score.1581.json').read())
# print(items.keys())

d = {}

for key, sr_score_pair, label in zip(items['keys'], items['sr_score_pairs'], items['labels']):
    # print(key, sr_score_pair, label)
    d[key] = {
        'sr': sr_score_pair,
        'label': label
    }

with open('20180918.label.map.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(d, ensure_ascii=False))

from os import listdir

files = list(listdir('./data/attitudes/videos'))

d2 = {}

for file in files:
    file = file.replace('.mp4', '')
    d2[file] = d[file]

print(json.dumps(d2, indent=2))

with open('/private/var/py/AttitudeDetection/data/attitudes/sr.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(d2, indent=2))
