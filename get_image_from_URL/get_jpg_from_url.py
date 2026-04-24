#!/usr/bin/env python3

"""
-------------------------------------------
J[2020-02-05] modify to get jpeg from URLs.
-------------------------------------------

For download jpg by its URLs from utf-8 CSV file.

ps. due to write 5.7GB (~5000 image) to disk, Threads/workers will take long time to close.

Usage:
    get_jpg_from_url.py [-h] -i CSV_FILE -s SAVE_DIR -n NUM_WORKERS


Example usage:

    $time python get_jpg_from_url.py -i url_list2.csv -s `date +"%Y-%m-%d-%H-%M"` -n 128
    save jpg to 2020-02-06-00-25/

Example usage:

    [Wget:]
    $time python get_jpg_from_url.py -i url_list2.csv -s jpg_out -n 1
    real    14m35.736s
    user    0m39.410s
    sys     0m21.729s
      
    
    [Urq:]
    $time python get_jpg_from_url.py -i url_list2.csv -s jpg_out -n 128
    real    0m39.569s
    user    0m24.189s
    sys     0m19.536s
     
"""


import os
#import wget
import argparse
import pandas as pd
import urllib.request as urq

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor as ppe


def write_stream_frame(url):
    img_name=os.path.basename(url)
    path=dir + '/' + img_name
#    wget.download(url, path)
    urq.urlretrieve(url, path) # urq seems faster than wget.
#    print('* * * Check: path', path)


# check cpu of sys
#print('* * * Check: cpu', os.cpu_count())


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--csv_file", required=True, help="Path to CSV file.")
ap.add_argument("-s", "--save_dir", required=True, help="Path to save jpg")
ap.add_argument("-n", "--num_work", required=True, help="number of wrokers")
args=vars(ap.parse_args())
csv=args["csv_file"]
dir=args["save_dir"]
par=args["num_work"]
#print('* * * Check: input', csv, dir, par)


#create output dir.
os.mkdir(dir)


# read csv
df=pd.read_csv(csv)
index=df.shape[0]
print('* * * Check: index', index)


# parser url
url_list=[]
for id_count in range(index):
    url=str(df.iloc[id_count][4])
    if url.endswith(".jpg"):
        url_list.append(url)
#        print('* * * Check:', url)
#print('* * * Check: New List', url_list)
print('* * * Check: new list index', len(url_list))


# download jpg to local dir
with ppe(max_workers=int(par)) as exc:
    print('* * * Check:  Sending workers.....')
    for id_count in tqdm(range(len(url_list))):
        url=url_list[id_count]
#        print('* * * Check: works', id_count)
        exc.submit(write_stream_frame, url)
    print('* * * Check:  Downloading.....')
