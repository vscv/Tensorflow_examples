
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
    $time python get_jpg_from_url.py -i url_list2.csv -s jpg_out -n 2048
    real    2m48.058s
    user    1m26.038s
    sys     2m35.598s
    
    $time python get_jpg_from_url.py -i url_list2.csv -s jpg_out -n 512
    real    1m20.815s
    user    0m47.943s
    sys     0m31.255s
    
    $time python get_jpg_from_url.py -i url_list2.csv -s jpg_out -n 256
    real    1m54.265s
    user    0m44.560s
    sys     0m25.226s
    
    $time python get_jpg_from_url.py -i url_list2.csv -s jpg_out -n 1
    real    14m35.736s
    user    0m39.410s
    sys     0m21.729s
      
    
    [Urq:]
    $time python get_jpg_from_url.py -i url_list2.csv -s jpg_out -n 1024
    real    0m59.613s
    user    0m46.034s
    sys     0m54.899s
    
    $time python get_jpg_from_url.py -i url_list2.csv -s jpg_out -n 512
    real    0m43.469s
    user    0m32.559s
    sys     0m29.511s
 
    $time python get_jpg_from_url.py -i url_list2.csv -s jpg_out -n 256
    real    0m52.468s
    user    0m28.172s
    sys     0m22.509s
    
    $time python get_jpg_from_url.py -i url_list2.csv -s jpg_out -n 128
    real    0m39.569s
    user    0m24.189s
    sys     0m19.536s


    #[2020-02-07]
    #todo: find a way to get the now/total count of task finished!!
    config.count +=1 , simply sum howmany time the callback function was ran.
    
    $time python get_jpg_from_url_ORG.py -i url_list2.csv -s `date +"%Y-%m-%d-%H-%M"` -n 256
    * * * Check: cpu 36
    * * * Check: index 5308
    * * * Check: new list len 5225
    * * * Check: Sending workers.....
    100%|████████████████████████████████████████████████████████████████████████████████| 5225/5225 [00:02<00:00, 2060.43it/s]
    * * * Check: Downloading.........
    * * * Check: now in 4393 for 5226/5225
    real    1m0.922s
    user    0m29.626s
    sys     0m20.953s
    
"""


import os
#import wget
import argparse
import pandas as pd
import urllib.request as urq

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor as ppe
from concurrent.futures import as_completed

import config
config.count=1

def write_stream_frame(id_count, url):
    img_name=os.path.basename(url)
    path=dir + '/' + img_name
#    wget.download(url, path)
    urq.urlretrieve(url, path) # urq seems faster than wget.
#    print('* * * Check: path', path)
    return id_count

def handle(now):
    now=now.result()
    config.count +=1
    print('\r* * * Check: now in {} for {}/{}'.format(now, config.count, len_url_list), end="")


# check cpu of sys
print('* * * Check: cpu', os.cpu_count())

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
global index
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
print('* * * Check: new list len', len(url_list))
len_url_list=len(url_list)

## download jpg to local dir
#with ppe(max_workers=int(par)) as exc:
#    print('* * * Check:  Sending workers.....')
#    for id_count in tqdm(range(len(url_list))):
#        url=url_list[id_count]
##        print('* * * Check: works', id_count)
#        exc.submit(write_stream_frame, url)
#    print('* * * Downloading.....')


## download jpg to local dir with checking
#with ppe(max_workers=int(par)) as exc:
#    print('* * * Check:  Sending workers.....')
#    for id_count in range(len(url_list)):#len(url_list)
#        url=url_list[id_count]
##        print('* * * Check: works', id_count, "/", len(url_list))
#        all_task=[exc.submit(write_stream_frame, id_count, url)]
#
##        print('* * * Check: as_completed', all_task)
#        for future in as_completed(all_task): # as_completed check will slowdown download due to the lock.
#            data = future.result()
#            print("* * * Check: Downloading #{} ok.".format(data))

# download jpg to local dir with callback fun
with ppe(max_workers=int(par)) as exc:
    print('* * * Check: Sending workers.....')
    for id_count in tqdm(range(len(url_list))):#len(url_list)
        url=url_list[id_count]
#        print('* * * Check: works', id_count)
        exc.submit(write_stream_frame, id_count, url).add_done_callback(handle)
    print('* * * Check: Downloading.........')

