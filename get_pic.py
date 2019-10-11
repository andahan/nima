#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'get wesee cover pic'

import pandas as pd
import os
import sys
import math
import json
import requests
import time
import urllib.request


def main():
    file_path = sys.argv[1]
    target_path = sys.argv[2]

    # file_path = '/data/algceph/templezhang/anhan/NIMA-aesthetic/t_weishi_tonality_sample_v0828.txt'
    # target_path = '/data/algceph/templezhang/anhan/NIMA-aesthetic/images/'

    # file_path = './t_weishi_tonality_sample_v0828.txt'
    # target_path = './images/'

    file = pd.read_csv(file_path, encoding='utf-8', sep='\t',header=0, names=['feedid','docid','ispgc','cate1','cate2',
                                                                              'manual_cate1','manual_cate2','manual_quality',
                                                                              'manual_tags','music_id','personid',
                                                                              'rcmd_auditstate','video_duration',
                                                                              'cover_url','video_fileid'])
    feedid_list = file['feedid']
    cover_list = file['cover_url']

    for i, url in enumerate(cover_list):
        print(url)
        name = str(feedid_list[i]) + ".jpg"
        try:
            urllib.request.urlretrieve(url, target_path+'/%s' % name)
            time.sleep(0.1)
        except:
            print('下载失败')
            continue

if __name__ == '__main__':
    main()