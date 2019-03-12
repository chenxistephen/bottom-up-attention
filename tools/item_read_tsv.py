#!/usr/bin/env python


import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import numpy as np
import os.path as osp
import os
import sys
csv.field_size_limit(sys.maxsize)
   
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
# infile = '/data/coco/tsv/trainval/karpathy_val_resnet101_faster_rcnn_genome.tsv'



if __name__ == '__main__':
    infile = sys.argv[1]
    imgOrder = int(sys.argv[2])
    indir = osp.dirname(infile)
    npydir = osp.join('.','image_boxes')
    if not osp.isdir(npydir):
        os.makedirs(npydir)
    # Verify we can read a tsv
    in_data = {}
    poolh = 14
    poolw = 14
    dim = 1024
    count = 0
    with open(infile, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for ix, item in enumerate(reader):
            print ix
            if ix == imgOrder:
                item['image_id'] = int(item['image_id'])
                #print item['image_id']
                item['image_h'] = int(item['image_h'])
                item['image_w'] = int(item['image_w'])   
                item['num_boxes'] = int(item['num_boxes'])
                
                for field in ['boxes', 'features']:
                    # if field == 'boxes':
                    item[field] = np.frombuffer(base64.decodestring(item[field]), 
                          dtype=np.float32).reshape((item['num_boxes'],-1))
                    print item[field]
                    # elif field == 'features':
                        # item[field] = np.frombuffer(base64.decodestring(item[field]), 
                              # dtype=np.float32).reshape((item['num_boxes'], dim, poolh, poolw))
                    np.save(osp.join(npydir, 'order_{}_imageid_{}_{}.npy'.format(imgOrder, item['image_id'], field)), item[field])
                break
                


