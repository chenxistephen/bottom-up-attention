#!/usr/bin/env python


import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap

csv.field_size_limit(sys.maxsize)
   
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features', 'max_conf']
# infile = '/data/coco/tsv/trainval/karpathy_val_resnet101_faster_rcnn_genome.tsv'



if __name__ == '__main__':
    infile = sys.argv[1]
    # Verify we can read a tsv
    in_data = {}
    poolh = 14
    poolw = 14
    dim = 1024
    count = 0
    with open(infile, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
            item['image_id'] = int(item['image_id'])
            #print item['image_id']
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])   
            item['num_boxes'] = int(item['num_boxes'])
            count += 1
            for field in ['boxes', 'features', 'max_conf']:
                if field == 'boxes':
                    item[field] = np.frombuffer(base64.decodestring(item[field]), 
                          dtype=np.float32).reshape((item['num_boxes'],-1))
                elif field == 'features':
                    item[field] = np.frombuffer(base64.decodestring(item[field]), 
                          dtype=np.float32).reshape((item['num_boxes'], -1)) #dim, poolh, poolw))
                elif field == 'max_conf': # Note: should dtype=np.float64!, not float32
                    item[field] = np.frombuffer(base64.decodestring(item[field]), dtype=np.float64).reshape((item['num_boxes'], -1))
            print count
            if (count ) % 100 == 0:
                print "max_conf = {}".format(item['max_conf'])
                print "max_conf.shape = {}".format(item['max_conf'].shape)
                print "{}: item['features'] dimension = {}".format(count+1, item['features'].shape)
            in_data[item['image_id']] = item
    #print in_data
    print len(in_data)


