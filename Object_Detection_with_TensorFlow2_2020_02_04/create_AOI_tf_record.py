# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
-------------------------------------------
J:2020-01-07: modify to fit isvlab dataset.
-------------------------------------------

Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=/home/user/VOCdevkit \
        --year=VOC2012 \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('year', 'VOC2007', 'Desired challenge year.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test', 'All'] # add 'All' for ivs
YEARS = ['VOC2007', 'VOC2012', 'merged', 'All'] # add 'All' for ivs


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  #J: check the xml data parting. the vis is mutiple objects not as VOC is single obj.
  #print('* * * Check xml img name',data['object'][0]['filename'])

  #J: reset the parameters.
#  img_path = os.path.join(data['folder'], image_subdirectory, data['filename'])

  #J: for ivs
#  img_path = os.path.join(image_subdirectory, data['object'][0]['filename'])

  #J: 2020-02-04 for AOI from VoTT
  # org_bmp but add new_jpg, keep xml same but modify by this tf_record.py
  aoi_img_name =  os.path.splitext(data['filename'])[0] + '.jpg'
  img_path = os.path.join(image_subdirectory, aoi_img_name)
  
  
  full_path = os.path.join(dataset_directory, img_path)
  
  #print('* * * Check img full_path', dataset_directory, img_path)
  
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  #J:ivs donot have image size infor.
  width = int(data['size']['width'])
  height = int(data['size']['height'])
  #ivslab
#  width = int(1920)
#  height = int(1080)
  # AOI
#  width = int(2592)
#  height = int(2048)

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  if 'object' in data:
    for obj in data['object']:
      #J:ivs donot have 'difficult' infor.
#      difficult = bool(int(obj['difficult']))
      difficult = bool(int(0))
      if ignore_difficult_instances and difficult:
        continue

      difficult_obj.append(int(difficult))

      #J: check obj is read right.
      #print('* * * Check obj label name:', obj['name'].encode('utf8'), 'label:', label_map_dict[obj['name']], 'bbox:', obj['bndbox']['xmin'], obj['bndbox']['ymin'])
      
      #J:20200109 check bbox is correct! for Loss is INF/NaN issue when training.
#      x1=int(obj['bndbox']['xmin'])
#      y1=int(obj['bndbox']['ymin'])
#      x2=int(obj['bndbox']['xmax'])
#      y2=int(obj['bndbox']['ymax'])
      
      # for AOI,  <xmin>1221.9739670798042</xmin> need to round it. os.path.splitext(data['filename'])[0]
      x1=int(os.path.splitext(obj['bndbox']['xmin'])[0])
      y1=int(os.path.splitext(obj['bndbox']['ymin'])[0])
      x2=int(os.path.splitext(obj['bndbox']['xmax'])[0])
      y2=int(os.path.splitext(obj['bndbox']['ymax'])[0])
      print('* * * Check bbox :', x1, y1, x2, y2)
      
      if x1 > x2 or y1 > y2:
        raise Exception('bbox rnage error!')
#        print('bbox rnage error!')
      if x2 > width or y2 > height:# bcs, x1 was sure less then x2,
        raise Exception('bbox over image size!')
#        print('bbox over image size!')
      if int(x2 - x1) < 16 or int(y2 - y1) < 16: # should check % of image not simple pixel.
        print('* * * Check WXH :', int(x2 - x1), int(y2 - y1))
        raise Exception('object too small')
#        print('object too small')

      xmin.append(float(obj['bndbox']['xmin']) / width)
      ymin.append(float(obj['bndbox']['ymin']) / height)
      xmax.append(float(obj['bndbox']['xmax']) / width)
      ymax.append(float(obj['bndbox']['ymax']) / height)
      classes_text.append(obj['name'].encode('utf8'))
      classes.append(label_map_dict[obj['name']])
      
      #J:ivs donot have 'truncated' infor.
#      truncated.append(int(obj['truncated']))
      truncated.append(int(0))
      #J:ivs donot have 'truncated' infor.
#      poses.append(obj['pose'].encode('utf8'))
      poses.append('Unspecified'.encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      #J:Fixed. data['filename'].encode('utf8') -->  data['object'][0]['filename'] --> aoi_img_name
      'image/filename': dataset_util.bytes_feature(
          aoi_img_name.encode('utf8')),
      #J:Fixed. data['filename'].encode('utf8') -->  data['object'][0]['filename'] --> aoi_img_name
      'image/source_id': dataset_util.bytes_feature(
          aoi_img_name.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example


def main(_):
  if FLAGS.set not in SETS:
    raise ValueError('set must be in : {}'.format(SETS))
  if FLAGS.year not in YEARS:
    raise ValueError('year must be in : {}'.format(YEARS))

  data_dir = FLAGS.data_dir
  years = ['VOC2007', 'VOC2012']
  if FLAGS.year != 'merged':
    years = [FLAGS.year]

  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  for year in years:
    logging.info('Reading from IVSLAB %s dataset.', year)
    examples_path = os.path.join(data_dir, 'ImageSets', FLAGS.set + '.txt')
    annotations_dir = os.path.join(data_dir, FLAGS.annotations_dir)
    
    print('* * * check path', data_dir, FLAGS.annotations_dir, annotations_dir)
    
    examples_list = dataset_util.read_examples_list(examples_path)
    #print(examples_list)#J# path include "'All'/xxxx.mp4/xxxx" automatically.
    for idx, example in enumerate(examples_list):
      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(examples_list))
      #J:mode
      example=os.path.splitext(example)[0]
      print('****** Check name ', example)
      path = os.path.join(annotations_dir, example + '.xml')
      
      #J: check where image on process.
      #gprint('* * * xml path', annotations_dir, path, example + '.xml')
      
      with tf.gfile.GFile(path, 'rb') as fid:
        xml_str = fid.read()
      xml = etree.fromstring(xml_str)
      data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
#      data = dataset_util.recursive_parse_xml_to_dict(xml)
      
      #J:mode
      print('****** Check data ', data)

      tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict,
                                  FLAGS.ignore_difficult_instances)
      writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()
