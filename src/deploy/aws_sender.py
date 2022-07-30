import boto3
import json
import time
import os
import shutil
from absl import app
from absl import flags
import glob

FLAGS = flags.FLAGS
flags.DEFINE_string("location", None, "Location of Virtual Machine")
flags.DEFINE_integer("num", 2, "Number of request")
flags.DEFINE_string("region", None, "Region Name, such as us-east-1")
flags.DEFINE_string("mode", 'S', "mode: 'S' or 'P'")

# required flag
flags.mark_flag_as_required("region")
flags.mark_flag_as_required("location")


hour_ind = [0, 206, 412, 618, 824, 1030, 1236, 1442, 1648, 1854, 2060, 2266, 2472, 2678, 2884, 3090, 3296, 3502, 3708, 3914, 4120, 4326, 4532, 4738, 4952]


def main(argv):
    del argv
    location = FLAGS.location
    region = FLAGS.region
    num = FLAGS.num
    mode = FLAGS.mode

    day = time.strftime("%d", time.localtime())
    hour = int(time.strftime("%H", time.localtime()))

    region_path = '/home/ec2-user/' + region + '_' + day
    if not os.path.exists(region_path): os.mkdir(region_path)

    res_path = os.path.join(region_path, 'res')
    if not os.path.exists(res_path): os.mkdir(res_path)

    lag_path = os.path.join(region_path, '{}_lags.json'.format(hour))

    # if os.path.exists(res_path): shutil.rmtree(res_path)
    # os.mkdir(res_path)
    # if os.path.exists(lag_path): os.remove(lag_path)

    data_path = '/home/ec2-user/data'
    image_paths = glob.glob(data_path + '/*.jpg')
    image_paths.sort()

    lags = {}

    with open('/home/ec2-user/event.json', 'r') as f:
        event = json.load(f)
    keys = event['key']

    t1 = 0
    t2 = 0
    t0 = 0
    t3 = 0

    t0 = time.time()

    client = boto3.client('rekognition', region_name='us-east-2')

    for i in range(hour_ind[hour], hour_ind[hour+1]):
        image_path = data_path + '/' + keys[i]
        image_name = keys[i]
        # response file name pattern: rank_location_imagename.json
        res_filename = '_'.join([str(i), location, image_name]) + '.json'
        res_file_path = os.path.join(res_path, res_filename)

        try:
            with open(image_path, 'rb') as image:
                t1 = time.time()
                response = client.detect_labels(Image={'Bytes': image.read()})
                t2 = time.time()
            print('{} image ok!'.format(i))
        except Exception as e:
            print(e)

        lags[image_name] = t2 - t1

        try:
            with open(res_file_path, 'w') as f:
                json.dump(response, f)
            print('{} response ok!'.format(i))
        except Exception as e:
            print(e)
    # it include the time to store the file
    t3 = time.time()

    lag_all = [t3-t0, lags]

    try:
        with open(lag_path, 'w') as f:
            json.dump(lag_all, f)
        print('lags ok!')
    except Exception as e:
        print(e)


if __name__ == '__main__':
    app.run(main)