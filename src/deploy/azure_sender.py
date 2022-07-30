import json
import time
import os

from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

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

# singapore
# SUB_KEY = ''
# END_POINT = ''

# us east
SUB_KEY = ''
END_POINT = ''

hour_ind = [0, 206, 412, 618, 824, 1030, 1236, 1442, 1648, 1854, 2060, 2266, 2472, 2678, 2884, 3090, 3296, 3502, 3708, 3914, 4120, 4326, 4532, 4738, 4952]

def drawRectangle(object, draw):
    # Represent all sides of a box
    rect = object.rectangle
    left = rect.x
    top = rect.y
    right = left + rect.w
    bottom = top + rect.h
    coordinates = ((left, top), (right, bottom))
    draw.rectangle(coordinates, outline='red')

def getObjects(results, draw):
    # Print results of detection with bounding boxes
    print("OBJECTS DETECTED:")
    if len(results.objects) == 0:
        print("No objects detected.")
    else:
        for object in results.objects:
            print("label", object.object_property)
            print("object at location {}, {}, {}, {}".format(
                object.rectangle.x, object.rectangle.x + object.rectangle.w,
                object.rectangle.y, object.rectangle.y + object.rectangle.h))
            drawRectangle(object, draw)
        print()
        print('Bounding boxes drawn around objects... see popup.')
    print()

def getTags(results):
    # Print results with confidence score
    print("TAGS: ")
    if (len(results.tags) == 0):
        print("No tags detected.")
    else:
        for tag in results.tags:
            print("'{}' with confidence {:.2f}%".format(
                tag.name, tag.confidence * 100))
    print()

def saveResult(results, path):
    print("OBJECTS DETECTED:")
    if len(results.objects) == 0:
        print("No objects detected.")
    else:
        with open(path, 'w') as f:
            if len(results.objects) == 0:
                print("No objects detected.")
            else:
                for obj in results.objects:
                    print("label", obj.object_property)
                    # x,y,x+w,y+h,label
                    box_mess = '-'.join([obj.object_property, str(obj.confidence),
                                         str(obj.rectangle.x), str(obj.rectangle.y),
                                         str(obj.rectangle.x + obj.rectangle.w),
                                         str(obj.rectangle.y + obj.rectangle.h)]) + '\n'
                    f.write(box_mess)
                    print('\t' + str(box_mess).strip())
    return True


def detect_labels_local_file(photo, region, subscription_key, endpoint):
    image_features = ['objects', 'tags']
    # Get local image with different objects in it
    local_image_objects = open(photo, "rb")
    # Opens image to get PIL type of image, for drawing to
    image_l = Image.open(photo)
    draw = ImageDraw.Draw(image_l)

    computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

    t1 = time.time()
    results_local = computervision_client.analyze_image_in_stream(local_image_objects, image_features)
    t2 = time.time()

    return results_local, t2 - t1




def main(argv):
    del argv
    location = FLAGS.location
    region = FLAGS.region
    num = FLAGS.num
    mode = FLAGS.mode

    day = time.strftime("%d", time.localtime())
    hour = int(time.strftime("%H", time.localtime()))

    region_path = '/home/azureuser/' + region + '_' + day

    if not os.path.exists(region_path): os.mkdir(region_path)

    res_path = os.path.join(region_path, 'res')
    if not os.path.exists(res_path): os.mkdir(res_path)

    lag_path = os.path.join(region_path, '{}_lags.json'.format(hour))

    # if os.path.exists(res_path): shutil.rmtree(res_path)
    # os.mkdir(res_path)
    # if os.path.exists(lag_path): os.remove(lag_path)

    data_path = '/home/azureuser/data'
    image_paths = glob.glob(data_path + '/*.jpg')
    image_paths.sort()

    lags = {}

    with open('/home/azureuser/event.json', 'r') as f:
        event = json.load(f)
    keys = event['key']

    t1 = 0
    t2 = 0
    t0 = 0
    t3 = 0

    t0 = time.time()

    computervision_client = ComputerVisionClient(END_POINT, CognitiveServicesCredentials(SUB_KEY))
    image_features = ['objects', 'tags']

    for i in range(hour_ind[hour], hour_ind[hour+1]):
        image_path = data_path + '/' + keys[i]
        image_name = keys[i]
        # response file name pattern: rank_location_imagename.json
        res_filename = '_'.join([str(i), location, image_name]) + '.json'
        res_file_path = os.path.join(res_path, res_filename)

        try:
            with open(image_path, 'rb') as image:
                t1 = time.time()
                results_local = computervision_client.analyze_image_in_stream(image, image_features)
                t2 = time.time()
            print('{} image ok!'.format(i))
        except Exception as e:
            print(e)

        lags[image_name] = t2 - t1

        saveResult(results_local, res_file_path)

    # it includes the time to store the file
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