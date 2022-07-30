import os
import json
import time
from google.cloud import vision
import glob


def save_results(objects, path):
    print("OBJECTS DETECTED:")
    if len(objects) == 0:
        print("No objects detected.")
    else:
        with open(path, 'w') as f:
            if len(objects) == 0:
                print("No objects detected.")
            else:
                for obj in objects:
                    print("label", obj.name)
                    # x,y,x+w,y+h,label
                    left = 10
                    top = 10
                    right = -10
                    down = -10
                    for vertex in obj.bounding_poly.normalized_vertices:
                        print(' - ({}, {})'.format(vertex.x, vertex.y))
                        left = min(left, vertex.x)
                        top = min(top, vertex.y)
                        right = max(right, vertex.x)
                        down = max(down, vertex.y)

                    box_mess = '-'.join([obj.name,
                                         str(obj.score),
                                         str(left),
                                         str(top),
                                         str(right),
                                         str(down)]) + '\n'
                    f.write(box_mess)
                    print('\t' + str(box_mess).strip())
    return True


def google_vision(location, region, cur_user_path):
    hour_ind = [0, 206, 412, 618, 824, 1030, 1236, 1442, 1648, 1854, 2060, 2266, 2472, 2678, 2884, 3090, 3296, 3502,
                3708, 3914, 4120, 4326, 4532, 4738, 4952]

    day = time.strftime("%d", time.localtime())
    hour = int(time.strftime("%H", time.localtime()))

    region_path = cur_user_path + '/' + region + '_' + day
    if not os.path.exists(region_path):
        os.mkdir(region_path)

    res_path = os.path.join(region_path, 'res')
    if not os.path.exists(res_path):
        os.mkdir(res_path)

    lag_path = os.path.join(region_path, '{}_lags.json'.format(hour))

    data_path = os.path.join(cur_user_path, 'data')
    image_paths = glob.glob(data_path + '/*.jpg')
    image_paths.sort()

    with open(cur_user_path + '/event.json', 'r') as f:
        event = json.load(f)
    keys = event['key']

    t1 = 0
    t2 = 0
    t0 = 0
    t3 = 0
    lags = {}

    t0 = time.time()

    client = vision.ImageAnnotatorClient()

    for i in range(hour_ind[hour], hour_ind[hour + 1]):
        image_path = data_path + '/' + keys[i]
        image_name = keys[i]
        # response file name pattern: rank_location_imagename.json
        res_filename = '_'.join([str(i), location, image_name]) + '.json'
        res_file_path = os.path.join(res_path, res_filename)

        t1 = time.time()
        with open(image_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        objects = client.object_localization(image=image).localized_object_annotations
        t2 = time.time()
        print('Number of objects found: {}'.format(len(objects)))
        for object_ in objects:
            print('\n{} (confidence: {})'.format(object_.name, object_.score))
            print('Normalized bounding polygon vertices: ')
            for vertex in object_.bounding_poly.normalized_vertices:
                print(' - ({}, {})'.format(vertex.x, vertex.y))
        print('{} image ok!'.format(i))

        lags[image_name] = t2 - t1

        save_results(objects, res_file_path)

        print('{} response ok!'.format(i))

    # it include the time to store the file

    t3 = time.time()

    lag_all = [t3 - t0, lags]

    try:
        with open(lag_path, 'w') as f:
            json.dump(lag_all, f)
        print('lags ok!')
    except Exception as e:
        print(e)

if __name__ == '__main__':
    google_vision(location='sg', region='sg-sg', cur_user_path='/home/azureuser')