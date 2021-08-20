import json
import os
import sys
import math
import numpy as np
import copy
from pathlib import Path


def get_transformation_matrix_for_pose(x, z, angle):
    M = np.zeros((3, 3))
    M[0][0] = +math.cos(-angle)
    M[0][1] = -math.sin(-angle)
    M[0][2] = x
    M[1][0] = +math.sin(-angle)
    M[1][1] = +math.cos(-angle)
    M[1][2] = z
    M[2][2] = 1.0
    return M


def compute_robot_pose(walls):
    if len(walls) == 4:
        w = walls[3]
        vw = (walls[0]['x2'] - walls[0]['x1'], walls[0]['y2'] - walls[0]['y1'])
        l = math.sqrt(vw[0] * vw[0] + vw[1] * vw[1])
    else:
        w = walls[5]
        w['x1'], w['x2'] = w['x2'], w['x1']
        w['y1'], w['y2'] = w['y2'], w['y1']
        vw = (walls[3]['x2'] - walls[3]['x1'], walls[3]['y2'] - walls[3]['y1'])
        l = math.sqrt(vw[0] * vw[0] + vw[1] * vw[1])

    ang = math.atan2(w['y2'] - w['y1'], w['x2'] - w['x1'])
    p = np.array([[(w['x1'] + w['x2']) / 2.], [(w['y1'] + w['y2']) / 2.], [1.0]], dtype=float)
    M = get_transformation_matrix_for_pose(0, 0, ang)
    p = M.dot(p)
    p[0][0] = -p[0][0]
    p[1][0] = l / 2. - p[1][0]
    return p[0][0], p[1][0], ang


if len(sys.argv) < 3:
    print("USAGE: python3 add_robot_pose_to_dataset.py old_file_or_directory new_directory")
    exit()

directory_path = sys.argv[1]
dest_directory = sys.argv[2]

Path(dest_directory).mkdir(parents=True, exist_ok=True)
Path(dest_directory + '_absolute').mkdir(parents=True, exist_ok=True)

if os.path.isfile(directory_path):
    filename = os.path.basename(directory_path)
    directory_path = directory_path.split(filename)[0]
    fileList = [filename]
else:
    fileList = os.listdir(directory_path)

for filename in fileList:
    if not filename.endswith('.json'):
        continue

    save = filename

    # Read JSON data into the datastore variable
    if filename:
        with open(directory_path + '/' + filename, 'r') as f:
            datastore = json.load(f)
            f.close()

    datastore_absolute = copy.deepcopy(datastore)
    robot_pose = dict()
    x, y, a = compute_robot_pose(datastore[0]['walls'])
    M = get_transformation_matrix_for_pose(x, y, a)
    M0 = np.linalg.inv(M)
    for i, data in enumerate(datastore):
        x, y, a = compute_robot_pose(data['walls'])
        robot_pose['x'] = x
        robot_pose['y'] = y
        robot_pose['a'] = a

        M = get_transformation_matrix_for_pose(x, y, a)
        M = np.dot(M0, M)

        data['robot_pose'] = copy.deepcopy(robot_pose)
        datastore_absolute[i]['robot_pose'] = copy.deepcopy(robot_pose)

        for g in datastore_absolute[i]['goal']:
            point = np.array([[g['x']], [g['y']], [1.0]], dtype=float)
            point = M.dot(point)
            g['x'] = point[0][0]
            g['y'] = point[1][0]

        for p in datastore_absolute[i]['people']:
            point = np.array([[p['x']], [p['y']], [1.0]], dtype=float)
            point = M.dot(point)
            p['x'] = point[0][0]
            p['y'] = point[1][0]
            p['a'] = math.atan2(math.sin(p['a'] + a), math.cos(p['a'] + a))

        for o in datastore_absolute[i]['objects']:
            point = np.array([[o['x']], [o['y']], [1.0]], dtype=float)
            point = M.dot(point)
            o['x'] = point[0][0]
            o['y'] = point[1][0]
            o['a'] = math.atan2(math.sin(o['a'] + a), math.cos(o['a'] + a))

        for w in datastore_absolute[i]['walls']:
            point1 = np.array([[w['x1']], [w['y1']], [1.0]], dtype=float)
            point1 = M.dot(point1)
            point2 = np.array([[w['x2']], [w['y2']], [1.0]], dtype=float)
            point2 = M.dot(point2)
            w['x1'] = point1[0][0]
            w['y1'] = point1[1][0]
            w['x2'] = point2[0][0]
            w['y2'] = point2[1][0]

    for i in range(len(datastore)):
        for j, p in enumerate(datastore_absolute[i]['people']):
            if i == 0:
                p['vx'] = 0.0
                p['vy'] = 0.0
                p['va'] = 0.0
            # elif j == len(datastore_absolute[i]['people']):
            #     break
            else:
                p['vx'] = datastore_absolute[i]['people'][j]['x'] - datastore_absolute[i-1]['people'][j]['x']
                p['vy'] = datastore_absolute[i]['people'][j]['y'] - datastore_absolute[i-1]['people'][j]['y']
                p['va'] = datastore_absolute[i]['people'][j]['a'] - datastore_absolute[i-1]['people'][j]['a']

        for j, o in enumerate(datastore_absolute[i]['objects']):
            if i == 0:
                o['vx'] = 0.0
                o['vy'] = 0.0
                o['va'] = 0.0
            # elif j == len(datastore_absolute[i]['objects']):
            #     break
            else:
                o['vx'] = datastore_absolute[i]['objects'][j]['x'] - datastore_absolute[i-1]['objects'][j]['x']
                o['vy'] = datastore_absolute[i]['objects'][j]['y'] - datastore_absolute[i-1]['objects'][j]['y']
                o['va'] = datastore_absolute[i]['objects'][j]['a'] - datastore_absolute[i-1]['objects'][j]['a']

    with open(dest_directory + '/' + save, 'w') as outfile:
        json.dump(datastore, outfile, indent=4, sort_keys=True)
        outfile.close()

    with open(dest_directory + '_absolute' + '/' + save, 'w') as outfile:
        json.dump(datastore_absolute, outfile, indent=4, sort_keys=True)
        outfile.close()
