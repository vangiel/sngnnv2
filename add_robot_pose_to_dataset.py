import json
import os
import sys
import math
import numpy as np
import copy


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

    # datastore_absolute = copy.deepcopy(datastore)
    robot_pose = dict()
    for i, data in enumerate(datastore):
        x, y, a = compute_robot_pose(data['walls'])
        robot_pose['x'] = x
        robot_pose['y'] = y
        robot_pose['a'] = a

        # M = get_transformation_matrix_for_pose(x, y, a)

        data['robot_pose'] = robot_pose

    # for p in datastore_absolute[i]['people']:
    # 	point = np.array([[p['x']], [p['y']], [1.0]], dtype=float)
    # 	point = M.dot(point)
    # 	p['x'] = point[0][0]
    # 	p['y'] = point[1][0]
    # 	p['a'] = math.atan2(math.sin(p['a']+a), math.cos(p['a']+a))
    #
    # for o in datastore_absolute[i]['objects']:
    # 	point = np.array([[o['x']], [o['y']], [1.0]], dtype=float)
    # 	point = M.dot(point)
    # 	o['x'] = point[0][0]
    # 	o['y'] = point[1][0]
    # 	o['a'] = math.atan2(math.sin(o['a']+a), math.cos(o['a']+a))

    # for i in range(len(datastore)-1):
    # 	for j, p in enumerate(datastore[i]['people']):
    # 		p['vx'] = datastore_absolute[i]['people'][j]['x'] - datastore_absolute[i+1]['people'][j]['x']
    # 		p['vy'] = datastore_absolute[i]['people'][j]['y'] - datastore_absolute[i+1]['people'][j]['y']
    #
    # 	for j, o in enumerate(datastore[i]['objects']):
    # 		o['vx'] = datastore_absolute[i]['objects'][j]['x'] - datastore_absolute[i+1]['objects'][j]['x']
    # 		o['vy'] = datastore_absolute[i]['objects'][j]['y'] - datastore_absolute[i+1]['objects'][j]['y']

    with open(dest_directory + '/' + save, 'w') as outfile:
        json.dump(datastore, outfile, indent=4, sort_keys=True)
        outfile.close()
