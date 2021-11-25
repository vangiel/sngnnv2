import json
import os
import sys
import math
import numpy as np
import copy
from pathlib import Path
from scipy.interpolate import splprep, splev
from shapely.geometry.point import Point
import matplotlib.pyplot as plt


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
        # w['x1'], w['x2'] = w['x2'], w['x1']
        # w['y1'], w['y2'] = w['y2'], w['y1']
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
    print(filename)

    # Read JSON data into the datastore variable
    if filename:
        with open(directory_path + '/' + filename, 'r') as f:
            datastore = json.load(f)
            f.close()

    if len(datastore[0]['people'])!=len(datastore[-1]['people']) or len(datastore[0]['objects'])!=len(datastore[-1]['objects']):
        continue

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

    entity_radius = 0.15
    for i in range(len(datastore)):
        # Robot pose
        if i != 0:
            total_divisions = 100
            extrapolation_amount = 10 / (datastore[i-1]['timestamp'] - datastore[i]['timestamp'])

            x_r = []
            y_r = []
            s = i-2 if i > 2 else 0
            for d in datastore_absolute[s:i]:
                x_r.append(d['robot_pose']['x'])
                y_r.append(d['robot_pose']['y'])
            x_r.append(datastore_absolute[i]['robot_pose']['x'])
            y_r.append(datastore_absolute[i]['robot_pose']['y'])
            x_r = np.array(x_r)
            y_r = np.array(y_r)

            k = 2 if x_r.size > 2 else 1
            tck_r = splprep([x_r, y_r], k=k, s=0)
            ex_r, ey_r = splev(np.linspace(0, extrapolation_amount, total_divisions), tck_r[0][0:3], der=0)

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

            # Calculate time to collision
            if i == 0:
                datastore[i]['people'][j]['t_collision'] = 1.
            else:
                x_p = []
                y_p = []
                for d in datastore_absolute[s:i]:
                    x_p.append(d['people'][j]['x'])
                    y_p.append(d['people'][j]['y'])
                x_p.append(p['x'])
                y_p.append(p['y'])
                x_p = np.array(x_p)
                y_p = np.array(y_p)

                tck_p = splprep([x_p, y_p], k=k, s=0)
                ex_p, ey_p = splev(np.linspace(0, extrapolation_amount, total_divisions), tck_p[0][0:3], der=0)

                collision = False
                for t in range(total_divisions):
                    point1 = Point(ex_p[t], ey_p[t])
                    point2 = Point(ex_r[t], ey_r[t])
                    circle1 = point1.buffer(entity_radius)
                    circle2 = point2.buffer(entity_radius)

                    if circle1.intersects(circle2):
                        collision = True

                    if collision:
                        break

                datastore[i]['people'][j]['t_collision'] = (t+1) / total_divisions

                # print(datastore[i]['people'][j]['t_collision'])
                # if collision:
                #     plt.plot(ex_p, ey_p, 'o', x_p, y_p, 'o', ex_r, ey_r, 'o', x_r, y_r, 'o',  ex_r[t], ey_r[t], 'o')
                #     plt.legend(['spline1', 'data1', 'spline2', 'data2', 'collision'])
                # else:
                #     plt.plot(ex_p, ey_p, 'o', x_p, y_p, 'o', ex_r, ey_r, 'o', x_r, y_r, 'o')
                #     plt.legend(['spline1', 'data1', 'spline2', 'data2'])
                # plt.title("Figure " + str(i))
                # plt.axis([x_r.min() - 5, x_r.max() + 5, y_r.min() - 5, y_r.max() + 5])
                # plt.show()
                #
                # if i == 15:
                #     sys.exit(0)

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

            # Calculate time to collision
            if i == 0:
                datastore[i]['objects'][j]['t_collision'] = 1.
            else:
                x_o = []
                y_o = []
                for d in datastore_absolute[s:i]:
                    x_o.append(d['objects'][j]['x'])
                    y_o.append(d['objects'][j]['y'])
                x_o.append(o['x'])
                y_o.append(o['y'])
                x_o = np.array(x_o)
                y_o = np.array(y_o)

                tck_o = splprep([x_o, y_o], k=k, s=0)
                ex_o, ey_o = splev(np.linspace(0, extrapolation_amount, total_divisions), tck_o[0][0:3], der=0)

                collision = False
                for t in range(total_divisions):
                    point1 = Point(ex_o[t], ey_o[t])
                    point2 = Point(ex_r[t], ey_r[t])
                    circle1 = point1.buffer(entity_radius)
                    circle2 = point2.buffer(entity_radius)

                    if circle1.intersects(circle2):
                        collision = True

                    if collision:
                        break

                datastore[i]['objects'][j]['t_collision'] = (t+1) / total_divisions

                # print(datastore[i]['objects'][j]['t_collision'])
                # if collision:
                #     plt.plot(ex_o, ey_o, 'o', x_o, y_o, 'o', ex_r, ey_r, 'o', x_r, y_r, 'o',  ex_r[t], ey_r[t], 'o')
                #     plt.legend(['spline1', 'data1', 'spline2', 'data2', 'collision'])
                # else:
                #     plt.plot(ex_o, ey_o, 'o', x_o, y_o, 'o', ex_r, ey_r, 'o', x_r, y_r, 'o')
                #     plt.legend(['spline1', 'data1', 'spline2', 'data2'])
                # plt.title("Figure " + str(i))
                # plt.axis([x_r.min() - 5, x_r.max() + 5, y_r.min() - 5, y_r.max() + 5])
                # plt.show()
                #
                # if i == 15:
                #     sys.exit(0)

    with open(dest_directory + '/' + save, 'w') as outfile:
        json.dump(datastore, outfile, indent=4, sort_keys=True)
        outfile.close()

    with open(dest_directory + '_absolute' + '/' + save, 'w') as outfile:
        json.dump(datastore_absolute, outfile, indent=4, sort_keys=True)
        outfile.close()
