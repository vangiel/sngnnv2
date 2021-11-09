import copy
from collections import namedtuple
import math

import torch as th
import dgl
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info
import numpy as np

N_INTERVALS = 3
FRAMES_INTERVAL = 1.
WITH_EDGE_FEATURES = True

grid_width = 11  # 30 #18
output_width = 73  # 121 #73
area_width = 1000.  # horizontal/vertical distance between two contiguous nodes of the grid. Previously it was taken as the spatial area of the grid

threshold_human_wall = 1.5
limit = 50000  # Limit of graphs to load
path_saves = 'saves/'  # This variable is necessary due to a bug in dgl.DGLDataset source code
graphData = namedtuple('graphData', ['src_nodes', 'dst_nodes', 'n_nodes', 'features', 'edge_feats', 'edge_types',
                                     'edge_norms', 'position_by_id', 'typeMap', 'labels', 'w_segments'])


#  human to wall distance
def dist_h_w(h, wall):
    if 'xPos' in h.keys():
        hxpos = float(h['xPos']) / 100.
        hypos = float(h['yPos']) / 100.
    else:
        hxpos = float(h['x'])
        hypos = float(h['y'])

    wxpos = float(wall.xpos) / 100.
    wypos = float(wall.ypos) / 100.
    return math.sqrt((hxpos - wxpos) * (hxpos - wxpos) + (hypos - wypos) * (hypos - wypos))


# return de number of grid nodes if a grid is used in the specified alternative
def grid_nodes_number(alt):
    if alt == '2' or alt == '7' or alt == '8':
        return grid_width * grid_width
    else:
        return 0


def central_grid_nodes(alt, r):
    if alt == '7' or alt == '8':
        grid_node_ids = np.zeros((grid_width, grid_width), dtype=int)
        for y in range(grid_width):
            for x in range(grid_width):
                grid_node_ids[x][y] = y * grid_width + x
        central_nodes = closest_grid_nodes(grid_node_ids, area_width, grid_width, r, 0, 0)
        return central_nodes
    else:
        return []


# Calculate the closet node in the grid to a given node by its coordinates
def closest_grid_node(grid_ids, w_a, w_i, x, y):
    c_x = int(round(x / w_a) + (w_i // 2))
    if c_x < 0: c_x = 0
    if c_x >= grid_width: c_x = grid_width - 1
    c_y = int(round(y / w_a) + (w_i // 2))
    if c_y < 0: c_y = 0
    if c_y >= grid_width: c_y = grid_width - 1
    return grid_ids[c_x][c_y]

    # if 0 <= c_x < grid_width and 0 <= c_y < grid_width:
    #     return grid_ids[c_x][c_y]
    # return None


def closest_grid_nodes(grid_ids, w_a, w_i, r, x, y):
    c_x = int(round(x / w_a) + (w_i // 2))
    c_y = int(round(y / w_a) + (w_i // 2))
    cols, rows = (int(math.ceil(r / w_a)), int(math.ceil(r / w_a)))
    rangeC = list(range(-cols, cols + 1))
    rangeR = list(range(-rows, rows + 1))
    p_arr = [[c, r] for c in rangeC for r in rangeR]
    grid_nodes = []
    # r_g = r / w_a
    for p in p_arr:
        g_x, g_y = c_x + p[0], c_y + p[1]
        gw_x, gw_y = (g_x - w_i // 2) * w_a, (g_y - w_i // 2) * w_a
        if math.sqrt((gw_x - x) * (gw_x - x) + (gw_y - y) * (gw_y - y)) <= r:
            # if math.sqrt(p[0] * p[0] + p[1] * p[1]) <= r_g:
            if 0 <= g_x < grid_width and 0 <= g_y < grid_width:
                grid_nodes.append(grid_ids[g_x][g_y])

    return grid_nodes


def get_relations(alt):
    rels = None
    if alt == '1':
        rels = {'p_r', 'o_r', 'l_r', 'l_p', 'l_o', 'p_p', 'p_o', 'w_l', 'w_p'}
        # p = person
        # r = robot
        # l = room (lounge)
        # o = object
        # w = wall
        # n = node (generic)
        for e in list(rels):
            rels.add(e[::-1])
        rels.add('self')
        rels = sorted(list(rels))
    elif alt == '2':
        room_set = {'l_p', 'l_o', 'l_w', 'l_g', 'p_p', 'p_o', 'p_g', 'o_g', 'w_g'}
        grid_set = {'g_c', 'g_ri', 'g_le', 'g_u', 'g_d', 'g_uri', 'g_dri', 'g_ule', 'g_dle'}
        # ^
        # |_p = person             g_ri = grid right
        # |_w = wall               g_le = grid left
        # |_l = lounge             g_u = grid up
        # |_o = object             g_d = grid down
        # |_g = grid node
        self_edges_set = {'P', 'O', 'W', 'L'}

        for e in list(room_set):
            room_set.add(e[::-1])
        relations_class = room_set | grid_set | self_edges_set
        rels = sorted(list(relations_class))
    elif alt == '3':
        rels = {'p_r', 'o_r', 'p_p', 'p_o', 'w_r', 't_r', 'w_p'}  # add 'w_w' for links between wall nodes
        # p = person
        # r = room
        # o = object
        # w = wall
        # t = goal
        for e in list(rels):
            rels.add(e[::-1])
        rels.add('self')
        rels = sorted(list(rels))
    elif alt == '4':
        rels = {'o_r', 'g_r'}
        # r = room
        # o = object
        # g = goal
        for e in list(rels):
            rels.add(e[::-1])
        rels.add('self')
        rels = sorted(list(rels))
    elif alt == '5' or alt == '6':
        rels = {'p_r', 't_r'}
        # r = room
        # p = person
        # t = goal (target)
        for e in list(rels):
            rels.add(e[::-1])
        rels.add('self')
        rels = sorted(list(rels))
    elif alt == '7':
        rels = {'p_r', 't_r', 'w_r', 'p_g', 't_g', 'r_g', 'w_g'}
        grid_set = {'g_c', 'g_ri', 'g_le', 'g_u', 'g_d', 'g_uri', 'g_dri', 'g_ule', 'g_dle'}
        # r = room
        # p = person
        # t = goal (target)
        # w = wall
        # g = grid
        for e in list(rels):
            rels.add(e[::-1])
        rels.add('self')
        rels = rels | grid_set
        rels = sorted(list(rels))
    elif alt == '8' or alt == '9':
        rels = {'p_r', 'o_r', 'p_p', 'p_o', 't_r', 'w_r', 'p_g', 'o_g', 't_g', 'r_g', 'w_g'}
        grid_set = {'g_c', 'g_ri', 'g_le', 'g_u', 'g_d', 'g_uri', 'g_dri', 'g_ule', 'g_dle'}
        # r = room
        # p = person
        # o = object
        # t = goal (target)
        # w = wall
        # g = grid
        for e in list(rels):
            rels.add(e[::-1])
        rels.add('self')
        rels = rels | grid_set
        rels = sorted(list(rels))

    num_rels = len(rels)

    return rels, num_rels


def get_features(alt):
    all_features = None
    if alt == '1':
        node_types_one_hot = ['robot', 'human', 'object', 'room', 'wall']
        human_metric_features = ['hum_distance', 'hum_distance2', 'hum_angle_sin', 'hum_angle_cos',
                                 'hum_orientation_sin', 'hum_orientation_cos', 'hum_robot_sin',
                                 'hum_robot_cos']
        object_metric_features = ['obj_distance', 'obj_distance2', 'obj_angle_sin', 'obj_angle_cos',
                                  'obj_orientation_sin', 'obj_orientation_cos']
        room_metric_features = ['room_min_human', 'room_min_human2', 'room_humans', 'room_humans2']
        wall_metric_features = ['wall_distance', 'wall_distance2', 'wall_angle_sin', 'wall_angle_cos',
                                'wall_orientation_sin', 'wall_orientation_cos']
        all_features = node_types_one_hot + human_metric_features + object_metric_features + room_metric_features + \
                       wall_metric_features
    elif alt == '2':
        node_types_one_hot = ['human', 'object', 'room', 'wall', 'grid']
        human_metric_features = ['hum_x_pos', 'hum_y_pos', 'hum_orientation_sin', 'hum_orientation_cos']
        object_metric_features = ['obj_x_pos', 'obj_y_pos', 'obj_orientation_sin', 'obj_orientation_cos']
        room_metric_features = ['room_humans', 'room_humans2']
        wall_metric_features = ['wall_x_pos', 'wall_y_pos', 'wall_orientation_sin', 'wall_orientation_cos']
        grid_metric_features = ['grid_x_pos', 'grid_y_pos']  # , 'flag_inside_room']  # , 'count']
        all_features = node_types_one_hot + human_metric_features + object_metric_features + room_metric_features + \
                       wall_metric_features + grid_metric_features
    elif alt == '3':
        time_one_hot = ['is_t_0', 'is_t_m1', 'is_t_m2']
        # time_sequence_features = ['is_first_frame', 'time_left']
        human_metric_features = ['hum_x_pos', 'hum_y_pos', 'human_a_vel', 'human_x_vel', 'human_y_vel',
                                 'hum_orientation_sin', 'hum_orientation_cos',
                                 'hum_dist', 'hum_inv_dist']
        object_metric_features = ['obj_x_pos', 'obj_y_pos', 'obj_a_vel', 'obj_x_vel', 'obj_y_vel',
                                  'obj_orientation_sin', 'obj_orientation_cos',
                                  'obj_x_size', 'obj_y_size',
                                  'obj_dist', 'obj_inv_dist']
        room_metric_features = ['room_humans', 'room_humans2']
        robot_features = ['robot_adv_vel', 'robot_rot_vel']
        wall_metric_features = ['wall_x_pos', 'wall_y_pos', 'wall_orientation_sin', 'wall_orientation_cos',
                                'wall_dist', 'wall_inv_dist']
        goal_metric_features = ['goal_x_pos', 'goal_y_pos', 'goal_dist', 'goal_inv_dist']
        grid_metric_features = ['grid_x_pos', 'grid_y_pos']
        node_types_one_hot = ['human', 'object', 'room', 'wall', 'goal']
        all_features = node_types_one_hot + time_one_hot + human_metric_features + robot_features + \
                       object_metric_features + room_metric_features + wall_metric_features + goal_metric_features
    elif alt == '4':
        object_metric_features = ['obj_x_pos', 'obj_y_pos', 'obj_a_vel', 'obj_x_vel', 'obj_y_vel',
                                  'obj_orientation_sin', 'obj_orientation_cos',
                                  'obj_x_size', 'obj_y_size',
                                  'obj_dist', 'obj_inv_dist']
        robot_features = ['robot_adv_vel', 'robot_rot_vel']
        goal_metric_features = ['goal_x_pos', 'goal_y_pos', 'goal_dist', 'goal_inv_dist']
        node_types_one_hot = ['object', 'room', 'goal']
        all_features = node_types_one_hot + robot_features + \
                       object_metric_features + goal_metric_features
        if N_INTERVALS > 1:
            time_one_hot = ['is_t_0']
            for i in range(1, N_INTERVALS):
                time_one_hot.append('is_t_m' + str(i))
            all_features += time_one_hot
    elif alt == '5':
        human_metric_features = ['hum_x_pos', 'hum_y_pos', 'human_a_vel', 'human_x_vel', 'human_y_vel',
                                 'hum_orientation_sin', 'hum_orientation_cos',
                                 'hum_dist', 'hum_inv_dist']
        robot_features = ['robot_adv_vel', 'robot_rot_vel']
        goal_metric_features = ['goal_x_pos', 'goal_y_pos', 'goal_dist', 'goal_inv_dist']
        node_types_one_hot = ['human', 'room', 'goal']
        all_features = node_types_one_hot + robot_features + \
                       human_metric_features + goal_metric_features
        if N_INTERVALS > 1:
            time_one_hot = ['is_t_0']
            for i in range(1, N_INTERVALS):
                time_one_hot.append('is_t_m' + str(i))
            all_features += time_one_hot
    elif alt == '6':
        human_metric_features = ['hum_x_pos', 'hum_y_pos', 'human_a_vel', 'human_x_vel', 'human_y_vel',
                                 'hum_orientation_sin', 'hum_orientation_cos',
                                 'hum_dist', 'hum_inv_dist']
        robot_features = ['robot_adv_vel', 'robot_rot_vel']
        goal_metric_features = ['goal_x_pos', 'goal_y_pos', 'goal_dist', 'goal_inv_dist']
        step_features = ['step_fraction']
        all_features_1_instant = robot_features + human_metric_features + goal_metric_features + step_features
        all_features = copy.deepcopy(all_features_1_instant)

        # One hot time features
        time_features = ["t0"]
        for i in range(1, N_INTERVALS):
            all_features += [f + '_t' + str(i) for f in all_features_1_instant]
            time_features.append('t' + str(i))

        all_features += time_features
        node_types_one_hot = ['human', 'room', 'goal']
        all_features += node_types_one_hot

    elif alt == '7':
        human_metric_features = ['hum_x_pos', 'hum_y_pos', 'human_a_vel', 'human_x_vel', 'human_y_vel',
                                 'hum_orientation_sin', 'hum_orientation_cos',
                                 'hum_dist', 'hum_inv_dist']
        wall_metric_features = ['wall_x_pos', 'wall_y_pos', 'wall_orientation_sin', 'wall_orientation_cos',
                                'wall_dist', 'wall_inv_dist']
        robot_features = ['robot_adv_vel', 'robot_rot_vel']
        goal_metric_features = ['goal_x_pos', 'goal_y_pos', 'goal_dist', 'goal_inv_dist']
        step_features = ['step_fraction']
        all_features_1_instant = robot_features + human_metric_features + \
                                 wall_metric_features + goal_metric_features + step_features
        all_features = copy.deepcopy(all_features_1_instant)

        # One hot time features
        time_features = ["t0"]
        for i in range(1, N_INTERVALS):
            all_features += [f + '_t' + str(i) for f in all_features_1_instant]
            time_features.append('t' + str(i))

        all_features += time_features
        node_types_one_hot = ['human', 'room', 'goal', 'wall', 'grid']
        grid_metric_features = ['grid_x_pos', 'grid_y_pos']
        all_features += node_types_one_hot + grid_metric_features

    elif alt == '8' or alt == '9':
        human_metric_features = ['hum_x_pos', 'hum_y_pos', 'human_a_vel', 'human_x_vel', 'human_y_vel',
                                 'hum_orientation_sin', 'hum_orientation_cos',
                                 'hum_dist', 'hum_inv_dist']
        object_metric_features = ['obj_x_pos', 'obj_y_pos', 'obj_a_vel', 'obj_x_vel', 'obj_y_vel',
                                  'obj_orientation_sin', 'obj_orientation_cos',
                                  'obj_x_size', 'obj_y_size',
                                  'obj_dist', 'obj_inv_dist']
        wall_metric_features = ['wall_x_pos', 'wall_y_pos', 'wall_orientation_sin', 'wall_orientation_cos',
                                'wall_dist', 'wall_inv_dist']
        robot_features = ['robot_adv_vel', 'robot_rot_vel']
        goal_metric_features = ['goal_x_pos', 'goal_y_pos', 'goal_dist', 'goal_inv_dist']
        step_features = ['step_fraction']
        all_features_1_instant = robot_features + human_metric_features + object_metric_features + \
                                 wall_metric_features + goal_metric_features + step_features
        all_features = copy.deepcopy(all_features_1_instant)

        # One hot time features
        time_features = ["t0"]
        for i in range(1, N_INTERVALS):
            all_features += [f + '_t' + str(i) for f in all_features_1_instant]
            time_features.append('t' + str(i))

        all_features += time_features
        node_types_one_hot = ['human', 'object', 'room', 'goal', 'wall', 'grid']
        grid_metric_features = ['grid_x_pos', 'grid_y_pos']
        all_features += node_types_one_hot + grid_metric_features

    feature_dimensions = len(all_features)

    return all_features, feature_dimensions

# Same as alternative 9 but with relative position, OH and time to collision in the edge features
def initializeAlt10(data_sequence, alt='8', w_segments=[]):
    # Initialize variables
    rels, num_rels = get_relations(alt)
    edge_types = []  # List to store the relation of each edge
    edge_norms = []  # List to store the norm of each edge

    # Compute the number of nodes
    # one for the robot  + humans  + one for the goal
    n_nodes = 1 + len(data_sequence[0]['people']) + len(data_sequence[0]['objects']) + 1

    walls, w_segments = generate_walls_information(data_sequence[0], w_segments)
    n_nodes += len(walls)

    # Feature dimensions
    all_features, n_features = get_features(alt)
    # print(all_features, n_features)
    features = th.zeros(n_nodes, n_features)
    edge_feats_list = []

    # Nodes variables
    typeMap = dict()
    position_by_id = {}
    src_nodes = []  # List to store source nodes
    dst_nodes = []  # List to store destiny nodes

    # Labels
    if 'label_Q1' in data_sequence[0].keys():
        labels = np.array([float(data_sequence[0]['label_Q1']), float(data_sequence[0]['label_Q2'])])
    else:
        labels = np.array([0, 0])
    labels[0] = labels[0] / 100.
    labels[1] = labels[1] / 100.

    t_tag = ['']
    for i in range(1, N_INTERVALS):
        t_tag.append('_t' + str(i))

    if 'step_fraction' in data_sequence[0].keys():
        step_fraction = data_sequence[0]['step_fraction']
    else:
        step_fraction = 0

    n_instants = 0
    frames_in_interval = []
    first_frame = True
    for data in data_sequence:
        if n_instants == N_INTERVALS:
            break
        if not first_frame and math.fabs(
                data['timestamp'] - frames_in_interval[-1]['timestamp']) < FRAMES_INTERVAL:  # Truncated to N seconds
            continue

        if 'step_fraction' in data.keys():
            step_fraction = data['step_fraction']
        else:
            step_fraction = 0

        frames_in_interval.append(data)

        max_used_id = 0  # Initialise id counter (0 for the robot)
        # room (id 0)
        room_id = 0

        if first_frame:
            typeMap[room_id] = 'r'  # 'r' for 'room'
            position_by_id[room_id] = [0, 0]
            features[room_id, all_features.index('room')] = 1.

        features[room_id, all_features.index('step_fraction' + t_tag[n_instants])] = step_fraction
        features[room_id, all_features.index('robot_adv_vel' + t_tag[n_instants])] = data['command'][0] / MAX_ADV
        features[room_id, all_features.index('robot_rot_vel' + t_tag[n_instants])] = data['command'][2] / MAX_ROT
        features[room_id, all_features.index('t' + str(n_instants))] = 1.

        max_used_id += 1

        # objects
        for o in data['objects']:
            o_id = o['id']

            xpos = o['x'] / 10.
            ypos = o['y'] / 10.

            dist = math.sqrt(xpos ** 2 + ypos ** 2)
            va = o['va'] / 10.
            vx = o['vx'] / 10.
            vy = o['vy'] / 10.
            orientation = o['a']

            if first_frame:
                src_nodes.append(o_id)
                dst_nodes.append(room_id)
                edge_types.append(rels.index('o_r'))
                edge_norms.append([1.])

                src_nodes.append(room_id)
                dst_nodes.append(o_id)
                edge_types.append(rels.index('r_o'))
                edge_norms.append([1.])
                # Edge features
                edge_features = th.zeros(num_rels + 4)
                edge_features[rels.index('o_r')] = 1
                edge_features[-1] = dist
                edge_feats_list.append(edge_features)

                edge_features = th.zeros(num_rels + 4)
                edge_features[rels.index('r_o')] = 1
                edge_features[-1] = dist
                edge_feats_list.append(edge_features)

                typeMap[o_id] = 'o'  # 'o' for 'object'
                position_by_id[o_id] = [xpos, ypos]
                features[o_id, all_features.index('object')] = 1

            max_used_id += 1

            features[o_id, all_features.index('step_fraction' + t_tag[n_instants])] = step_fraction
            features[o_id, all_features.index('obj_orientation_sin' + t_tag[n_instants])] = math.sin(orientation)
            features[o_id, all_features.index('obj_orientation_cos' + t_tag[n_instants])] = math.cos(orientation)
            features[o_id, all_features.index('obj_x_pos' + t_tag[n_instants])] = xpos
            features[o_id, all_features.index('obj_y_pos' + t_tag[n_instants])] = ypos
            features[o_id, all_features.index('obj_a_vel' + t_tag[n_instants])] = va
            features[o_id, all_features.index('obj_x_vel' + t_tag[n_instants])] = vx
            features[o_id, all_features.index('obj_y_vel' + t_tag[n_instants])] = vy
            features[o_id, all_features.index('obj_x_size' + t_tag[n_instants])] = o['size_x']
            features[o_id, all_features.index('obj_y_size' + t_tag[n_instants])] = o['size_y']
            features[o_id, all_features.index('obj_dist' + t_tag[n_instants])] = dist
            features[o_id, all_features.index('obj_inv_dist' + t_tag[n_instants])] = 1. - dist  # /(1.+dist*10.)

        # humans
        for h in data['people']:
            h_id = h['id']

            xpos = h['x'] / 10.
            ypos = h['y'] / 10.
            dist = math.sqrt(xpos ** 2 + ypos ** 2)
            va = h['va'] / 10.
            vx = h['vx'] / 10.
            vy = h['vy'] / 10.
            orientation = h['a']

            if first_frame:
                src_nodes.append(h_id)
                dst_nodes.append(room_id)
                edge_types.append(rels.index('p_r'))
                edge_norms.append([1. / len(data['people'])])

                src_nodes.append(room_id)
                dst_nodes.append(h_id)
                edge_types.append(rels.index('r_p'))
                edge_norms.append([1.])

                # Edge features
                edge_features = th.zeros(num_rels + 4)
                edge_features[rels.index('p_r')] = 1
                edge_features[-1] = dist
                edge_feats_list.append(edge_features)

                edge_features = th.zeros(num_rels + 4)
                edge_features[rels.index('r_p')] = 1
                edge_features[-1] = dist
                edge_feats_list.append(edge_features)

                typeMap[h_id] = 'p'  # 'p' for 'person'
                position_by_id[h_id] = [xpos, ypos]
                features[h_id, all_features.index('human')] = 1.

            max_used_id += 1

            features[h_id, all_features.index('step_fraction' + t_tag[n_instants])] = step_fraction
            features[h_id, all_features.index('hum_orientation_sin' + t_tag[n_instants])] = math.sin(orientation)
            features[h_id, all_features.index('hum_orientation_cos' + t_tag[n_instants])] = math.cos(orientation)
            features[h_id, all_features.index('hum_x_pos' + t_tag[n_instants])] = xpos
            features[h_id, all_features.index('hum_y_pos' + t_tag[n_instants])] = ypos
            features[h_id, all_features.index('human_a_vel' + t_tag[n_instants])] = va
            features[h_id, all_features.index('human_x_vel' + t_tag[n_instants])] = vx
            features[h_id, all_features.index('human_y_vel' + t_tag[n_instants])] = vy
            features[h_id, all_features.index('hum_dist' + t_tag[n_instants])] = dist
            features[h_id, all_features.index('hum_inv_dist' + t_tag[n_instants])] = 1. - dist  # /(1.+dist*10.)
            features[h_id, all_features.index('t' + str(n_instants))] = 1.

        # Goal
        goal_id = max_used_id
        max_used_id += 1

        xpos = data['goal'][0]['x'] / 10.
        ypos = data['goal'][0]['y'] / 10.
        dist = math.sqrt(xpos ** 2 + ypos ** 2)

        if first_frame:
            typeMap[goal_id] = 't'  # 't' for 'goal'
            src_nodes.append(goal_id)
            dst_nodes.append(room_id)
            edge_types.append(rels.index('t_r'))
            edge_norms.append([1.])
            # edge_norms.append([1. / len(data['objects'])])

            src_nodes.append(room_id)
            dst_nodes.append(goal_id)
            edge_types.append(rels.index('r_t'))
            edge_norms.append([1.])

            # Edge features
            edge_features = th.zeros(num_rels + 4)
            edge_features[rels.index('t_r')] = 1
            edge_features[-1] = dist
            edge_feats_list.append(edge_features)

            edge_features = th.zeros(num_rels + 4)
            edge_features[rels.index('r_t')] = 1
            edge_features[-1] = dist
            edge_feats_list.append(edge_features)

            position_by_id[goal_id] = [xpos, ypos]

            features[goal_id, all_features.index('goal')] = 1

        features[goal_id, all_features.index('step_fraction' + t_tag[n_instants])] = step_fraction
        features[goal_id, all_features.index('goal_x_pos' + t_tag[n_instants])] = xpos
        features[goal_id, all_features.index('goal_y_pos' + t_tag[n_instants])] = ypos
        features[goal_id, all_features.index('goal_dist' + t_tag[n_instants])] = dist
        features[goal_id, all_features.index('goal_inv_dist' + t_tag[n_instants])] = 1. - dist  # /(1.+dist*10.)
        features[goal_id, all_features.index('t' + str(n_instants))] = 1.

        # Walls
        if not first_frame:
            walls, w_segments = generate_walls_information(data, w_segments)

        for wall in walls:
            wall_id = max_used_id
            max_used_id += 1

            if first_frame:
                typeMap[wall_id] = 'w'  # 'w' for 'walls'

                dist = math.sqrt((wall.xpos / 1000.) ** 2 + (wall.ypos / 1000.) ** 2)

                # Links to room node
                src_nodes.append(wall_id)
                dst_nodes.append(room_id)
                edge_types.append(rels.index('w_r'))
                edge_norms.append([1. / len(walls)])

                src_nodes.append(room_id)
                dst_nodes.append(wall_id)
                edge_types.append(rels.index('r_w'))
                edge_norms.append([1.])

                # Edge features
                edge_features = th.zeros(num_rels + 4)
                edge_features[rels.index('w_r')] = 1
                edge_features[-1] = dist
                edge_feats_list.append(edge_features)

                edge_features = th.zeros(num_rels + 4)
                edge_features[rels.index('r_w')] = 1
                edge_features[-1] = dist
                edge_feats_list.append(edge_features)

                position_by_id[wall_id] = [wall.xpos / 1000., wall.ypos / 1000.]
                features[wall_id, all_features.index('wall')] = 1.

            features[wall_id, all_features.index('step_fraction' + t_tag[n_instants])] = step_fraction
            features[wall_id, all_features.index('wall_orientation_sin' + t_tag[n_instants])] = math.sin(
                wall.orientation)
            features[wall_id, all_features.index('wall_orientation_cos' + t_tag[n_instants])] = math.cos(
                wall.orientation)
            features[wall_id, all_features.index('wall_x_pos' + t_tag[n_instants])] = wall.xpos / 1000.
            features[wall_id, all_features.index('wall_y_pos' + t_tag[n_instants])] = wall.ypos / 1000.
            features[wall_id, all_features.index('wall_dist' + t_tag[n_instants])] = dist
            features[wall_id, all_features.index('wall_inv_dist' + t_tag[n_instants])] = 1. - dist  # 1./(1.+dist*10.)
            features[wall_id, all_features.index('t' + str(n_instants))] = 1.

        n_instants += 1
        first_frame = False

    # interaction links
    for link in data['interaction']:
        typeLdir = typeMap[link['src']] + '_' + typeMap[link['dst']]
        typeLinv = typeMap[link['dst']] + '_' + typeMap[link['src']]

        dist = math.sqrt((position_by_id[link['src']][0] - position_by_id[link['dst']][0]) ** 2 +
                         (position_by_id[link['src']][1] - position_by_id[link['dst']][1]) ** 2)

        src_nodes.append(link['src'])
        dst_nodes.append(link['dst'])
        edge_types.append(rels.index(typeLdir))
        edge_norms.append([1.])

        src_nodes.append(link['dst'])
        dst_nodes.append(link['src'])
        edge_types.append(rels.index(typeLinv))
        edge_norms.append([1.])

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index(typeLdir)] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index(typeLinv)] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

    # self edges
    for node_id in range(n_nodes):
        src_nodes.append(node_id)
        dst_nodes.append(node_id)
        edge_types.append(rels.index('self'))
        edge_norms.append([1.])

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('self')] = 1
        edge_features[-1] = 0
        edge_feats_list.append(edge_features)

    # Convert outputs to tensors
    src_nodes = th.LongTensor(src_nodes)
    dst_nodes = th.LongTensor(dst_nodes)

    edge_types = th.LongTensor(edge_types)
    edge_norms = th.Tensor(edge_norms)

    edge_feats = th.stack(edge_feats_list)

    return src_nodes, dst_nodes, n_nodes, features, edge_feats, edge_types, edge_norms, position_by_id, typeMap, \
           labels, []