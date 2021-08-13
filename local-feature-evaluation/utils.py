# Adapted from https://github.com/ahojnnes/local-feature-evaluation/blob/master/scripts/reconstruction_pipeline.py.
# Copyright 2017, Johannes L. Schoenberger <jsch at inf.ethz.ch>.
import numpy as np

import os

import subprocess

import sqlite3

import torch

from tqdm import tqdm


def mnn_ratio_matcher(descriptors1, descriptors2, ratio=0.9):
    # Mutual NN + symmetric Lowe's ratio test matcher.
    descriptors1 = torch.from_numpy(np.array(descriptors1)).float().cuda()
    descriptors2 = torch.from_numpy(np.array(descriptors2)).float().cuda()

    # L2 normalize descriptors.
    descriptors1 /= torch.norm(descriptors1, dim=-1).unsqueeze(-1)
    descriptors2 /= torch.norm(descriptors2, dim=-1).unsqueeze(-1)

    # Similarity matrix.
    device = descriptors1.device
    sim = descriptors1 @ descriptors2.t()

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim, 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios12 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN and match similarity.
    nn12 = nns[:, 0]
    match_sim = nns_sim[:, 0]

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim.t(), 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios21 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN.
    nn21 = nns[:, 0]
    
    # Mutual NN + symmetric ratio test.
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = torch.min(ids1 == nn21[nn12], torch.min(ratios12 <= ratio, ratios21[nn12] <= ratio))
    
    # Final matches.
    matches = torch.stack([ids1[mask], nn12[mask]], dim=-1)
    match_sim = match_sim[mask]

    return (
        matches.data.cpu().numpy(),
        match_sim.data.cpu().numpy()
    )


def translate_descriptors(descriptors, source_feature, target_feature, encoders, decoders, batch_size, device):
    descriptors_torch = torch.from_numpy(descriptors).to(device)
    start_idx = 0
    translated_descriptors_torch = torch.zeros([0, 128]).to(device)
    while start_idx < descriptors_torch.shape[0]:
        aux = encoders[source_feature](descriptors_torch[start_idx : start_idx + batch_size])
        if target_feature is not None:
            aux = decoders[target_feature](aux)
        translated_descriptors_torch = torch.cat([translated_descriptors_torch, aux], dim=0)
        start_idx += batch_size
    translated_descriptors = translated_descriptors_torch.cpu().numpy()
    return translated_descriptors


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        return 2147483647 * image_id2 + image_id1
    else:
        return 2147483647 * image_id1 + image_id2


def array_to_blob(array):
    return array.tobytes()


def blob_to_array(blob, dtype, shape=(-1,)):
    if blob is None:
        return np.zeros(shape, dtype=dtype)
    return np.frombuffer(blob, dtype=dtype).reshape(*shape)


def build_hybrid_database(features, dataset_path, database_path):
    sorted_features = sorted(features)
    n_features = len(sorted_features)
    
    # Connect to database.
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    
    cursor.execute('SELECT name, image_id FROM images;')
    images = {}
    image_names = {}
    for row in cursor:
        images[row[0]] = row[1]
        image_names[row[1]] = row[0]

    # Randomize features
    random = np.random.RandomState()
    random.seed(1)
    assigned_features = np.array(
        list(range(n_features)) * (len(images) // n_features) +
        list(random.randint(0, n_features, len(images) % n_features))
    )
    random.shuffle(assigned_features)

    image_ids = np.array(list(images.values()))
    for feature_idx, feature in enumerate(sorted_features):
        connection_aux = sqlite3.connect(os.path.join(
            dataset_path, '%s-features.db' % feature
        ))
        cursor_aux = connection_aux.cursor()

        image_indices = np.where(assigned_features == feature_idx)[0]
        for image_id in image_ids[image_indices]:
            image_id = int(image_id)
            assert image_names[image_id] == cursor_aux.execute("SELECT name FROM images WHERE image_id=?", (image_id,)).fetchall()[0][0]
            data, rows, cols = (cursor_aux.execute("SELECT data, rows, cols FROM keypoints WHERE image_id=?", (image_id,))).fetchall()[0]
            cursor.execute(
                'INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?, ?, ?, ?);',
                (image_id, rows, cols, data)
            )
            data, rows, cols = (cursor_aux.execute("SELECT data, rows, cols FROM descriptors WHERE image_id=?", (image_id,))).fetchall()[0]
            cursor.execute(
                'INSERT INTO descriptors(image_id, rows, cols, data) VALUES(?, ?, ?, ?);',
                (image_id, rows, cols, data)
            )
        
        cursor_aux.close()
        connection_aux.close()
    connection.commit()
    
    cursor.close()
    connection.close()

    image_features = {}
    for image_id, feature in zip(image_ids, np.array(sorted_features)[assigned_features]):
        image_features[image_id] = feature

    return image_features


def match_features(colmap_path, database_path, image_path, match_list_path):
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    cursor.execute(
        'SELECT name FROM sqlite_master WHERE type=\'table\' AND name=\'inlier_matches\';'
    )
    try:
        inlier_matches_table_exists = bool(next(cursor)[0])
    except StopIteration:
        inlier_matches_table_exists = False

    cursor.execute('DELETE FROM matches;')
    if inlier_matches_table_exists:
        cursor.execute('DELETE FROM inlier_matches;')
    else:
        cursor.execute('DELETE FROM two_view_geometries;')
    connection.commit()

    images = {}
    cursor.execute('SELECT name, image_id FROM images;')
    for row in cursor:
        images[row[0]] = row[1]

    with open(match_list_path, 'r') as f:
        raw_image_pairs = f.readlines()
    image_pairs = list(map(lambda s: s.strip('\n').split(' '), raw_image_pairs))
    
    image_pair_ids = set()
    for image_name1, image_name2 in tqdm(image_pairs):
        image_id1, image_id2 = images[image_name1], images[image_name2]
        image_pair_id = image_ids_to_pair_id(image_id1, image_id2)
        if image_pair_id in image_pair_ids:
            continue
        image_pair_ids.add(image_pair_id)

        data, rows, cols = (cursor.execute("SELECT data, rows, cols FROM descriptors WHERE image_id=?", (image_id1,))).fetchall()[0]
        try:
            descriptors1 = blob_to_array(data, np.float32, (rows, cols))
        except ValueError:
            descriptors1 = blob_to_array(data, bool, (rows, cols)).astype(np.float32)
        data, rows, cols = (cursor.execute("SELECT data, rows, cols FROM descriptors WHERE image_id=?", (image_id2,))).fetchall()[0]
        try:
            descriptors2 = blob_to_array(data, np.float32, (rows, cols))
        except ValueError:
            descriptors2 = blob_to_array(data, bool, (rows, cols)).astype(np.float32)

        # Match.
        if descriptors1.shape[0] == 0 or descriptors2.shape[0] == 0:
            matches = np.zeros([0, 2], dtype=np.int)
        else:
            matches, _ = mnn_ratio_matcher(descriptors1, descriptors2)

        matches = np.array(matches).astype(np.uint32)
        if matches.shape[0] == 0:
            matches = np.zeros([0, 2])
        assert matches.shape[1] == 2
        if image_id1 > image_id2:
            matches = matches[:, [1, 0]]
        cursor.execute(
            'INSERT INTO matches(pair_id, rows, cols, data) VALUES(?, ?, ?, ?);',
            (image_pair_id, matches.shape[0], matches.shape[1], matches.tostring())
        )
    connection.commit()

    cursor.close()
    connection.close()


def match_features_hybrid(features, image_features, colmap_path, database_path, image_path, match_list_path, encoders, decoders, batch_size, device):
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    cursor.execute(
        'SELECT name FROM sqlite_master WHERE type=\'table\' AND name=\'inlier_matches\';'
    )
    try:
        inlier_matches_table_exists = bool(next(cursor)[0])
    except StopIteration:
        inlier_matches_table_exists = False

    cursor.execute('DELETE FROM matches;')
    if inlier_matches_table_exists:
        cursor.execute('DELETE FROM inlier_matches;')
    else:
        cursor.execute('DELETE FROM two_view_geometries;')
    connection.commit()

    images = {}
    cursor.execute('SELECT name, image_id FROM images;')
    for row in cursor:
        images[row[0]] = row[1]

    with open(match_list_path, 'r') as f:
        raw_image_pairs = f.readlines()
    image_pairs = list(map(lambda s: s.strip('\n').split(' '), raw_image_pairs))
    
    image_pair_ids = set()
    for image_name1, image_name2 in tqdm(image_pairs):
        image_id1, image_id2 = images[image_name1], images[image_name2]
        image_pair_id = image_ids_to_pair_id(image_id1, image_id2)
        if image_pair_id in image_pair_ids:
            continue
        image_pair_ids.add(image_pair_id)

        data, rows, cols = (cursor.execute("SELECT data, rows, cols FROM descriptors WHERE image_id=?", (image_id1,))).fetchall()[0]
        try:
            descriptors1 = blob_to_array(data, np.float32, (rows, cols))
        except ValueError:
            descriptors1 = blob_to_array(data, bool, (rows, cols)).astype(np.float32)
        data, rows, cols = (cursor.execute("SELECT data, rows, cols FROM descriptors WHERE image_id=?", (image_id2,))).fetchall()[0]
        try:
            descriptors2 = blob_to_array(data, np.float32, (rows, cols))
        except ValueError:
            descriptors2 = blob_to_array(data, bool, (rows, cols)).astype(np.float32)

        # Check feature consistency.
        feature1, feature2 = image_features[image_id1], image_features[image_id2]
        
        if feature1 != feature2:
            ford1 = features.index(feature1)
            ford2 = features.index(feature2)
            if ford1 > ford2:
                feature = feature1
                descriptors2 = translate_descriptors(descriptors2, feature2, feature1, encoders, decoders, batch_size, device)
            else:
                feature = feature2
                descriptors1 = translate_descriptors(descriptors1, feature1, feature2, encoders, decoders, batch_size, device)
        else:
            feature = feature1

        # Match.
        if descriptors1.shape[0] == 0 or descriptors2.shape[0] == 0:
            matches = np.zeros([0, 2], dtype=np.int)
        else:
            matches, _ = mnn_ratio_matcher(descriptors1, descriptors2)

        matches = np.array(matches).astype(np.uint32)
        if matches.shape[0] == 0:
            matches = np.zeros([0, 2])
        assert matches.shape[1] == 2
        if image_id1 > image_id2:
            matches = matches[:, [1, 0]]
        cursor.execute(
            'INSERT INTO matches(pair_id, rows, cols, data) VALUES(?, ?, ?, ?);',
            (image_pair_id, matches.shape[0], matches.shape[1], matches.tostring())
        )
    connection.commit()

    cursor.close()
    connection.close()


def match_features_subset(feature, image_features, colmap_path, database_path, image_path, match_list_path):
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    cursor.execute(
        'SELECT name FROM sqlite_master WHERE type=\'table\' AND name=\'inlier_matches\';'
    )
    try:
        inlier_matches_table_exists = bool(next(cursor)[0])
    except StopIteration:
        inlier_matches_table_exists = False

    cursor.execute('DELETE FROM matches;')
    if inlier_matches_table_exists:
        cursor.execute('DELETE FROM inlier_matches;')
    else:
        cursor.execute('DELETE FROM two_view_geometries;')
    connection.commit()

    images = {}
    cursor.execute('SELECT name, image_id FROM images;')
    for row in cursor:
        images[row[0]] = row[1]
    
    image_pairs = set()
    for image_name1, image_id1 in images.items():
        for image_name2, image_id2  in images.items():
            if image_features[image_id1] != feature or image_features[image_id2] != feature:
                continue
            if image_name1 == image_name2:
                continue
            if (image_name2, image_name1) not in image_pairs:
                image_pairs.add((image_name1, image_name2))

    f = open(match_list_path + '.aux', 'w')
    image_pair_ids = set()
    for image_name1, image_name2 in tqdm(image_pairs):
        image_id1, image_id2 = images[image_name1], images[image_name2]
        image_pair_id = image_ids_to_pair_id(image_id1, image_id2)
        if image_pair_id in image_pair_ids:
            continue
        image_pair_ids.add(image_pair_id)

        data, rows, cols = (cursor.execute("SELECT data, rows, cols FROM descriptors WHERE image_id=?", (image_id1,))).fetchall()[0]
        try:
            descriptors1 = blob_to_array(data, np.float32, (rows, cols))
        except ValueError:
            descriptors1 = blob_to_array(data, bool, (rows, cols)).astype(np.float32)
        data, rows, cols = (cursor.execute("SELECT data, rows, cols FROM descriptors WHERE image_id=?", (image_id2,))).fetchall()[0]
        try:
            descriptors2 = blob_to_array(data, np.float32, (rows, cols))
        except ValueError:
            descriptors2 = blob_to_array(data, bool, (rows, cols)).astype(np.float32)

        # Match.
        if descriptors1.shape[0] == 0 or descriptors2.shape[0] == 0:
            matches = np.zeros([0, 2], dtype=np.int)
        else:
            matches, _ = mnn_ratio_matcher(descriptors1, descriptors2)

        matches = np.array(matches).astype(np.uint32)
        if matches.shape[0] == 0:
            matches = np.zeros([0, 2])
        assert matches.shape[1] == 2
        if image_id1 > image_id2:
            matches = matches[:, [1, 0]]
        cursor.execute(
            'INSERT INTO matches(pair_id, rows, cols, data) VALUES(?, ?, ?, ?);',
            (image_pair_id, matches.shape[0], matches.shape[1], matches.tostring())
        )
        f.write('%s %s\n' % (image_name1, image_name2))
    connection.commit()

    cursor.close()
    connection.close()


def geometric_verification(colmap_path, database_path, match_list_path):
    subprocess.call([
        os.path.join(colmap_path, 'colmap'), 'matches_importer',
        '--database_path', database_path,
        '--match_list_path', match_list_path,
        '--match_type', 'pairs',
        '--SiftMatching.num_threads', str(8),
        '--SiftMatching.use_gpu', '0',
        '--SiftMatching.min_inlier_ratio', '0.1'
    ])

    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    cursor.execute('SELECT count(*) FROM images;')
    num_images = next(cursor)[0]

    cursor.execute('SELECT count(*) FROM two_view_geometries WHERE rows > 0;')
    num_inlier_pairs = next(cursor)[0]

    cursor.execute('SELECT sum(rows) FROM two_view_geometries WHERE rows > 0;')
    num_inlier_matches = next(cursor)[0]

    cursor.close()
    connection.close()

    return dict(
        num_images=num_images,
        num_inlier_pairs=num_inlier_pairs,
        num_inlier_matches=num_inlier_matches
    )

def reconstruct(colmap_path, database_path, image_path, sparse_path, refine_intrinsics=False):
    # Run the sparse reconstruction.
    if not os.path.exists(sparse_path):
        os.mkdir(sparse_path)
    if not refine_intrinsics:
        extra_mapper_params = [
            '--Mapper.ba_refine_focal_length', str(0),
            '--Mapper.ba_refine_principal_point', str(0),
            '--Mapper.ba_refine_extra_params', str(0)
        ]
    else:
        extra_mapper_params = [
            '--Mapper.ba_refine_focal_length', str(1),
            '--Mapper.ba_refine_principal_point', str(0),
            '--Mapper.ba_refine_extra_params', str(1)
        ]
    subprocess.call([
        os.path.join(colmap_path, 'colmap'), 'mapper',
        '--database_path', database_path,
        '--image_path', image_path,
        '--output_path', sparse_path,
        '--Mapper.abs_pose_min_inlier_ratio', str(0.05),
        '--Mapper.num_threads', str(16),
    ] + extra_mapper_params)

    # Find the largest reconstructed sparse model.
    models = os.listdir(sparse_path)
    if len(models) == 0:
        print('Warning: Could not reconstruct any model')
        return

    largest_model = None
    largest_model_num_images = 0
    for model in models:
        subprocess.call([
            os.path.join(colmap_path, 'colmap'), 'model_converter',
            '--input_path', os.path.join(sparse_path, model),
            '--output_path', os.path.join(sparse_path, model),
            '--output_type', 'TXT'
        ])
        with open(os.path.join(sparse_path, model, 'cameras.txt'), 'r') as fid:
            for line in fid:
                if line.startswith('# Number of cameras'):
                    num_images = int(line.split()[-1])
                    if num_images > largest_model_num_images:
                        largest_model = model
                        largest_model_num_images = num_images
                    break

    assert largest_model_num_images > 0

    largest_model_path = os.path.join(sparse_path, largest_model)

    # Convert largest model to ply.
    subprocess.call([
        os.path.join(colmap_path, 'colmap'), 'model_converter',
        '--input_path', largest_model_path,
        '--output_path', os.path.join(sparse_path, 'pointcloud.ply'),
        '--output_type', 'PLY'
    ])

    # Recover model statistics.
    stats = subprocess.check_output([
        os.path.join(colmap_path, 'colmap'), 'model_analyzer',
        '--path', largest_model_path
    ])

    stats = stats.decode().split('\n')
    for stat in stats:
        if stat.startswith('Registered images'):
            num_reg_images = int(stat.split()[-1])
        elif stat.startswith('Points'):
            num_sparse_points = int(stat.split()[-1])
        elif stat.startswith('Observations'):
            num_observations = int(stat.split()[-1])
        elif stat.startswith('Mean track length'):
            mean_track_length = float(stat.split()[-1])
        elif stat.startswith('Mean observations per image'):
            num_observations_per_image = float(stat.split()[-1])
        elif stat.startswith('Mean reprojection error'):
            mean_reproj_error = float(stat.split()[-1][:-2])

    return largest_model_path, dict(
        num_reg_images=num_reg_images,
        num_sparse_points=num_sparse_points,
        num_observations=num_observations,
        mean_track_length=mean_track_length,
        num_observations_per_image=num_observations_per_image,
        mean_reproj_error=mean_reproj_error
    )


def compute_extra_stats(image_features, largest_model_path):
    with open(os.path.join(largest_model_path, 'images.txt'), 'r') as f:
        raw_images = f.readlines()
        raw_images = raw_images[4 :][:: 2]

    counter = {}
    for raw_image in raw_images:
        image = raw_image.strip('\n').split(' ')
        feature = image_features[int(image[0])]
        if feature not in counter:
            counter[feature] = 0
        counter[feature] += 1
    
    return counter
