import argparse

import numpy as np

import os

import subprocess


def qvec_to_rotmat(qvec):
    w, x, y, z = qvec
    R = np.array([
        [
            1 - 2 * y * y - 2 * z * z,
            2 * x * y - 2 * z * w,
            2 * x * z + 2 * y * w
        ],
        [
            2 * x * y + 2 * z * w,
            1 - 2 * x * x - 2 * z * z,
            2 * y * z - 2 * x * w
        ],
        [
            2 * x * z - 2 * y * w,
            2 * y * z + 2 * x * w,
            1 - 2 * x * x - 2 * y * y
        ]
    ])
    return R


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_path', required=True,
        help='path to the model'
    )
    parser.add_argument(
        '--reference_model_path', required=True,
        help='path to the reference model'
    )
    parser.add_argument(
        '--colmap_path', required=True,
        help='path to the COLMAP executable folder'
    )
    
    args = parser.parse_args()

    # Create the output path.
    aligned_model_path = os.path.join(args.model_path, 'aligned')

    if not os.path.exists(aligned_model_path):
        os.mkdir(aligned_model_path)

    # Read and cache the reference model.
    with open(os.path.join(args.reference_model_path, 'images.txt'), 'r') as f:
        lines = f.readlines()
    reference_poses = {}
    for line in lines[4 :: 2]:
        line = line.strip('\n').split(' ')
        image_name = line[-1]
        qvec = np.array(list(map(float, line[1 : 5])))
        t = np.array(list(map(float, line[5 : 8])))
        R = qvec_to_rotmat(qvec)
        reference_poses[image_name] = [R, t]

    # Run the model aligner.
    subprocess.call([
        os.path.join(args.colmap_path, 'colmap'), 'model_aligner',
        '--input_path', args.model_path,
        '--output_path', aligned_model_path,
        '--ref_images_path', os.path.join(args.reference_model_path, 'geo.txt'),
        '--robust_alignment_max_error', str(0.25)
    ])

    subprocess.call([
        os.path.join(args.colmap_path, 'colmap'), 'model_converter',
        '--input_path', aligned_model_path,
        '--output_path', aligned_model_path,
        '--output_type', 'TXT'
    ])

    # Parse the aligned model.
    with open(os.path.join(aligned_model_path, 'images.txt'), 'r') as f:
        lines = f.readlines()
    ori_errors = []
    center_errors = []
    image_ids = []
    for line in lines[4 :: 2]:
        line = line.strip('\n').split(' ')
        image_id = int(line[0])
        image_name = line[-1]
        qvec = np.array(list(map(float, line[1 : 5])))
        t = np.array(list(map(float, line[5 : 8])))
        R = qvec_to_rotmat(qvec)
        # Compute the error.
        annotated_R, annotated_t = reference_poses[image_name]

        rotation_difference = R @ annotated_R.transpose()
        ori_error = np.rad2deg(np.arccos(np.clip((np.trace(rotation_difference) - 1) / 2, -1, 1)))

        annotated_C = (-1) * annotated_R.transpose() @ annotated_t
        C = (-1) * R.transpose() @ t
        center_error = np.linalg.norm(C - annotated_C)
        if center_error > 0.10:
            image_ids.append((image_id, image_name))

        ori_errors.append(ori_error)
        center_errors.append(center_error)
    ori_errors = np.array(ori_errors)
    center_errors = np.array(center_errors)

print('Registered %d out of %d images.' % (len(ori_errors), len(reference_poses)))
print('0.25m, 2 deg:', np.sum(np.logical_and(ori_errors <= 2, center_errors <= 0.25)) / len(reference_poses))
print('0.50m, 5 deg:', np.sum(np.logical_and(ori_errors <= 5, center_errors <= 0.50)) / len(reference_poses))
print('inf         :', len(ori_errors) / len(reference_poses))
