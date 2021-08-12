python feature-utils/extract_sift.py --colmap_path $COLMAP_PATH --dataset_path data/train/
for feature in 'brief' 'sift-kornia' 'hardnet' 'sosnet'; do
    echo $feature
    python feature-utils/extract_descriptors.py --dataset_path data/train/ --feature $feature
    python feature-utils/convert_database_to_numpy.py --dataset_path data/train/ --feature $feature
done
mkdir data/train/colmap
mv data/train/*.db data/train/colmap
mv data/train/*.npy data/train/colmap
