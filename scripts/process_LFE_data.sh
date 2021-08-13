LFE_PATH=data/eval/LFE-release
for dataset in 'Gendarmenmarkt' 'Madrid_Metropolis' 'Tower_of_London'; do
    echo $dataset
    # Dataset already provides keypoints consistent with the reference database.
    # python feature-utils/extract_sift.py --colmap_path $COLMAP_PATH --dataset_path $LFE_PATH/$dataset --image_path $LFE_PATH/$dataset/images
    for feature in 'brief' 'sift-kornia' 'hardnet' 'sosnet'; do
        echo $feature
        python feature-utils/extract_descriptors.py --dataset_path $LFE_PATH/$dataset --image_path $LFE_PATH/$dataset/images --feature $feature
    done
done
