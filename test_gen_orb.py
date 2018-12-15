from pathlib import Path
from tqdm import tqdm

from matching import orb_matching


VAL_ROOT = '/home/ee303/DATASETS/posetrack/posetrack_2018/images/val'
OUTPUT_ROOT = '/home/ee303/WORKSPACE_DATA/PoseFlow/PoseTrack_2018_orb'
sample = ['000342_mpii_test', '000522_mpii_test']

if __name__ == '__main__':
    for n in sample:

        out_path = Path(OUTPUT_ROOT) / n
        out_path.mkdir(exist_ok=True)

        all_img = [n for n in sorted((Path(VAL_ROOT) / n).glob('*.jpg'))]
        for cur_img in tqdm(all_img[:-1]):
            im1 = str(cur_img)
            cur_idx = cur_img.stem
            next_idx = f"{int(cur_idx) + 1:06}"
            im2 = str(cur_img.with_name(f"{next_idx}.jpg"))

            orb_matching(im1, im2, out_path, cur_idx, next_idx)

