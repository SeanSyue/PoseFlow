from collections import defaultdict
from pathlib import Path, PurePath
import json
from tqdm import tqdm
import numpy as np
import pickle

import utils as U

# AlphaPose keypoint detection root
KP_DET_ROOT = '/home/ee303/WORKSPACE_DATA/AlphaPose/results/posetrack_2018'
INFO_PKL = '/home/ee303/WORKSPACE_DATA/PoseFlow/test_simple/info-simple.pkl'


def split_im_path(im_path_):
    p = PurePath(im_path_)
    return str(p.parent), p.name


if __name__ == '__main__':
    info = U.NestedDict()

    # all_jsons = [n for n in Path(KP_DET_ROOT).rglob('*.json')]
    all_jsons = [f'{KP_DET_ROOT}/{n}/alphapose-results.json' for n in ('000342_mpii_test', '000522_mpii_test')]
    # all_jsons = ['/home/ee303/WORKSPACE_DATA/AlphaPose/results/posetrack_2018/000342_mpii_test/alphapose-results.json']
    for j in tqdm(all_jsons, total=len(all_jsons)):

        with Path(j).open() as f:
            j_obj = json.loads(f.read())

        id_reg = defaultdict(lambda: 1)

        for n in j_obj:
            im_path = '/'.join(PurePath(n['image_id']).parts[-3:])
            video, frame = split_im_path(im_path)

            info[video][frame]['num_boxes'] = id_reg[im_path]
            info[video][frame][id_reg[im_path]] = {
                "box_score": n['score'],
                "box_pos": U.get_box(n['keypoints'], n['image_id']),
                "box_pose_pos": np.array(n['keypoints']).reshape(-1, 3)[:, 0:2],
                "box_pose_score": np.array(n['keypoints']).reshape(-1, 3)[:, -1]
            }

            id_reg[im_path] += 1

    # print(info.keys())

    with Path(INFO_PKL).open('wb') as f:
        pickle.dump(info, f)
