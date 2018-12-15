from copy import deepcopy
from pathlib import Path, PurePath
import pickle
from tqdm import tqdm
import numpy as np
import utils as U


DATASET_ROOT = '/home/ee303/DATASETS/posetrack/posetrack_2018'
COR_ROOT = '/home/ee303/WORKSPACE_DATA/PoseFlow/PoseTrack_2018_orb'
INFO_PKL = '/home/ee303/WORKSPACE_DATA/PoseFlow/test_simple/info-simple.pkl'
TRACK_PKL = '/home/ee303/WORKSPACE_DATA/PoseFlow/test_simple/track-simple.pkl'

LINK_LEN = 100
WEIGHTS = [1, 2, 1, 2, 0, 0]
WEIGHTS_FFF = [0, 1, 0, 1, 0, 0]
NUM = 7
MAG = 30
MATCH_THRES = 0.2
# RMPE_PARTS_ID = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 8, 9]
RMPE_PARTS_ID = [16, 14, 12, 11, 13, 15, 10, 8, 6, 5, 7, 9, 0, 0, 1]

if __name__ == '__main__':

    with Path(INFO_PKL).open('rb') as f:
        track = pickle.load(f)

    for video_name in tqdm(track.keys()):

        frame_list = sorted(list(track[video_name].keys()))
        for idx, frame_name in enumerate(frame_list[:-1]):

            frame_id = PurePath(frame_name).stem
            next_frame_id = f"{int(frame_id) + 1:06}"
            next_frame_name = str(PurePath(frame_name).with_name(f"{next_frame_id}.jpg"))

            # if there is no people in this frame, then copy the info from former frame
            if track[video_name][next_frame_name]['num_boxes'] == 0:
                track[video_name][next_frame_name] = deepcopy(track[video_name][frame_name])
                continue

            # init tracking info of the first frame in one video
            if idx == 0:
                for pid in range(1, track[video_name][frame_name]['num_boxes'] + 1):
                    track[video_name][frame_name][pid]['new_pid'] = pid
                    track[video_name][frame_name][pid]['match_score'] = 0

            max_pid_id = track[video_name][frame_name]['num_boxes']

            all_cors = np.loadtxt(str(PurePath(COR_ROOT) / PurePath(video_name).name / f'{frame_id}_{next_frame_id}_orb.txt'))

            cur_all_pids, cur_all_pids_fff = U.stack_all_pids(track[video_name], frame_list[:-1], idx, max_pid_id, LINK_LEN)

            match_indexes, match_scores = U.best_matching_hungarian(
                all_cors, cur_all_pids, cur_all_pids_fff, track[video_name][next_frame_name], WEIGHTS, WEIGHTS_FFF, NUM,
                MAG)
            
            for pid1, pid2 in match_indexes:
                if match_scores[pid1][pid2] > MATCH_THRES:
                    track[video_name][next_frame_name][pid2 + 1]['new_pid'] = cur_all_pids[pid1]['new_pid']
                    max_pid_id = max(max_pid_id, track[video_name][next_frame_name][pid2 + 1]['new_pid'])
                    track[video_name][next_frame_name][pid2 + 1]['match_score'] = match_scores[pid1][pid2]

            # add the untracked new person
            for next_pid in range(1, track[video_name][next_frame_name]['num_boxes'] + 1):
                if 'new_pid' not in track[video_name][next_frame_name][next_pid]:
                    max_pid_id += 1
                    track[video_name][next_frame_name][next_pid]['new_pid'] = max_pid_id
                    track[video_name][next_frame_name][next_pid]['match_score'] = 0

            # # deal with unconsecutive frames caused by this fucking terrible dataset
            # gap = int(next_frame_id) - int(frame_id)
            # if gap > 1:
            #     for i in range(gap):
            #         if i > 0:
            #             # new_frame_name = "%08d.jpg" % (int(frame_id) + i)
            #             new_frame_name = f'{(int(frame_id) + i):08}.jpg'
            #             track[video_name][new_frame_name] = deepcopy(track[video_name][frame_name])

    for video_name in tqdm(track.keys()):
        num_persons = 0
        frame_list = sorted(list(track[video_name].keys()))
        for fid, frame_name in enumerate(frame_list):
            for pid in range(1, track[video_name][frame_name]['num_boxes'] + 1):
                new_score = deepcopy(track[video_name][frame_name][pid]['box_pose_score'])
                new_pose = deepcopy(track[video_name][frame_name][pid]['box_pose_pos'])
                track[video_name][frame_name][pid]['box_pose_score'] = new_score[RMPE_PARTS_ID]
                track[video_name][frame_name][pid]['box_pose_pos'] = new_pose[RMPE_PARTS_ID, :]
                num_persons = track[video_name][frame_name][pid]['new_pid']
        track[video_name]['num_persons'] = num_persons

    # print(track)

    with Path(TRACK_PKL).open('wb') as f:
        pickle.dump(track, f)
