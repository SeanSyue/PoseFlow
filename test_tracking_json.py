from pathlib import Path, PurePath
import json
import pickle
from tqdm import tqdm

TRACK_PKL = '/home/ee303/WORKSPACE_DATA/PoseFlow/test_simple/track-simple.pkl'
ANNO_ROOT = '/home/ee303/DATASETS/posetrack/posetrack_2018/posetrack_data/annotations'
RESULT_ROOT = '/home/ee303/WORKSPACE_DATA/PoseFlow/test_simple/track_results'


DROP = 0.0

if __name__ == '__main__':

    with Path(TRACK_PKL).open('rb') as f:
        track = pickle.load(f)

    Path(RESULT_ROOT).mkdir(exist_ok=True)

    # export tracking result into json files
    for video_name in tqdm(track.keys()):

        f_annot = Path(ANNO_ROOT) / f'{video_name}.json'
        with f_annot.open() as f:
            annot = json.load(f)
        # print(annot.keys())

        tracked_frames = sorted([n for n in track[video_name].keys() if n != 'num_persons'])
        # print(tracked_frames)

        orig_imgs = [str(PurePath(img['file_name']).name) for img in annot['images']]
        # print(orig_imgs)

        final = {'annolist': []}

        for fid, frame_name in enumerate(tracked_frames):
            final['annolist'].append({"image": [{"name": str(PurePath(f'images/val/{video_name}/{frame_name}'))}],
                                      "annorect": []})

            if frame_name not in orig_imgs:
                continue

            else:
                for pid in range(1, track[video_name][frame_name]['num_boxes'] + 1):
                    pid_info = track[video_name][frame_name][pid]
                    box_pos = pid_info['box_pos']
                    box_score = pid_info['box_score']
                    pose_pos = pid_info['box_pose_pos']
                    pose_score = pid_info['box_pose_score']
                    # pose_pos = U.add_nose(pid_info['box_pose_pos'])
                    # pose_score = U.add_nose(pid_info['box_pose_score'])
                    new_pid = pid_info['new_pid']

                    point_struct = []
                    for idx, pose in enumerate(pose_pos):
                        print(pose_score[idx])
                        if pose_score[idx] > DROP:
                            print("BANG~!")
                            point_struct.append({"id": [idx], "x": [pose[0]], "y": [pose[1]], "score": [pose_score[idx]]})

                    final['annolist'][fid]['annorect'].append({"x1": [box_pos[0]],
                                                               "x2": [box_pos[1]],
                                                               "y1": [box_pos[2]],
                                                               "y2": [box_pos[3]],
                                                               "score": [box_score],
                                                               "track_id": [new_pid - 1],
                                                               "annopoints": [{"point": point_struct}]})

        with Path(f'{RESULT_ROOT}/{PurePath(video_name).stem}.json').open('w') as f:
            json.dump(final, f)
