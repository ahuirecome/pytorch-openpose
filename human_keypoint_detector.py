import torch
import torch.nn as nn
import cv2
import os
import numpy as np
from src import util
from src.body import Body
from src.hand import Hand


class HumanKeypointDetector(nn.Module):
    def __init__(self):
        super().__init__()
        model_dir = os.path.dirname(__file__) + '/model/'
        self.body_estimation = Body(model_dir + 'body_pose_model.pth')
        self.hand_estimation = Hand(model_dir + 'hand_pose_model.pth')

    def predict(self, cv2_img):
        candidate, subset = self.body_estimation(cv2_img)
        if len(subset) == 0:
            return None
        subset = subset.astype(np.int32)
        result = np.zeros(shape=(len(subset), 18+21+21, 3), dtype=np.int32) - 1
        for n, person in enumerate(subset):
            for k in range(18):
                index = person[k]
                if index == -1:
                    continue
                else:
                    x, y = candidate[index][0:2]
                    result[n, k] = [x, y, 1]

            person = person[np.newaxis, :]
            hands = util.handDetect(candidate, person, cv2_img)

            for x, y, w, is_left in hands:
                peaks = self.hand_estimation(cv2_img[y:y + w, x:x + w, :])
                peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
                peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
                ones = np.ones(shape=(len(peaks), 1), dtype=np.int32)
                peaks = np.hstack((peaks, ones))
                for k in range(len(peaks)):
                    if peaks[k, 0] == 0 and peaks[k, 1] == 0:
                        peaks[k, 2] = -1

                if is_left:
                    result[n, 18:39] = peaks
                else:
                    result[n, 39:60] = peaks

        return result

    @classmethod
    def match_face(cls, people_keypoints, face_keypoints68):
        face_width = max(face_keypoints68[:, 0]) - min(face_keypoints68[:, 0])
        face_height = max(face_keypoints68[:, 1]) - min(face_keypoints68[:, 1])
        face_scale = max(face_width, face_height)
        face_eye_center = np.mean(face_keypoints68[36:48], axis=0)

        dists = []
        for person_keypoints in people_keypoints:
            if person_keypoints[15, 2] == -1 or people_keypoints[16, 2] == -1:
                dists.append(1000000)
                continue
            else:
                center = (person_keypoints[15, 0:2] + person_keypoints[16, 0:2]) * 0.5
                dist = np.linalg.norm(face_eye_center - center)
                dists.append(dist)
        k = np.argmin(dists)
        if dists[k] > face_scale * 0.5:
            return None
        else:
            return people_keypoints[k]

    def get_central_person(self, people_keypoints, img_shape):
        if len(people_keypoints) == 0:
            return None
        [img_h, img_w] = img_shape[0:2]
        img_center = np.array([img_w/2, img_h/2])
        dists = []
        for person_keypoints in people_keypoints:
            head_points = []
            head_points_indices = [0, 14, 15, 16, 17]
            for index in head_points_indices:
                if person_keypoints[index, 2] != -1:
                    head_points.append(person_keypoints[index, 0:2])
            if len(head_points) == 0:
                dists.append(100000)
                continue
            else:
                head_center = np.mean(head_points, axis=0)
                dist = np.linalg.norm(head_center - img_center)
                dists.append(dist)

        k = np.argmin(dists)
        if dists[k] > max(img_w, img_h):
            return None
        else:
            return people_keypoints[k]


    def keypoint_names(cls):
        # {0, "Nose"},{1, "Neck"}, {2, "RShoulder"},{3, "RElbow"},{4, "RWrist"},{5, "LShoulder"},
        # {6, "LElbow"},{7, "LWrist"},{8, "MidHip"},{9, "RHip"},{10, "RKnee"},{11, "RAnkle"},
        # {12, "LHip"},{13, "LKnee"},{14, "LAnkle"},{15, "REye"},{16, "LEye"},{17, "REar"},
        # {18, "LEar"}
        pass

    @classmethod
    def draw_keypoints(cls, keypoints, cv2_img, limb_width=10, finger_width=3):
        # limbs = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
        #            [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
        #            [1, 16], [16, 18], [3, 17], [6, 18]]
        limbs = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
                   [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
                   [1, 16], [16, 18]]
        limbs = np.array(limbs) - 1

        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                  [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
                  [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

        fingers = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
                 [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

        body_points = keypoints[0:18]
        left_hand_points = keypoints[18:39]
        right_hand_points = keypoints[39:60]

        img_drawn = cv2_img.copy()
        for k, limb in enumerate(limbs):
            start = body_points[limb[0]]
            end = body_points[limb[1]]
            if start[2] == -1 or end[2] == -1:
                continue
            cv2.line(img_drawn, tuple(start[0:2]), tuple(end[0:2]), colors[k], thickness=limb_width)

        left_hand_color = (125, 50, 150)
        for k, finger in enumerate(fingers):
            start = left_hand_points[finger[0]]
            end = left_hand_points[finger[1]]
            if start[2] == -1 or end[2] == -1:
                continue
            cv2.line(img_drawn, tuple(start[0:2]), tuple(end[0:2]), left_hand_color, thickness=finger_width)

        right_hand_color = (75, 200, 50)
        for k, finger in enumerate(fingers):
            start = right_hand_points[finger[0]]
            end = right_hand_points[finger[1]]
            if start[2] == -1 or end[2] == -1:
                continue
            cv2.line(img_drawn, tuple(start[0:2]), tuple(end[0:2]), right_hand_color, thickness=finger_width)

        return img_drawn


if __name__  == '__main__':
    cv2_img = cv2.imread('./images/demo1.jpg')
    detector = HumanKeypointDetector()

    people_keypoints = detector.predict(cv2_img)
    # for keypoints in people_keypoints:
    #     cv2_img = detector.draw_keypoints(keypoints, cv2_img)

    central_person = detector.get_central_person(people_keypoints, cv2_img.shape)
    cv2_img = detector.draw_keypoints(central_person, cv2_img)

    cv2.imshow('img', cv2_img)
    cv2.waitKey()


