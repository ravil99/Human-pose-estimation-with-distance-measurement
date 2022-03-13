import os
import tensorflow as tf
import cv2
import numpy as np

# For now, it's written manually
intrinsic_matrix = tf.constant([[586.69337515,   0,         307.83631346],
                                [  0,         624.43501009,  233.78929737],
                                 [  0,           0,           1        ]])


distortion_coeffs = tf.constant([7.86832682e-03,  5.06307790e+00,  1.26011321e-01,  5.76976883e-02,
  -1.88811379e+01])

def download_model(model_type):
    server_prefix = 'https://omnomnom.vision.rwth-aachen.de/data/metrabs'
    model_zippath = tf.keras.utils.get_file(
        origin=f'{server_prefix}/{model_type}.zip',
        extract=True, cache_subdir='models')
    model_path = os.path.join(os.path.dirname(model_zippath), model_type)
    return model_path

class HumanPoint:
    def __init__(self, point_2d, point_3d):
        self.point_2d = point_2d
        self.point_3d = point_3d
    
    def draw(self, frame, id):
        # Draw the human chest point
        frame = cv2.circle(frame, self.point_2d, radius=8,
                            color=(0, 0, 255), thickness=-1)
        # Put human "ID" near the chest point
        cv2.putText(frame, str(id), (self.point_2d[0] - 5, self.point_2d[1] - 5),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)


class PoseEstimator:
    def __init__(self, skeleton='smpl+head_30', use_poseviz=True):
        model = download_model('metrabs_rn18_y4')
        self.skeleton = skeleton
        self.model = tf.saved_model.load(model)
        joint_names = self.model.per_skeleton_joint_names[self.skeleton].numpy().astype(
            str)
        joint_edges = self.model.per_skeleton_joint_edges[self.skeleton].numpy(
        )
        self.use_poseviz = use_poseviz
        if use_poseviz:
            import poseviz
            self.viz = poseviz.PoseViz(joint_names, joint_edges)

        self.colors = []
        for i in range(10):
            color = np.random.choice(range(256), size=3)
            color = (int(color[0]), int(color[1]), int(color[2]))
            self.colors.append(color)

        self.rsho_ind = -1
        self.lsho_ind = -1
        joint_names = self.model.per_skeleton_joint_names[self.skeleton].numpy()
        joint_names = np.array([b.decode('utf8') for b in joint_names])
        for i, joint in enumerate(joint_names):
            if 'rsho' in joint:
                self.rsho_ind = i
            if 'lsho' in joint:
                self.lsho_ind = i

    def predict(self, rgb_image):
        pred = self.model.detect_poses(
            rgb_image, skeleton=self.skeleton, default_fov_degrees=55, detector_threshold=0.5)

        if self.use_poseviz:
            import poseviz
            camera = poseviz.Camera.from_fov(55, rgb_image.shape[:2])
            self.viz.update(rgb_image, pred['boxes'], pred['poses3d'], camera)

        return pred

    def draw_2d_skeleton(self, pred, image):
        for human in range(len(pred['poses2d'])):
            for i, j in self.model.per_skeleton_joint_edges[self.skeleton]:
                p1 = tuple(pred['poses2d'][human][i].numpy().astype(int))
                p2 = tuple(pred['poses2d'][human][j].numpy().astype(int))
                image = cv2.line(image, p1, p2, self.colors[human], 4)
        return image

    def get_human_points(self, pred):
        chest_points_2d = []
        for human in range(len(pred['poses2d'])):
            lsho = pred['poses2d'][human][self.lsho_ind].numpy()
            rsho = pred['poses2d'][human][self.rsho_ind].numpy()
            chest = (lsho + rsho)/2
            chest = (round(chest[0]), round(chest[1]))
            chest_points_2d.append(chest)

        chest_points_3d = []
        for human in range(len(pred['poses3d'])):
            lsho = pred['poses3d'][human][self.lsho_ind].numpy()
            rsho = pred['poses3d'][human][self.rsho_ind].numpy()
            chest = (lsho + rsho)/2
            chest = (round(chest[0]), round(chest[1]), round(chest[2]))
            chest_points_3d.append(chest)

        human_points = []
        for chest_point_2d, chest_point_3d in zip(chest_points_2d, chest_points_3d):
            human_points.append(HumanPoint(chest_point_2d, chest_point_3d))

        return human_points