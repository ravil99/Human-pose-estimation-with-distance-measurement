import os
import tensorflow as tf


def download_model(model_type):
    server_prefix = 'https://omnomnom.vision.rwth-aachen.de/data/metrabs'
    model_zippath = tf.keras.utils.get_file(
        origin=f'{server_prefix}/{model_type}.zip',
        extract=True, cache_subdir='models')
    model_path = os.path.join(os.path.dirname(model_zippath), model_type)
    return model_path


class PoseEstimator:
    def __init__(self, use_poseviz=True):
        model = download_model('metrabs_rn18_y4')
        self.model = tf.saved_model.load(model)
        self.skeleton = 'smpl+head_30'
        joint_names = self.model.per_skeleton_joint_names[self.skeleton].numpy().astype(
            str)
        joint_edges = self.model.per_skeleton_joint_edges[self.skeleton].numpy(
        )
        self.use_poseviz = use_poseviz
        if use_poseviz:
            import poseviz
            self.viz = poseviz.PoseViz(joint_names, joint_edges)

    def predict(self, rgb_image):
        pred = self.model.detect_poses(
            rgb_image, skeleton=self.skeleton, default_fov_degrees=55, detector_threshold=0.5)
        
        if self.use_poseviz:
            import poseviz     
            camera = poseviz.Camera.from_fov(55, rgb_image.shape[:2])
            self.viz.update(rgb_image, pred['boxes'], pred['poses3d'], camera)

        return pred
