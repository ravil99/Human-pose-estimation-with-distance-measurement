import torch
import numpy as np


class DepthEstimator:
    """Depth estimator. Based on MiDaS work.
    https://github.com/isl-org/MiDaS
    """

    def __init__(self, device="cpu"):
        """Depth estimator initialization

        Args:
            device (str, optional): Еhe device on which the calculations will take place. 
            Can be "cpu" or "gpu". Defaults to "cpu".
        """
        # model_type = "DPT_Hybrid" or "DPT_Large" or "MiDaS_small"
        # For more information see https://pytorch.org/hub/intelisl_midas_v2/
        model_type = "MiDaS_small"
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.device = torch.device(device)
        self.midas.to(self.device)
        self.midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def predict(self, rgb_image):
        """Predict image depth at each pixel

        Args:
            rgb_image (array): RGB image

        Returns:
            tuple: (disparity map, scaled disparity map)
        """
        input_batch = self.transform(rgb_image).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb_image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()

        # Make scale
        bits = 2
        depth_min = output.min()
        depth_max = output.max()

        max_val = (2**(8*bits))-1

        if depth_max - depth_min > np.finfo("float").eps:
            scaled_output = max_val * \
                (output - depth_min) / (depth_max - depth_min)
        else:
            scaled_output = np.zeros(output.shape, dtype=output.type)

        return output, scaled_output.astype("uint16")

    def get_depth(disparity_map, point):
        """Get depth value at a single point

        Args:
            disparity_map (array): predicted disparity map
            point (tuple): (x coord in pixels, y coord in pixels)

        Returns:
            float: depth value
        """
        point_x = min(disparity_map.shape[1] - 1, max(0, point[0]))
        point_y = min(disparity_map.shape[0] - 1, max(0, point[1]))
        return disparity_map[point_y, point_x]

    def get_distance(disparity_map, distance_to_point1, point1, point2):
        """Get distance to second point relative to the first point

        Args:
            disparity_map (array): predicted disparity map
            distance_to_point1 (int): real distance to the point1 in meters.
            point1 (tuple): (x coord in pixels, y coord in pixels)
            point2 (tuple): (x coord in pixels, y coord in pixels)

        Returns:
            float: distance to second point relative to the first point
        """
        point1_depth = DepthEstimator.get_depth(disparity_map, point1)
        point2_depth = DepthEstimator.get_depth(disparity_map, point2)
        return distance_to_point1 * point1_depth / point2_depth
