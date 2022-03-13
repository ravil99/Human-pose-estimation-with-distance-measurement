import torch
import numpy as np


class DepthEstimator:
    def __init__(self):
        # model_type = "DPT_Hybrid" or "DPT_Large" or "MiDaS_small"
        model_type = "MiDaS_small"
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.device = torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def predict(self, rgb_image):
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
        point_x = min(disparity_map.shape[1] - 1, max(0, point[0]))
        point_y = min(disparity_map.shape[0] - 1, max(0, point[1]))
        return disparity_map[point_y, point_x]

    def get_distance(disparity_map, distance_to_point1, point1, point2):
        max_depth = np.max(disparity_map)
        point1_depth = DepthEstimator.get_depth(disparity_map, point1)
        point2_depth = DepthEstimator.get_depth(disparity_map, point2)
        print("arUco depth", point1_depth)
        print("human depth", point2_depth)
        print("max depth", max_depth)
        return distance_to_point1 / (max_depth - point1_depth) * (max_depth - point2_depth)
