from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import mediapy as media
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import colormaps
from nerfstudio.utils.eval_utils import eval_setup

import argparse

from nerfstudio.viewer.server.utils import three_js_perspective_camera_focal_length

class RenderImages():
    def __init__(self, args):
        self.output_format: Literal["images", "video"] = "images"
        self.load_config = Path(args.config)
        self.output_path = Path(args.output)
        self.jpeg_quality: int = 100
        self.image_format: Literal["jpeg", "png"] = "jpeg"
        self.downscale_factor: float = 1.0
        self.eval_num_rays_per_chunk: Optional[int] = None
        self.rendered_output_names: List[str] = ["rgb"]
        self.depth_near_plane: Optional[float] = None
        self.depth_far_plane: Optional[float] = None
        self.colormap_options: colormaps.ColormapOptions = colormaps.ColormapOptions()
        _, self.pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
        )

        self.image_height = 540
        self.image_width = 960
        self.fov = 50
        self.camera_type = CameraType.PERSPECTIVE
        self.img_cnt = 0


    def render_img(
        self,
        pipeline: Pipeline,
        cameras: Cameras,
        output_foldername: Path,
        rendered_output_names: List[str],
        rendered_resolution_scaling_factor: float = 1.0,
        seconds: float = 5.0,
        output_format: Literal["images", "video"] = "images",
        image_format: Literal["jpeg", "png"] = "jpeg",
        jpeg_quality: int = 100,
        depth_near_plane: Optional[float] = None,
        depth_far_plane: Optional[float] = None,
        colormap_options: colormaps.ColormapOptions = colormaps.ColormapOptions(),
    ) -> None:

        cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
        cameras = cameras.to(pipeline.device)
        print("Check_in_render_img")
        camera_idx = 0
        aabb_box = None
        camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx, aabb_box=aabb_box)
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        
        output_image = outputs["rgb"]
        output_image = (
                        colormaps.apply_colormap(
                            image=output_image,
                            colormap_options=colormap_options,
                        )
                        .cpu()
                        .numpy()
                    )

        output_filename = output_foldername / (str(self.img_cnt + 1) + ".jpg")
        media.write_image(
            output_filename, output_image, fmt="jpeg", quality=jpeg_quality
        )

        self.img_cnt += 1
            
        print("End_in_render_img")

    def create_camera_object(self, camera_to_world_matrix, focal_length):
        c2ws = []
        fxs = []
        fys = []

        c2w = torch.tensor(camera_to_world_matrix).view(4, 4)[:3]
        c2ws.append(c2w)
        fxs.append(focal_length)
        fys.append(focal_length)

        camera_to_worlds = torch.stack(c2ws, dim=0)
        fx = torch.tensor(fxs)
        fy = torch.tensor(fys)

        camera_path = Cameras(
            fx=fx,
            fy=fy,
            cx=self.image_width / 2,
            cy=self.image_height / 2,
            camera_to_worlds=camera_to_worlds,
            camera_type=self.camera_type,
            times=None,
        )
        return camera_path

    def combine_matrices(self, position, rotations):
        # Convert rotations to a 3x3 rotation matrix
        rotation_matrix = self.euler_to_rotation_matrix(rotations)

        # Create the camera-to-world matrix
        camera_to_world_matrix = np.eye(4)
        camera_to_world_matrix[:3, :3] = rotation_matrix
        camera_to_world_matrix[:3, 3] = position

        return camera_to_world_matrix

    def euler_to_rotation_matrix(self, rotations):
        # Convert Euler angles to a rotation matrix
        roll, pitch, yaw = rotations

        cos_r = np.cos(roll)
        sin_r = np.sin(roll)
        cos_p = np.cos(pitch)
        sin_p = np.sin(pitch)
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)

        rotation_matrix = np.array([
            [cos_p * cos_y, cos_y * sin_p * sin_r - cos_r * sin_y, cos_r * cos_y * sin_p + sin_r * sin_y],
            [cos_p * sin_y, cos_r * cos_y + sin_p * sin_r * sin_y, -cos_y * sin_r + cos_r * sin_p * sin_y],
            [-sin_p, cos_p * sin_r, cos_p * cos_r]
        ])

        return rotation_matrix

    def create_camera_to_world_matrix(self, unity_position, obj_eular_in_deg):
        x, y, z = unity_position
        world_position = [z, -x, y]

        obj_eular_in_rad = obj_eular_in_deg * np.pi / 180
        v_c = np.array([0, 0, 1])
        R_uc = self.euler_to_rotation_matrix(obj_eular_in_rad)
        R_mu = np.array([[0, 0, 1,], [-1, 0, 0], [0, 1, 0]])
        R_mc = np.matmul(R_mu, R_uc)
        v_m = np.matmul(R_mc, v_c)
        unit_m = np.array([0, 0, -1])

        v1 = unit_m
        v2 = v_m

        cross_product = np.cross(v1, v2)
        yaw_angle = np.arctan2(cross_product[1], cross_product[0])
        pitch_angle = np.arcsin(cross_product[2])
        roll_angle = np.arctan2(-v1[2], v1[0])

        rotations = np.array([roll_angle, pitch_angle, yaw_angle])

        camera_to_world_matrix = self.combine_matrices(world_position, rotations)
        return camera_to_world_matrix

    def get_unity_position_rotation(self):
        x, y, z = map(float, input("Enter x, y, and z coordinates separated by spaces: ").split())
        r_x, r_y, r_z = map(float, input("Enter x, y, and z rotations separated by spaces: ").split())

        return [x, y, z], np.array([r_x, r_y, r_z])

    def execute(self) -> None:
        unity_position, obj_eular_in_deg = self.get_unity_position_rotation()

        camera_to_world_matrix = self.create_camera_to_world_matrix(unity_position, obj_eular_in_deg)

        focal_length = three_js_perspective_camera_focal_length(self.fov, self.image_height)
        
        camera_path = self.create_camera_object(camera_to_world_matrix, focal_length)
        
        self.render_img(
            self.pipeline,
            camera_path,
            output_foldername=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            output_format=self.output_format,
            image_format=self.image_format,
            jpeg_quality=self.jpeg_quality,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            colormap_options=self.colormap_options,
        )


def main_func():
    parser = argparse.ArgumentParser(description='Render Images')
    parser.add_argument('config', help='config file path')
    parser.add_argument('output', help='output file path')
    args = parser.parse_args()
    
    ri = RenderImages(args)
    
    while True:
        ri.execute()

if __name__ == "__main__":
    main_func()
