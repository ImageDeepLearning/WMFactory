import os
import cv2
import json
import numpy as np
from PIL import Image
import open3d as o3d

import torch
from typing import Union, Tuple

from .adaptive_depth_compression import create_adaptive_depth_compressor

from ..utils import (
    get_no_fg_img,
    get_fg_mask,
    get_bg_mask,
    get_filtered_mask,
    sheet_warping,
    depth_match,
    seed_all,
    build_depth_model,
    pred_pano_depth,
)


class WorldComposer:
    r"""WorldComposer is responsible for composing a layered world from input images and masks.
    It handles foreground object generation, background layer composition, and depth inpainting.
    Args:
        device (torch.device): The device to run the model on (default: "cuda").
        resolution (Tuple[int, int]): The resolution of the input images (width, height).
        filter_mask (bool): Whether to filter the foreground masks.
        kernel_scale (int): The scale factor for kernel size in mask processing (default: 1).
        adaptive_depth_compression (bool): Whether to enable adaptive depth compression (default: True).
        seed (int): Random seed for reproducibility.
    """

    def __init__(
        self,
        device: torch.device = "cuda",
        resolution: Tuple[int, int] = (3840, 1920),
        seed: int = 42,
        filter_mask: bool = False,
        kernel_scale: int = 1,
        adaptive_depth_compression: bool = True,
        max_fg_mesh_res: int = 3840,
        max_bg_mesh_res: int = 3840,
        max_sky_mesh_res: int = 1920,
        sky_mask_dilation_kernel: int = 5,
        bg_depth_compression_quantile: float = 0.92,
        fg_mask_erode_scale: float = 2.5,
        fg_filter_beta_scale: float = 3.3,
        fg_filter_alpha_scale: float = 0.15,
        sky_depth_margin: float = 1.02,
    ):
        r"""Initialize"""
        self.device = device
        self.resolution = resolution
        self.filter_mask = filter_mask
        self.kernel_scale = kernel_scale
        self.max_fg_mesh_res = max_fg_mesh_res
        self.max_bg_mesh_res = max_bg_mesh_res
        self.max_sky_mesh_res = max_sky_mesh_res
        self.sky_mask_dilation_kernel = sky_mask_dilation_kernel
        self.bg_depth_compression_quantile = bg_depth_compression_quantile
        self.fg_mask_erode_scale = fg_mask_erode_scale
        self.fg_filter_beta_scale = fg_filter_beta_scale
        self.fg_filter_alpha_scale = fg_filter_alpha_scale
        self.sky_depth_margin = sky_depth_margin

        # Adaptive deep compression configuration
        self.adaptive_depth_compression = adaptive_depth_compression
        self.depth_model = build_depth_model(device)

        # Initialize world composition variables
        self._init_list()
        # init seed
        seed_all(seed)

    def _init_list(self):
        self.layered_world_mesh = []
        self.layered_world_depth = []

    def _process_input(self, separate_pano, fg_bboxes):
        # get all inputs
        self.full_img = separate_pano["full_img"]
        self.no_fg1_img = separate_pano["no_fg1_img"]
        self.no_fg2_img = separate_pano["no_fg2_img"]
        self.sky_img = separate_pano["sky_img"]
        self.fg1_mask = separate_pano["fg1_mask"]
        self.fg2_mask = separate_pano["fg2_mask"]
        self.sky_mask = separate_pano["sky_mask"]
        self.fg1_bbox = fg_bboxes["fg1_bbox"]
        self.fg2_bbox = fg_bboxes["fg2_bbox"]

    def _process_sky_mask(self):
        r"""Process the sky mask to prepare it for further operations."""
        if self.sky_mask is not None:
            # The sky mask identifies non-sky regions, so it needs to be inverted.
            self.sky_mask = 1 - np.array(self.sky_mask) / 255.0
            if len(self.sky_mask.shape) > 2:
                self.sky_mask = self.sky_mask[:, :, 0]
            # Expand the sky mask to ensure complete coverage.
            kernel_size = self.sky_mask_dilation_kernel * self.kernel_scale
            self.sky_mask = (
                cv2.dilate(
                    self.sky_mask,
                    np.ones((kernel_size, kernel_size), np.uint8),
                    iterations=1,
                )
                if self.sky_mask.sum() > 0
                else self.sky_mask
            )
        else:
            # Create an empty mask if no sky is present.
            self.sky_mask = np.zeros((self.H, self.W))

    def _process_fg_mask(self, fg_mask):
        r"""Process the foreground mask to prepare it for further operations."""
        if fg_mask is not None:
            fg_mask = np.array(fg_mask)
            if len(fg_mask.shape) > 2:
                fg_mask = fg_mask[:, :, 0]
        return fg_mask

    def _load_separate_pano_from_dir(self, image_dir, sr):
        r"""Load separate panorama images and foreground bounding boxes from a directory.
        Args:
            image_dir (str): The directory containing the panorama images and bounding boxes.
            sr (bool): Whether to use super-resolution versions of the images.
        Returns:
            images (dict): A dictionary containing the loaded images with keys:
                - "full_img": Complete panorama image (PIL.Image.Image)
                - "no_fg1_img": Panorama with layer 1 foreground object removed (PIL.Image.Image)
                - "no_fg2_img": Panorama with layer 2 foreground object removed (PIL.Image.Image)
                - "sky_img": Sky region image (PIL.Image.Image)
                - "fg1_mask": Binary mask for layer 1 foreground object (PIL.Image.Image)
                - "fg2_mask": Binary mask for layer 2 foreground object (PIL.Image.Image)
                - "sky_mask": Binary mask for sky region (PIL.Image.Image)
            fg_bboxes (dict): A dictionary containing bounding boxes for foreground objects with keys:
                - "fg1_bbox": List of dicts with keys 'label', 'bbox', 'score' for layer 1 object
                - "fg2_bbox": List of dicts with keys 'label', 'bbox', 'score' for layer 2 object
        Raises:
            FileNotFoundError: If the specified image directory does not exist.
        """
        # Define base image files
        image_files = {
            "full_img": "full_image.png",
            "no_fg1_img": "remove_fg1_image.png",
            "no_fg2_img": "remove_fg2_image.png",
            "sky_img": "sky_image.png",
            "fg1_mask": "fg1_mask.png",
            "fg2_mask": "fg2_mask.png",
            "sky_mask": "sky_mask.png",
        }
        # Use super-resolution versions if sr flag is set
        if sr:
            print("***Using super-resolution input image***")
            for key in ["full_img", "no_fg1_img", "no_fg2_img", "sky_img"]:
                image_files[key] = image_files[key].replace(".png", "_sr.png")

        # Check if the directory exists
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"The image directory does not exist: {image_dir}")

        # Load and adjust all images
        images = {}
        fg1_bbox_scale = 1
        fg2_bbox_scale = 1
        for name, filename in image_files.items():
            filepath = os.path.join(image_dir, filename)
            if not os.path.exists(filepath):
                images[name] = None
            else:
                img = Image.open(filepath)
                if img.size != self.resolution:
                    print(
                        f"Transform the image {name} from {img.size} rescale to {self.resolution}"
                    )
                    # Select different resampling methods based on image type
                    resample = Image.NEAREST if "mask" in name else Image.BICUBIC
                    if "fg1_mask" in name and img.size != self.resolution:
                        fg1_bbox_scale = self.resolution[0] / img.size[0]
                    if "fg2_mask" in name and img.size != self.resolution:
                        fg2_bbox_scale = self.resolution[0] / img.size[0]
                    img = img.resize(self.resolution, resample=resample)
                images[name] = img

        # Check resolution
        if self.resolution is not None:
            for name, img in images.items():
                if img is not None:
                    assert (
                        img.size == self.resolution
                    ), f"{name} resolution does not match"

        # Load foreground object bbox
        fg_bboxes = {}
        fg_bbox_files = {
            "fg1_bbox": "fg1.json",
            "fg2_bbox": "fg2.json",
        }
        for name, filename in fg_bbox_files.items():
            filepath = os.path.join(image_dir, filename)
            if not os.path.exists(filepath):
                fg_bboxes[name] = None
            else:
                fg_bboxes[name] = json.load(open(filepath))
                if "fg1" in name:
                    for i in range(len(fg_bboxes[name]["bboxes"])):
                        fg_bboxes[name]["bboxes"][i]["bbox"] = [
                            x * fg1_bbox_scale
                            for x in fg_bboxes[name]["bboxes"][i]["bbox"]
                        ]
                if "fg2" in name:
                    for i in range(len(fg_bboxes[name]["bboxes"])):
                        fg_bboxes[name]["bboxes"][i]["bbox"] = [
                            x * fg2_bbox_scale
                            for x in fg_bboxes[name]["bboxes"][i]["bbox"]
                        ]

        return images, fg_bboxes

    def generate_world(self, **kwargs):
        r"""Generate a 3D world composition from panorama and foreground objects

        Args:
            **kwargs: Additional keyword arguments containing:
                separate_pano (np.ndarray):
                    Panorama image split into separate cubemap faces [6, H, W, C]
                fg_bboxes (List[Dict]):
                    List of foreground object bounding boxes
                world_type (str):
                    World generation mode:
                    - 'mesh': export mesh

        Returns:
            Tuple: A tuple containing:
                world (np.ndarray):
                    Rendered 3D world view [H,W,3] in RGB format
                layered_world_depth (np.ndarray):
                    Depth map of the composition [H,W]
                    with values in [0,1] range (1=far)
                generated_fg_objects (List[Dict]):
                    Processed foreground objects
        """
        # temporary input setting
        separate_pano = kwargs["separate_pano"]
        fg_bboxes = kwargs["fg_bboxes"]
        world_type = kwargs["world_type"]

        layered_world_mesh = self._compose_layered_world(
            separate_pano, fg_bboxes, world_type=world_type
        )
        return layered_world_mesh

    def _compose_background_layer(self):
        r"""Compose the background layer of the world."""
        # The background layer is composed of the full image without foreground objects.
        if self.BG_MASK.sum() == 0:
            return

        print(f"🏞️ Composing the background layer...")
        if self.fg_status == "no_fg":
            self.no_fg_img_depth = self.full_img_depth
        else:
            # For cascade inpainting, use the last layer's depth as known depth.
            if self.fg_status == "both_fg1_fg2":
                inpaint_mask = self.fg2_mask.astype(np.bool_).astype(np.uint8)
            else:
                inpaint_mask = self.FG_MASK

            # Align the depth of the background layer to the depth of the panoramic image
            self.no_fg_img_depth = pred_pano_depth(
                self.depth_model,
                self.no_fg_img,
                img_name="background",
                last_layer_mask=inpaint_mask,
                last_layer_depth=self.layered_world_depth[-1],
            )

            self.no_fg_img_depth = depth_match(
                self.full_img_depth, self.no_fg_img_depth, self.BG_MASK
            )

        # Apply adaptive depth compression considering foreground layers and scene characteristics
        distance = self.no_fg_img_depth["distance"]
        if (
            hasattr(self, "adaptive_depth_compression")
            and self.adaptive_depth_compression
        ):
            # Automatically determine scene type based on sky_img
            scene_type = "indoor" if self.sky_img is None else "outdoor"
            depth_compressor = create_adaptive_depth_compressor(scene_type=scene_type)
            self.no_fg_img_depth["distance"] = (
                depth_compressor.compress_background_depth(
                    distance, self.layered_world_depth, bg_mask=1 - self.sky_mask
                )
            )
        else:
            # Use a simple quantile-based depth compression method.
            q_val = torch.quantile(distance, self.bg_depth_compression_quantile)
            self.no_fg_img_depth["distance"] = torch.clamp(distance, max=q_val)

        layer_depth_i = self.no_fg_img_depth.copy()
        layer_depth_i["name"] = "background"
        layer_depth_i["mask"] = 1 - self.sky_mask
        layer_depth_i["type"] = "bg"
        self.layered_world_depth.append(layer_depth_i)

        if "mesh" in self.world_type:
            no_fg_img_mesh = sheet_warping(
                self.no_fg_img_depth,
                excluded_region_mask=torch.from_numpy(self.sky_mask).bool(),
                max_size=self.max_bg_mesh_res,
            )
            self.layered_world_mesh.append({"type": "bg", "mesh": no_fg_img_mesh})

    def _compose_foreground_layer(self):
        if self.fg_status == "no_fg":
            return

        print(f"🧩 Composing the foreground layers...")

        # Obtain the list of foreground layers
        fg_layer_list = []
        if self.fg_status == "both_fg1_fg2":
            fg_layer_list.append(
                [self.full_img, self.fg1_mask, self.fg1_bbox, "fg1"]
            )  # fg1 mesh
            fg_layer_list.append(
                [self.no_fg1_img, self.fg2_mask, self.fg2_bbox, "fg2"]
            )  # fg2 mesh
        elif self.fg_status == "only_fg1":
            fg_layer_list.append(
                [self.full_img, self.fg1_mask, self.fg1_bbox, "fg1"]
            )  # fg1 mesh
        elif self.fg_status == "only_fg2":
            fg_layer_list.append(
                [self.no_fg1_img, self.fg2_mask, self.fg2_bbox, "fg2"]
            )  # fg2 mesh

        # Determine whether to generate foreground objects or directly project foreground layers
        project_object_layer = ["fg1", "fg2"]

        for fg_i_img, fg_i_mask, fg_i_bbox, fg_i_type in fg_layer_list:
            print(f"\t - Composing the foreground layer: {fg_i_type}")
            # 1. Estimate the depth of the foreground layer
            # If there are fg1 and fg2, then fg1_img is the panoramic image itself, without the need to estimate depth
            if len(fg_layer_list) > 1:
                if fg_i_type == "fg1":
                    fg_i_img_depth = self.full_img_depth
                elif fg_i_type == "fg2":
                    fg_i_img_depth = pred_pano_depth(
                        self.depth_model,
                        fg_i_img,
                        img_name=f"{fg_i_type}",
                        last_layer_mask=self.fg1_mask.astype(np.bool_).astype(np.uint8),
                        last_layer_depth=self.full_img_depth,
                    )
                    # fg2 only needs to align the depth of the fg2 object area
                    fg2_exclude_fg1_mask = np.logical_and(
                        fg_i_mask.astype(np.bool_), 1 - self.fg1_mask.astype(np.bool_)
                    )

                    # Align the depth of the foreground layer to the depth of the panoramic image
                    fg_i_img_depth = depth_match(
                        self.full_img_depth, fg_i_img_depth, fg2_exclude_fg1_mask
                    )
                else:
                    raise ValueError(f"Invalid foreground object type: {fg_i_type}")
            else:
                # If only fg1 or fg2 exists, its image is the panoramic image, so depth estimation is not required.
                fg_i_img_depth = self.full_img_depth

            # Compress outliers in the foreground depth.
            if (
                hasattr(self, "adaptive_depth_compression")
                and self.adaptive_depth_compression
            ):
                depth_compressor = create_adaptive_depth_compressor()
                fg_i_img_depth["distance"] = depth_compressor.compress_foreground_depth(
                    fg_i_img_depth["distance"], fg_i_mask
                )

            in_fg_i_mask = fg_i_mask.copy()
            if fg_i_mask.sum() > 0:
                # 2. Perform sheet warping.
                if fg_i_type in project_object_layer:
                    in_fg_i_mask = self._project_fg_depth(
                        fg_i_img_depth, fg_i_mask, fg_i_type
                    )
                else:
                    raise ValueError(f"Invalid foreground object type: {fg_i_type}")
            else:
                # If no objects are in the foreground layer, it won't be added to the layered world depth.
                pass

            # save layered depth
            layer_depth_i = fg_i_img_depth.copy()
            layer_depth_i["name"] = fg_i_type
            # Using edge filtered masks to ensure the accuracy of foreground depth during depth compression
            layer_depth_i["mask"] = (
                in_fg_i_mask if in_fg_i_mask is not None else np.zeros_like(fg_i_mask)
            )
            layer_depth_i["type"] = fg_i_type
            self.layered_world_depth.append(layer_depth_i)

    def _project_fg_depth(self, fg_i_img_depth, fg_i_mask, fg_i_type):
        r"""Project the foreground depth to create a mesh or Gaussian splatting object."""
        in_fg_i_mask = fg_i_mask.astype(np.bool_).astype(
            np.uint8
        )
        # Erode the mask to remove edge artifacts from foreground objects.
        erode_size = int(self.fg_mask_erode_scale * self.kernel_scale)
        eroded_in_fg_i_mask = cv2.erode(
            in_fg_i_mask, np.ones((erode_size, erode_size), np.uint8), iterations=1
        )  # The result is a uint8 array with values of 0 or 1.

        # Filter edges
        if self.filter_mask:
            filtered_fg_i_img_mask = (
                1
                - get_filtered_mask(
                    1.0 / fg_i_img_depth["distance"][None, :, :, None],
                    beta=self.fg_filter_beta_scale * self.kernel_scale,
                    alpha_threshold=self.fg_filter_alpha_scale * self.kernel_scale,
                    device=self.device,
                )
                .squeeze()
                .cpu()
            )
            # Convert to binary mask
            filtered_fg_i_img_mask = 1 - filtered_fg_i_img_mask.numpy()

            # Combine eroded mask with filtered mask
            eroded_in_fg_i_mask = np.logical_and(
                eroded_in_fg_i_mask, filtered_fg_i_img_mask
            )

        # Process the eroded mask to create the final binary mask
        in_fg_i_mask = eroded_in_fg_i_mask > 0.5
        out_fg_i_mask = 1 - in_fg_i_mask

        # Convert the depth image to a mesh or Gaussian splatting object
        if "mesh" in self.world_type:
            fg_i_mesh = sheet_warping(
                fg_i_img_depth,
                excluded_region_mask=torch.from_numpy(out_fg_i_mask).bool(),
                max_size=self.max_fg_mesh_res,
            )
            self.layered_world_mesh.append({"type": fg_i_type, "mesh": fg_i_mesh})

        return in_fg_i_mask

    def _compose_sky_layer(self):
        r"""Compose the sky layer of the world."""
        if self.sky_img is not None:
            print(f"🕐 Composing the sky layer...")
            self.sky_img = torch.tensor(
                np.array(self.sky_img), device=self.full_img_depth["rgb"].device
            )

            # Calculate the maximum depth value of all foreground and background layers
            max_scene_depth = torch.tensor(
                0.0, device=self.full_img_depth["rgb"].device
            )
            for layer in self.layered_world_depth:
                layer_depth = layer["distance"]
                layer_mask = layer.get("mask", None)

                if layer_mask is not None:
                    if not isinstance(layer_mask, torch.Tensor):
                        layer_mask = torch.from_numpy(layer_mask).to(layer_depth.device)
                    mask_bool = layer_mask.bool()
                    if (
                        mask_bool.sum() > 0
                    ):  # Only search for the maximum value within the mask area
                        layer_max = layer_depth[mask_bool].max()
                        max_scene_depth = torch.max(max_scene_depth, layer_max)
                else:
                    # If there is no mask, consider the entire depth map
                    max_scene_depth = torch.max(max_scene_depth, layer_depth.max())

            # Set the sky depth to be slightly greater than the maximum scene depth.
            sky_distance = self.sky_depth_margin * max_scene_depth if max_scene_depth > 0 else 3.0

            sky_pred = {
                "rgb": self.sky_img,
                "rays": self.full_img_depth["rays"],
                "distance": sky_distance
                * torch.ones_like(self.full_img_depth["distance"]),
            }

            if "mesh" in self.world_type:
                # The sky doesn't need smooth edges with jagged edges
                sky_mesh = sheet_warping(
                    sky_pred,
                    connect_boundary_max_dist=None,
                    max_size=self.max_sky_mesh_res,
                )
                self.layered_world_mesh.append({"type": "sky", "mesh": sky_mesh})

    def _compose_layered_world(
        self,
        separate_pano: dict,
        fg_bboxes: dict,
        world_type: list = ["mesh"],
    ) -> Union[o3d.geometry.TriangleMesh]:
        r"""
        Compose each layer into a complete world
        Args:
            separate_pano: dict containing the following images:
                full_img: Complete panorama image (PIL.Image.Image)
                no_fg1_img: Panorama with layer 1 foreground object removed (PIL.Image.Image)
                no_fg2_img: Panorama with layer 2 foreground object removed (PIL.Image.Image)
                sky_img: Sky region image (PIL.Image.Image)
                fg1_mask: Binary mask for layer 1 foreground object (PIL.Image.Image)
                fg2_mask: Binary mask for layer 2 foreground object (PIL.Image.Image)
                sky_mask: Binary mask for sky region (PIL.Image.Image)

            fg_bboxes: dict containing bounding boxes for foreground objects:
                fg1_bbox: List of dicts with keys 'label', 'bbox', 'score' for layer 1 object
                fg2_bbox: List of dicts with keys 'label', 'bbox', 'score' for layer 2 object

            world_type: list, ["mesh"]

            filter_mask: bool, whether to filter the mask

        Returns:
            layered_world: dict containing the following:
                mesh: list of o3d.geometry.TriangleMesh
                objects: list of ImageWithOneObject
        """
        self.world_type = world_type
        self._process_input(separate_pano, fg_bboxes)
        self.W, self.H = self.full_img.size

        self._init_list()

        # Processing sky and foreground masks
        self._process_sky_mask()
        self.fg1_mask = self._process_fg_mask(self.fg1_mask)
        self.fg2_mask = self._process_fg_mask(self.fg2_mask)

        # Overall foreground mask: Merge multiple foreground masks, background mask: Excluding sky
        self.FG_MASK = get_fg_mask(self.fg1_mask, self.fg2_mask)
        self.BG_MASK = get_bg_mask(self.sky_mask, self.FG_MASK, self.kernel_scale)

        # Obtain background+sky layer (no_fg_img
        self.no_fg_img, self.fg_status = get_no_fg_img(
            self.no_fg1_img, self.no_fg2_img, self.full_img
        )

        # Predicting the Depth of Panoramic Images
        self.full_img_depth = pred_pano_depth(  # fg1 depth
            self.depth_model,
            self.full_img,
            img_name="full_img",
        )

        # Layered construction of the world
        print(f"🎨 Start to compose the world layer by layer...")
        # 1. The foreground layers
        self._compose_foreground_layer()

        # 2. The background layers
        self._compose_background_layer()

        # 3. The sky layers
        self._compose_sky_layer()

        print("🎉 Congratulations! World composition completed successfully!")

        return self.layered_world_mesh
