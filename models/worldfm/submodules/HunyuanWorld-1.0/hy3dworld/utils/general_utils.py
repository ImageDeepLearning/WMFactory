import cv2
import numpy as np
from typing import Optional, Literal

import random
import matplotlib
import open3d as o3d

import torch
import torch.nn.functional as F
from collections import defaultdict


def spherical_uv_to_directions(uv: np.ndarray):
    r"""
    Convert spherical UV coordinates to 3D directions.
    Args:
        uv (np.ndarray): UV coordinates in the range [0, 1]. Shape: (H, W, 2).
    Returns:
        directions (np.ndarray): 3D directions corresponding to the UV coordinates. Shape: (H, W, 3).
    """
    theta, phi = (1 - uv[..., 0]) * (2 * np.pi), uv[..., 1] * np.pi
    directions = np.stack([np.sin(phi) * np.cos(theta),
                          np.sin(phi) * np.sin(theta), np.cos(phi)], axis=-1)
    return directions


def depth_match(init_pred: dict, bg_pred: dict, mask: np.ndarray, quantile: float = 0.3) -> dict:
    r"""
    Match the background depth map to the scale of the initial depth map.
    Args:
        init_pred (dict): Initial depth prediction containing "distance" key.
        bg_pred (dict): Background depth prediction containing "distance" key.
        mask (np.ndarray): Binary mask indicating valid pixels in the background depth map.
        quantile (float): Quantile to use for selecting the depth range for scale matching.
    Returns:
        bg_pred (dict): Background depth prediction with adjusted "distance" key.
    """
    valid_mask = mask > 0
    if valid_mask.sum() == 0:
        return bg_pred  # No valid pixels to match
    init_distance = init_pred["distance"][valid_mask]
    bg_distance = bg_pred["distance"][valid_mask]

    init_mask = init_distance < torch.quantile(init_distance, quantile)
    bg_mask = bg_distance < torch.quantile(bg_distance, quantile)
    scale = init_distance[init_mask].median() / bg_distance[bg_mask].median()
    bg_pred["distance"] *= scale
    return bg_pred



def _fill_small_boundary_spikes(
    mesh: o3d.geometry.TriangleMesh,
    max_bridge_dist: float,
    repeat_times: int = 3,
    max_connection_step: int = 8,
) -> o3d.geometry.TriangleMesh:
    r"""
    Fill small boundary spikes in a mesh by creating triangles between boundary vertices.
    Args:
        mesh (o3d.geometry.TriangleMesh): The input mesh to process.
        max_bridge_dist (float): Maximum distance allowed for bridging boundary vertices.
        repeat_times (int): Number of iterations to repeat the filling process.
        max_connection_step (int): Maximum number of steps to connect boundary vertices.
    Returns:
        o3d.geometry.TriangleMesh: The mesh with small boundary spikes filled.
    """
    for iteration in range(repeat_times):
        if not mesh.has_triangles() or not mesh.has_vertices():
            return mesh

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        # 1. Identify boundary edges
        edge_to_triangle_count = defaultdict(int)

        for tri_idx, tri in enumerate(triangles):
            for i in range(3):
                v1_idx, v2_idx = tri[i], tri[(i + 1) % 3]
                edge = tuple(sorted((v1_idx, v2_idx)))
                edge_to_triangle_count[edge] += 1

        boundary_edges = [edge for edge,
                          count in edge_to_triangle_count.items() if count == 1]

        if not boundary_edges:
            return mesh

        # 2. Create an adjacency list for boundary vertices using only boundary edges
        boundary_adj = defaultdict(list)
        for v1_idx, v2_idx in boundary_edges:
            boundary_adj[v1_idx].append(v2_idx)
            boundary_adj[v2_idx].append(v1_idx)

        # 3. Process boundary vertices with new smooth filling algorithm
        new_triangles_list = []
        edge_added = defaultdict(bool)

        # print(f"DEBUG: Found {len(boundary_edges)} boundary edges.")
        # print(f"DEBUG: Max bridge distance set to: {max_bridge_dist}")

        new_triangles_added_count = 0

        for v_curr_idx, neighbors in boundary_adj.items():
            if len(neighbors) != 2:  # Only process vertices with exactly 2 boundary neighbors
                continue

            v_a_idx, v_b_idx = neighbors[0], neighbors[1]

            # Skip if these vertices already form a triangle
            potential_edge = tuple(sorted((v_a_idx, v_b_idx)))
            if edge_to_triangle_count[potential_edge] > 0 or edge_added[potential_edge]:
                continue

            # Calculate distances
            v_curr_coord = vertices[v_curr_idx]
            v_a_coord = vertices[v_a_idx]
            v_b_coord = vertices[v_b_idx]

            dist_a_b = np.linalg.norm(v_a_coord - v_b_coord)

            # Skip if distance exceeds threshold
            if dist_a_b > max_bridge_dist:
                continue

            # Create simple triangle (v_a, v_b, v_curr)
            new_triangles_list.append([v_a_idx, v_b_idx, v_curr_idx])
            new_triangles_added_count += 1
            edge_added[potential_edge] = True

            # Mark edges as processed
            edge_added[tuple(sorted((v_curr_idx, v_a_idx)))] = True
            edge_added[tuple(sorted((v_curr_idx, v_b_idx)))] = True

        # 4. Now process multi-step connections for better smoothing
        # First build boundary chains for multi-step connections
        boundary_loops = []
        visited_vertices = set()

        # Find boundary vertices with exactly 2 neighbors (part of continuous chains)
        chain_starts = [v for v in boundary_adj if len(
            boundary_adj[v]) == 2 and v not in visited_vertices]

        for start_vertex in chain_starts:
            if start_vertex in visited_vertices:
                continue

            chain = []
            curr_vertex = start_vertex

            # Follow the chain in one direction
            while curr_vertex not in visited_vertices:
                visited_vertices.add(curr_vertex)
                chain.append(curr_vertex)

                next_candidates = [
                    n for n in boundary_adj[curr_vertex] if n not in visited_vertices]
                if not next_candidates:
                    break

                curr_vertex = next_candidates[0]

            if len(chain) >= 3:
                boundary_loops.append(chain)

        # Process each boundary chain for multi-step smoothing
        for chain in boundary_loops:
            chain_length = len(chain)

            # Skip very small chains
            if chain_length < 3:
                continue

            # Compute multi-step connections
            max_step = min(max_connection_step, chain_length - 1)

            for i in range(chain_length):
                anchor_idx = chain[i]
                anchor_coord = vertices[anchor_idx]

                for step in range(3, max_step + 1):
                    if i + step >= chain_length:
                        break

                    far_idx = chain[i + step]
                    far_coord = vertices[far_idx]

                    # Check distance criteria
                    dist_anchor_far = np.linalg.norm(anchor_coord - far_coord)
                    if dist_anchor_far > max_bridge_dist * step:
                        continue

                    # Check if anchor and far are already connected
                    edge_anchor_far = tuple(sorted((anchor_idx, far_idx)))
                    if edge_to_triangle_count[edge_anchor_far] > 0 or edge_added[edge_anchor_far]:
                        continue

                    # Create fan triangles
                    fan_valid = True
                    fan_triangles = []

                    prev_mid_idx = anchor_idx

                    for j in range(1, step):
                        mid_idx = chain[i + j]

                        if prev_mid_idx != anchor_idx:
                            tri_edge1 = tuple(sorted((anchor_idx, mid_idx)))
                            tri_edge2 = tuple(sorted((prev_mid_idx, mid_idx)))

                            # Check if edges already exist (not created by our fan)
                            if (edge_to_triangle_count[tri_edge1] > 0 and not edge_added[tri_edge1]) or \
                               (edge_to_triangle_count[tri_edge2] > 0 and not edge_added[tri_edge2]):
                                fan_valid = False
                                break

                            fan_triangles.append(
                                [anchor_idx, prev_mid_idx, mid_idx])

                        prev_mid_idx = mid_idx

                    # Add final triangle to connect to far_idx
                    if fan_valid:
                        fan_triangles.append(
                            [anchor_idx, prev_mid_idx, far_idx])

                    # Add all fan triangles if valid
                    if fan_valid and fan_triangles:
                        for triangle in fan_triangles:
                            v_a, v_b, v_c = triangle
                            edge_ab = tuple(sorted((v_a, v_b)))
                            edge_bc = tuple(sorted((v_b, v_c)))
                            edge_ac = tuple(sorted((v_a, v_c)))

                            new_triangles_list.append(triangle)
                            new_triangles_added_count += 1

                            edge_added[edge_ab] = True
                            edge_added[edge_bc] = True
                            edge_added[edge_ac] = True

                        # Once we've added a fan, move to the next anchor
                        break

        if new_triangles_added_count == 0:
            break

        # Update the mesh with new triangles
        if new_triangles_list:
            all_triangles_np = np.vstack(
                (triangles, np.array(new_triangles_list, dtype=np.int32)))

            final_mesh = o3d.geometry.TriangleMesh()
            final_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            final_mesh.triangles = o3d.utility.Vector3iVector(all_triangles_np)

            if mesh.has_vertex_colors():
                final_mesh.vertex_colors = mesh.vertex_colors

            # Clean up the mesh
            final_mesh.remove_degenerate_triangles()
            final_mesh.remove_unreferenced_vertices()
            mesh = final_mesh

    return mesh


def pano_sheet_warping(
    rgb: torch.Tensor,  # (H, W, 3) RGB image, values [0, 1]
    distance: torch.Tensor,  # (H, W) Distance map
    rays: torch.Tensor,  # (H, W, 3) Ray directions (unit vectors ideally)
    # (H, W) Optional boolean mask
    excluded_region_mask: Optional[torch.Tensor] = None,
    max_size: int = 4096,  # Max dimension for resizing
    device: Literal["cuda", "cpu"] = "cuda",  # Computation device
    # Max distance to bridge boundary vertices
    connect_boundary_max_dist: Optional[float] = 0.5,
    connect_boundary_repeat_times: int = 2
) -> o3d.geometry.TriangleMesh:
    r"""
    Converts panoramic RGBD data (image, distance, rays) into an Open3D mesh.
    Args:
        image: Input RGB image tensor (H, W, 3), uint8 or float [0, 255].
        distance: Input distance map tensor (H, W).
        rays: Input ray directions tensor (H, W, 3). Assumed to originate from (0,0,0).
        excluded_region_mask: Optional boolean mask tensor (H, W). True values indicate regions to potentially exclude.
        max_size: Maximum size (height or width) to resize inputs to.
        device: The torch device ('cuda' or 'cpu') to use for computations.

    Returns:
        An Open3D TriangleMesh object.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, "Image must be HxWx3"
    assert distance.ndim == 2, "Distance must be HxW"
    assert rays.ndim == 3 and rays.shape[2] == 3, "Rays must be HxWx3"
    assert (
        rgb.shape[:2] == distance.shape[:2] == rays.shape[:2]
    ), "Input shapes must match"

    mask = excluded_region_mask

    if mask is not None:
        assert (
            mask.ndim == 2 and mask.shape[:2] == rgb.shape[:2]
        ), "Mask shape must match"
        assert mask.dtype == torch.bool, "Mask must be a boolean tensor"

    rgb = rgb.to(device)
    distance = distance.to(device)
    rays = rays.to(device)
    if mask is not None:
        mask = mask.to(device)

    H, W = distance.shape
    if max(H, W) > max_size:
        scale = max_size / max(H, W)
    else:
        scale = 1.0

    # --- Resize Inputs ---
    rgb_nchw = rgb.permute(2, 0, 1).unsqueeze(0)
    distance_nchw = distance.unsqueeze(0).unsqueeze(0)
    rays_nchw = rays.permute(2, 0, 1).unsqueeze(0)

    rgb_resized = (
        F.interpolate(
            rgb_nchw,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )
        .squeeze(0)
        .permute(1, 2, 0)
    )

    distance_resized = (
        F.interpolate(
            distance_nchw,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )
        .squeeze(0)
        .squeeze(0)
    )

    rays_resized_nchw = F.interpolate(
        rays_nchw,
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False,
    )

    # IMPORTANT: Renormalize ray directions after interpolation
    rays_resized = rays_resized_nchw.squeeze(0).permute(1, 2, 0)
    rays_norm = torch.linalg.norm(rays_resized, dim=-1, keepdim=True)
    rays_resized = rays_resized / (rays_norm + 1e-8)

    if mask is not None:
        mask_resized = (
            F.interpolate(
                # Needs float for interpolation
                mask.unsqueeze(0).unsqueeze(0).float(),
                scale_factor=scale,
                mode="nearest",  # Or 'nearest' if sharp boundaries are critical
                # align_corners=False,
                recompute_scale_factor=False,
            )
            .squeeze(0)
            .squeeze(0)
        )
        mask_resized = mask_resized > 0.5  # Convert back to boolean
    else:
        mask_resized = None

    H_new, W_new = distance_resized.shape  # Get new dimensions

    # --- Calculate 3D Vertices ---
    # Vertex position = origin + distance * ray_direction
    # Assuming origin is (0, 0, 0)
    distance_flat = distance_resized.reshape(-1, 1)  # (H*W, 1)
    rays_flat = rays_resized.reshape(-1, 3)  # (H*W, 3)
    vertices = distance_flat * rays_flat  # (H*W, 3)
    vertex_colors = rgb_resized.reshape(-1, 3)  # (H*W, 3)

    # --- Generate Mesh Faces (Triangles from Quads) ---
    # Vectorized approach for generating faces, including seam connection
    # Rows for the top of quads
    row_indices = torch.arange(0, H_new - 1, device=device)
    # Columns for the left of quads (includes last col for wrapping)
    col_indices = torch.arange(0, W_new, device=device)

    # Create 2D grids of row and column coordinates for quad corners
    # These represent the (row, col) of the top-left vertex of each quad
    # Shape: (H_new-1, W_new)
    quad_row_coords = row_indices.view(-1, 1).expand(-1, W_new)
    quad_col_coords = col_indices.view(
        1, -1).expand(H_new-1, -1)  # Shape: (H_new-1, W_new)

    # Top-left vertex indices
    tl_row, tl_col = quad_row_coords, quad_col_coords
    # Top-right vertex indices (with wrap-around)
    tr_row, tr_col = quad_row_coords, (quad_col_coords + 1) % W_new
    # Bottom-left vertex indices
    bl_row, bl_col = (quad_row_coords + 1), quad_col_coords
    # Bottom-right vertex indices (with wrap-around)
    br_row, br_col = (quad_row_coords + 1), (quad_col_coords + 1) % W_new

    # Convert 2D (row, col) coordinates to 1D vertex indices
    tl = tl_row * W_new + tl_col
    tr = tr_row * W_new + tr_col
    bl = bl_row * W_new + bl_col
    br = br_row * W_new + br_col

    # Apply mask if provided
    if mask_resized is not None:
        # Get mask values for each corner of the quads
        mask_tl_vals = mask_resized[tl_row, tl_col]
        mask_tr_vals = mask_resized[tr_row, tr_col]
        mask_bl_vals = mask_resized[bl_row, bl_col]
        mask_br_vals = mask_resized[br_row, br_col]

        # A quad is kept if none of its vertices are masked
        # Shape: (H_new-1, W_new)
        quad_keep_mask = ~(mask_tl_vals | mask_tr_vals |
                           mask_bl_vals | mask_br_vals)

        # Filter vertex indices based on the keep mask
        tl = tl[quad_keep_mask]  # Result is flattened
        tr = tr[quad_keep_mask]
        bl = bl[quad_keep_mask]
        br = br[quad_keep_mask]
    else:
        # If no mask, flatten all potential quads' vertex indices
        tl = tl.flatten()
        tr = tr.flatten()
        bl = bl.flatten()
        br = br.flatten()

    # Create triangles (two per quad)
    # Using the same winding order as before: (tl, tr, bl) and (tr, br, bl)
    tri1 = torch.stack([tl, tr, bl], dim=1)
    tri2 = torch.stack([tr, br, bl], dim=1)
    faces = torch.cat([tri1, tri2], dim=0)

    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices.cpu().numpy())
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(
        vertex_colors.cpu().numpy())
    mesh_o3d.remove_unreferenced_vertices()
    mesh_o3d.remove_degenerate_triangles()

    if connect_boundary_max_dist is not None and connect_boundary_max_dist > 0:
        mesh_o3d = _fill_small_boundary_spikes(
            mesh_o3d, connect_boundary_max_dist, connect_boundary_repeat_times)
        # Recompute normals after potential modification, if mesh still valid
        if mesh_o3d.has_triangles() and mesh_o3d.has_vertices():
            mesh_o3d.compute_vertex_normals()
            # Also computes triangle normals if vertex normals are computed
            mesh_o3d.compute_triangle_normals()

    return mesh_o3d


def get_no_fg_img(no_fg1_img, no_fg2_img, full_img):
    r"""Get the image without foreground objects based on available inputs.
    Args:
        no_fg1_img: Image with foreground layer 1 removed
        no_fg2_img: Image with foreground layer 2 removed
        full_img: Original full image
    Returns:
        Image without foreground objects, defaulting to full image if no fg-removed images available
    """
    fg_status = None
    if no_fg1_img is not None and no_fg2_img is not None:
        no_fg_img = no_fg2_img
        fg_status = "both_fg1_fg2"
    elif no_fg1_img is not None and no_fg2_img is None:
        no_fg_img = no_fg1_img
        fg_status = "only_fg1"
    elif no_fg1_img is None and no_fg2_img is not None:
        no_fg_img = no_fg2_img
        fg_status = "only_fg2"
    else:
        no_fg_img = full_img
        fg_status = "no_fg"

    assert fg_status is not None

    return no_fg_img, fg_status


def get_fg_mask(fg1_mask, fg2_mask):
    r"""
    Combine foreground masks from two layers.
    Args:
        fg1_mask: Foreground mask for layer 1
        fg2_mask: Foreground mask for layer 2
    Returns:
        Combined foreground mask, or None if both are None
    """
    if fg1_mask is not None and fg2_mask is not None:
        fg_mask = np.logical_or(fg1_mask, fg2_mask)
    elif fg1_mask is not None:
        fg_mask = fg1_mask
    elif fg2_mask is not None:
        fg_mask = fg2_mask
    else:
        fg_mask = None

    if fg_mask is not None:
        fg_mask = fg_mask.astype(np.bool_).astype(np.uint8)
    return fg_mask


def get_bg_mask(sky_mask, fg_mask, kernel_scale, dilation_kernel_size: int = 3):
    r"""
    Generate background mask based on sky and foreground masks.
    Args:
        sky_mask: Sky mask (boolean array)
        fg_mask: Foreground mask (boolean array)
        kernel_scale: Scale factor for the kernel size
        dilation_kernel_size: The size of the dilation kernel.
    Returns:
        Background mask as a boolean array, where True indicates background pixels.
    """
    kernel_size = dilation_kernel_size * kernel_scale
    if fg_mask is not None:
        bg_mask = np.logical_and(
            (1 - cv2.dilate(fg_mask,
             np.ones((kernel_size, kernel_size), np.uint8), iterations=1)),
            (1 - sky_mask),
        ).astype(np.uint8)
    else:
        bg_mask = 1 - sky_mask
    return bg_mask


def get_filtered_mask(disparity, beta=100, alpha_threshold=0.3, device="cuda"):
    """
    filter the disparity map using sobel kernel, then mask out the edge (depth discontinuity)
    Args:
        disparity: Disparity map in BHWC format, shape [b, h, w, 1]
        beta: Exponential decay factor for the Sobel magnitude
        alpha_threshold: Threshold for visibility mask
        device: Device to perform computations on, either 'cuda' or 'cpu'
    Returns:
        vis_mask: Visibility mask in BHWC format, shape [b, h, w, 1]
    """
    b, h, w, _ = disparity.size()
    # Permute to NCHW format: [b, 1, h, w]
    disparity_nchw = disparity.permute(0, 3, 1, 2)

    # Pad H and W dimensions with replicate padding
    disparity_padded = F.pad(
        disparity_nchw, (2, 2, 2, 2), mode="replicate"
    )  # Pad last two dims (W, H), [b, 1, h+4, w+4]

    kernel_x = (
        torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
        .to(device)
    )
    kernel_y = (
        torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    # Apply Sobel filters
    sobel_x = F.conv2d(
        disparity_padded, kernel_x, padding=(1, 1)
    )  # Output: [b, 1, h+4, w+4] # Corrected padding
    sobel_y = F.conv2d(
        disparity_padded, kernel_y, padding=(1, 1)
    )  # Output: [b, 1, h+4, w+4] # Corrected padding

    # Calculate magnitude
    sobel_mag_padded = torch.sqrt(
        sobel_x**2 + sobel_y**2
    )  # Shape: [b, 1, h+4, w+4]

    # Remove padding
    sobel_mag = sobel_mag_padded[
        :, :, 2:-2, 2:-2
    ]  # Shape: [b, 1, h, w] # Adjusted slicing

    # Calculate alpha and mask
    alpha = torch.exp(-1.0 * beta * sobel_mag)  # Shape: [b, 1, h, w]
    vis_mask_nchw = torch.greater(alpha, alpha_threshold).float()

    # Permute back to BHWC format: [b, h, w, 1]
    vis_mask = vis_mask_nchw.permute(0, 2, 3, 1)

    assert vis_mask.shape == disparity.shape  # Ensure output shape matches input
    return vis_mask


def sheet_warping(
    predictions, excluded_region_mask=None,
    connect_boundary_max_dist=0.5,
    connect_boundary_repeat_times=2,
    max_size=4096,
) -> o3d.geometry.TriangleMesh:
    r"""
    Convert depth predictions to a 3D mesh.
    Args:
        predictions: Dictionary containing:
            - "rgb": RGB image tensor of shape (H, W, 3) with
                values in [0, 255] (uint8) or [0, 1] (float).
            - "distance": Distance map tensor of shape (H, W).
            - "rays": Ray directions tensor of shape (H, W, 3).
        excluded_region_mask: Optional boolean mask tensor of shape (H, W).
        connect_boundary_max_dist: Maximum distance to bridge boundary vertices.
        connect_boundary_repeat_times: Number of iterations to repeat the boundary connection.
        max_size: Maximum size (height or width) to resize inputs to.
    Returns:
        An Open3D TriangleMesh object.
    """
    rgb = predictions["rgb"] / 255.0
    distance = predictions["distance"]
    rays = predictions["rays"]
    mesh = pano_sheet_warping(
        rgb, 
        distance, 
        rays, 
        excluded_region_mask, 
        connect_boundary_max_dist=connect_boundary_max_dist,
        connect_boundary_repeat_times=connect_boundary_repeat_times, 
        max_size=max_size
    )
    return mesh


def seed_all(seed: int = 0):
    r"""
    Set random seeds of all components.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def colorize_depth_maps(
    depth: np.ndarray, 
    mask: np.ndarray = None, 
    normalize: bool = True, 
    cmap: str = 'Spectral'
) -> np.ndarray:
    r"""
    Colorize depth maps using a colormap.
    Args:
        depth (np.ndarray): Depth map to colorize, shape (H, W).
        mask (np.ndarray, optional): Optional mask to apply to the depth map, shape (H, W).
        normalize (bool): Whether to normalize the depth values before colorization.
        cmap (str): Name of the colormap to use.
    Returns:
        np.ndarray: Colorized depth map, shape (H, W, 3).
    """
    # moge vis function
    if mask is None:
        depth = np.where(depth > 0, depth, np.nan)
    else:
        depth = np.where((depth > 0) & mask, depth, np.nan)

    # Convert depth to disparity (inverse of depth)
    disp = 1 / depth  # Closer objects have higher disparity values

    # Set invalid disparity values to the 0.1% quantile (avoids extreme outliers)
    if mask is not None:
        disp[~((depth > 0) & mask)] = np.nanquantile(disp, 0.001)

    # Normalize disparity values to [0,1] range if requested
    if normalize:
        min_disp, max_disp = np.nanquantile(
            disp, 0.001), np.nanquantile(disp, 0.99)
        disp = (disp - min_disp) / (max_disp - min_disp)
    # Apply colormap (inverted so closer=warmer colors)
    # Note: matplotlib colormaps return RGBA in [0,1] range
    colored = np.nan_to_num(
        matplotlib.colormaps[cmap](
            1.0 - disp)[..., :3],  # Invert and drop alpha
        nan=0  # Replace NaN with black
    )
    # Convert to uint8 and ensure contiguous memory layout
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))

    return colored
