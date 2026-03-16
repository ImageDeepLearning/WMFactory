# Tencent HunyuanWorld-1.0 is licensed under TENCENT HUNYUANWORLD-1.0 COMMUNITY LICENSE AGREEMENT
# THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA AND 
# IS EXPRESSLY LIMITED TO THE TERRITORY, AS DEFINED BELOW.
# By clicking to agree or by using, reproducing, modifying, distributing, performing or displaying 
# any portion or element of the Tencent HunyuanWorld-1.0 Works, including via any Hosted Service, 
# You will be deemed to have recognized and accepted the content of this Agreement, 
# which is effective immediately.

# For avoidance of doubts, Tencent HunyuanWorld-1.0 means the 3D generation models 
# and their software and algorithms, including trained model weights, parameters (including 
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code, 
# fine-tuning enabling code and other elements of the foregoing made publicly available 
# by Tencent at [https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0].
import torch
from typing import List, Dict, Tuple


class AdaptiveDepthCompressor:
    r"""
    Adaptive depth compressor to solve the problem of excessive background depth variance
    in 3D world generation. This class provides methods to compress background and foreground
    depth values based on statistical analysis of depth distributions, with options for
    smooth compression and outlier removal.
    Args:
        cv_thresholds: Tuple of (low, high) thresholds for coefficient of variation (CV).
        compression_quantiles: Tuple of (low, medium, high) quantiles for depth compression.
        fg_bg_depth_margin: Margin factor to ensure foreground depth is greater than background.
        enable_smooth_compression: Whether to use smooth compression instead of hard truncation.
        outlier_removal_method: Method for outlier removal, options are "iqr", "quantile", or "none".
        min_compression_depth: Minimum depth threshold for compression to be applied.
    """

    def __init__(
        self,
        cv_thresholds: Tuple[float, float] = (0.3, 0.8),
        compression_quantiles: Tuple[float, float, float] = (0.95, 0.92, 0.85),
        fg_bg_depth_margin: float = 1.1,
        enable_smooth_compression: bool = True,
        outlier_removal_method: str = "iqr",
        min_compression_depth: float = 6.0,
    ):
        self.cv_thresholds = cv_thresholds
        self.compression_quantiles = compression_quantiles
        self.fg_bg_depth_margin = fg_bg_depth_margin
        self.enable_smooth_compression = enable_smooth_compression
        self.outlier_removal_method = outlier_removal_method
        self.min_compression_depth = min_compression_depth

    def _remove_outliers(self, depth_vals: torch.Tensor) -> torch.Tensor:
        r"""
        Remove outliers from depth values
        based on the specified method (IQR or quantile).
        Args:
            depth_vals: Tensor of depth values to process.
        Returns:
            Tensor of depth values with outliers removed.
        """
        if self.outlier_removal_method == "iqr":
            q25, q75 = torch.quantile(depth_vals, torch.tensor(
                [0.25, 0.75], device=depth_vals.device))
            iqr = q75 - q25
            lower_bound, upper_bound = q25 - 1.5 * iqr, q75 + 1.5 * iqr
            valid_mask = (depth_vals >= lower_bound) & (
                depth_vals <= upper_bound)
        elif self.outlier_removal_method == "quantile":
            q05, q95 = torch.quantile(depth_vals, torch.tensor(
                [0.05, 0.95], device=depth_vals.device))
            valid_mask = (depth_vals >= q05) & (depth_vals <= q95)
        else:
            return depth_vals
        return depth_vals[valid_mask] if valid_mask.sum() > 0 else depth_vals

    def _collect_foreground_depths(
        self, 
        layered_world_depth: List[Dict]
    ) -> List[torch.Tensor]:
        r"""
        Collect depth information of all foreground layers (remove outliers)
        from the layered world depth representation.
        Args:
            layered_world_depth: List of dictionaries containing depth information for each layer.
        Returns:
            List of tensors containing cleaned foreground depth values.
        """
        fg_depths = []
        for layer_depth in layered_world_depth:
            if layer_depth["name"] == "background":
                continue

            depth_vals = layer_depth["distance"]
            mask = layer_depth.get("mask", None)

            # Process the depth values within the mask area
            if mask is not None:
                if not isinstance(mask, torch.Tensor):
                    mask = torch.from_numpy(mask).to(depth_vals.device)
                depth_vals = depth_vals[mask.bool()]

            if depth_vals.numel() > 0:
                cleaned_depths = self._remove_outliers(depth_vals)
                if len(cleaned_depths) > 0:
                    fg_depths.append(cleaned_depths)
        return fg_depths

    def _get_pixelwise_foreground_max_depth(
        self, 
        layered_world_depth: List[Dict], 
        bg_shape: torch.Size, 
        bg_device: torch.device
    ) -> torch.Tensor:
        r"""
        Calculate the maximum foreground depth for each pixel position
        Args:
            layered_world_depth: List of dictionaries containing depth information for each layer.
            bg_shape: Shape of the background depth tensor.
            bg_device: Device where the background depth tensor is located.
        Returns:
            Tensor of maximum foreground depth values for each pixel position.
        """
        fg_max_depth = torch.zeros(bg_shape, device=bg_device)

        for layer_depth in layered_world_depth:
            if layer_depth["name"] == "background":
                continue

            layer_distance = layer_depth["distance"]
            layer_mask = layer_depth.get("mask", None)

            # Ensure that the tensor is on the correct device
            if not isinstance(layer_distance, torch.Tensor):
                layer_distance = torch.from_numpy(layer_distance).to(bg_device)
            else:
                layer_distance = layer_distance.to(bg_device)

            # Update the maximum depth of the foreground
            if layer_mask is not None:
                if not isinstance(layer_mask, torch.Tensor):
                    layer_mask = torch.from_numpy(layer_mask).to(bg_device)
                else:
                    layer_mask = layer_mask.to(bg_device)
                fg_max_depth = torch.where(layer_mask.bool(), torch.max(
                    fg_max_depth, layer_distance), fg_max_depth)
            else:
                fg_max_depth = torch.max(fg_max_depth, layer_distance)

        return fg_max_depth

    def _analyze_depth_distribution(self, bg_depth_distance: torch.Tensor) -> Dict:
        r"""
        Analyze the distribution characteristics of background depth
        Args:
            bg_depth_distance: Tensor of background depth distances.
        Returns:
            Dictionary containing statistical properties of the background depth distribution.
        """
        bg_mean, bg_std = torch.mean(
            bg_depth_distance), torch.std(bg_depth_distance)
        cv = bg_std / bg_mean

        quantiles = torch.quantile(bg_depth_distance,
                                   torch.tensor([0.5, 0.75, 0.9, 0.95, 0.99], device=bg_depth_distance.device))
        bg_q50, bg_q75, bg_q90, bg_q95, bg_q99 = quantiles

        return {"mean": bg_mean, "std": bg_std, "cv": cv, "q50": bg_q50,
                "q75": bg_q75, "q90": bg_q90, "q95": bg_q95, "q99": bg_q99}

    def _determine_compression_strategy(self, cv: float) -> Tuple[str, float]:
        r"""
        Determine compression strategy based on coefficient of variation
        Args:
            cv: Coefficient of variation of the background depth distribution.
        Returns:
            Tuple containing the compression strategy ("gentle", "standard", "aggressive")
            and the quantile to use for compression.
        """
        low_cv_threshold, high_cv_threshold = self.cv_thresholds
        low_var_quantile, medium_var_quantile, high_var_quantile = self.compression_quantiles

        if cv < low_cv_threshold:
            return "gentle", low_var_quantile
        elif cv > high_cv_threshold:
            return "aggressive", high_var_quantile
        else:
            return "standard", medium_var_quantile

    def _smooth_compression(self, depth_values: torch.Tensor, max_depth: torch.Tensor,
                            mask: torch.Tensor = None, transition_start_ratio: float = 0.95,
                            transition_range_ratio: float = 0.2, verbose: bool = False) -> torch.Tensor:
        r"""
        Use smooth compression function instead of hard truncation
        Args:
            depth_values: Tensor of depth values to compress.
            max_depth: Maximum depth value for compression.
            mask: Optional mask to apply compression only to certain pixels.
            transition_start_ratio: Ratio to determine the start of the transition range.
            transition_range_ratio: Ratio to determine the range of the transition.
            verbose: Whether to print detailed information about the compression process.
        Returns:
            Compressed depth values as a tensor.
        """
        if not self.enable_smooth_compression:
            compressed = depth_values.clone()
            if mask is not None:
                compressed[mask] = torch.clamp(
                    depth_values[mask], max=max_depth)
            else:
                compressed = torch.clamp(depth_values, max=max_depth)
            return compressed

        transition_start = max_depth * transition_start_ratio
        transition_range = max_depth * transition_range_ratio
        compressed_depth = depth_values.clone()

        mask_far = depth_values > transition_start
        if mask is not None:
            mask_far = mask_far & mask

        if mask_far.sum() > 0:
            far_depths = depth_values[mask_far]
            normalized = (far_depths - transition_start) / transition_range
            compressed_normalized = torch.sigmoid(
                normalized * 2 - 1) * 0.5 + 0.5
            compressed_far = transition_start + \
                compressed_normalized * (max_depth - transition_start)
            compressed_depth[mask_far] = compressed_far
            if verbose:
                print(
                    f"\t   Applied smooth compression to {mask_far.sum()} pixels beyond {transition_start:.2f}")
        elif verbose:
            print(f"\t   No compression needed, all depths within reasonable range")

        return compressed_depth

    def compress_background_depth(self, bg_depth_distance: torch.Tensor, layered_world_depth: List[Dict],
                                  bg_mask: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        r"""
        Adaptive compression of background depth values
        Args:
            bg_depth_distance: Tensor of background depth distances.
            layered_world_depth: List of dictionaries containing depth information for each layer.
            bg_mask: Tensor or numpy array representing the mask for background depth.
            verbose: Whether to print detailed information about the compression process.
        Returns:
            Compressed background depth values as a tensor.
        """
        if verbose:
            print(f"\t - Applying adaptive depth compression...")

        # Process mask
        if not isinstance(bg_mask, torch.Tensor):
            bg_mask = torch.from_numpy(bg_mask).to(bg_depth_distance.device)
        mask_bool = bg_mask.bool()
        masked_depths = bg_depth_distance[mask_bool]

        if masked_depths.numel() == 0:
            if verbose:
                print(f"\t   No valid depths in mask region, skipping compression")
            return bg_depth_distance

        # 1. Collect prospect depth information
        fg_depths = self._collect_foreground_depths(layered_world_depth)

        # 2. Calculate prospect depth statistics
        if fg_depths:
            all_fg_depths = torch.cat(fg_depths)
            fg_max = torch.quantile(all_fg_depths, torch.tensor(
                0.99, device=all_fg_depths.device))
            if verbose:
                print(
                    f"\t   Foreground depth stats - 99th percentile: {fg_max:.2f}")
        else:
            fg_max = torch.quantile(masked_depths, torch.tensor(
                0.5, device=masked_depths.device))
            if verbose:
                print(f"\t   No foreground found, using background stats for reference")

        # 3. Analyze the depth distribution of the background
        depth_stats = self._analyze_depth_distribution(masked_depths)
        if verbose:
            print(
                f"\t   Background depth stats - mean: {depth_stats['mean']:.2f}, \
                    std: {depth_stats['std']:.2f}, CV: {depth_stats['cv']:.3f}")

        # 4. Determine compression strategy
        strategy, compression_quantile = self._determine_compression_strategy(
            depth_stats['cv'])
        max_depth = torch.quantile(masked_depths, torch.tensor(
            compression_quantile, device=masked_depths.device))

        if verbose:
            print(
                f"\t   {strategy.capitalize()} compression strategy \
                    (CV={depth_stats['cv']:.3f}), quantile={compression_quantile}")

        # 5. Pixel level depth constraint
        if fg_depths:
            fg_max_depth_pixelwise = self._get_pixelwise_foreground_max_depth(
                layered_world_depth, bg_depth_distance.shape, bg_depth_distance.device)
            required_min_bg_depth = fg_max_depth_pixelwise * self.fg_bg_depth_margin
            pixelwise_violations = (
                bg_depth_distance < required_min_bg_depth) & mask_bool

            if pixelwise_violations.sum() > 0:
                violation_ratio = pixelwise_violations.float().sum() / mask_bool.float().sum()
                violated_required_depths = required_min_bg_depth[pixelwise_violations]
                pixelwise_min_depth = torch.quantile(violated_required_depths, torch.tensor(
                    0.99, device=violated_required_depths.device))
                max_depth = torch.max(max_depth, pixelwise_min_depth)
                if verbose:
                    print(
                        f"\t   Pixelwise constraint violation: {violation_ratio:.1%}, \
                            adjusted max depth to {max_depth:.2f}")
            elif verbose:
                print(f"\t   Pixelwise depth constraints satisfied")

        # 6. Global statistical constraints
        if fg_depths:
            min_bg_depth = fg_max * self.fg_bg_depth_margin
            max_depth = torch.max(max_depth, min_bg_depth)
            if verbose:
                print(f"\t   Final max depth: {max_depth:.2f}")

        # 6.5. Depth threshold check: If max_depth is less than the threshold, skip compression
        if max_depth < self.min_compression_depth:
            if verbose:
                print(
                    f"\t   Max depth {max_depth:.2f} is below threshold \
                        {self.min_compression_depth:.2f}, skipping compression")
            return bg_depth_distance

        # 7. Application compression
        compressed_depth = self._smooth_compression(
            bg_depth_distance, max_depth, mask_bool, 0.9, 0.2, verbose)

        # 8. Hard truncation of extreme outliers
        final_max = max_depth * 1.2
        outliers = (compressed_depth > final_max) & mask_bool
        if outliers.sum() > 0:
            compressed_depth[outliers] = final_max

        # 9. statistic
        compression_ratio = ((bg_depth_distance > max_depth)
                             & mask_bool).float().sum() / mask_bool.float().sum()
        if verbose:
            print(
                f"\t   Compression summary - max depth: \
                    {max_depth:.2f}, affected: {compression_ratio:.1%}")

        return compressed_depth

    def compress_foreground_depth(
        self, 
        fg_depth_distance: torch.Tensor, 
        fg_mask: torch.Tensor,
        verbose: bool = False, 
        conservative_ratio: float = 0.99,
        iqr_scale: float = 2
    ) -> torch.Tensor:
        r"""
        Conservatively compress outliers for foreground depth
        Args:
            fg_depth_distance: Tensor of foreground depth distances.
            fg_mask: Tensor or numpy array representing the mask for foreground depth.
            verbose: Whether to print detailed information about the compression process.
            conservative_ratio: Ratio to use for conservative compression.
            iqr_scale: Scale factor for IQR-based upper bound.
        Returns:
            Compressed foreground depth values as a tensor.
        """
        if verbose:
            print(f"\t - Applying conservative foreground depth compression...")

        # Process mask
        if not isinstance(fg_mask, torch.Tensor):
            fg_mask = torch.from_numpy(fg_mask).to(fg_depth_distance.device)
        mask_bool = fg_mask.bool()
        masked_depths = fg_depth_distance[mask_bool]

        if masked_depths.numel() == 0:
            if verbose:
                print(f"\t   No valid depths in mask region, skipping compression")
            return fg_depth_distance

        # Calculate statistical information
        depth_mean, depth_std = torch.mean(
            masked_depths), torch.std(masked_depths)

        # Determine the upper bound using IQR and quantile methods
        q25, q75 = torch.quantile(masked_depths, torch.tensor(
            [0.25, 0.75], device=masked_depths.device))
        iqr = q75 - q25
        upper_bound = q75 + iqr_scale * iqr
        conservative_max = torch.quantile(masked_depths, torch.tensor(
            conservative_ratio, device=masked_depths.device))
        final_max = torch.max(upper_bound, conservative_max)

        # Statistical Outliers
        outliers = (fg_depth_distance > final_max) & mask_bool
        outlier_count = outliers.sum().item()

        if verbose:
            print(
                f"\t   Depth stats - mean: {depth_mean:.2f}, std: {depth_std:.2f}")
            print(
                f"\t   IQR bounds - Q25: {q25:.2f}, Q75: {q75:.2f}, upper: {upper_bound:.2f}")
            print(
                f"\t   Conservative max: {conservative_max:.2f}, final max: {final_max:.2f}")
            print(
                f"\t   Outliers: {outlier_count} ({(outlier_count/masked_depths.numel()*100):.2f}%)")

        # Depth threshold check: If final_max is less than the threshold, skip compression
        if final_max < self.min_compression_depth:
            if verbose:
                print(
                    f"\t   Final max depth {final_max:.2f} is below threshold \
                        {self.min_compression_depth:.2f}, skipping compression")
            return fg_depth_distance

        # Apply compression
        if outlier_count > 0:
            compressed_depth = self._smooth_compression(
                fg_depth_distance, final_max, mask_bool, 0.99, 0.1, verbose)
        else:
            compressed_depth = fg_depth_distance.clone()

        return compressed_depth


def create_adaptive_depth_compressor(
    scene_type: str = "auto",
    enable_smooth_compression: bool = True,
    outlier_removal_method: str = "iqr",
    min_compression_depth: float = 6.0,  # Minimum compression depth threshold
) -> AdaptiveDepthCompressor:
    r"""
    Create adaptive depth compressors suitable for different scene types
    Args:
        scene_type: Scenario Type ("indoor", "outdoor", "mixed", "auto")
        enable_smooth_compression: enable smooth compression or not
        outlier_removal_method: Outlier removal method ("iqr", "quantile", "none")
    """
    common_params = {
        "enable_smooth_compression": enable_smooth_compression,
        "outlier_removal_method": outlier_removal_method,
        "min_compression_depth": min_compression_depth,
    }

    if scene_type == "indoor":
        # Indoor scene: Depth variation is relatively small, conservative compression is used
        return AdaptiveDepthCompressor(
            cv_thresholds=(0.2, 0.6),
            compression_quantiles=(1.0, 0.975, 0.95),
            fg_bg_depth_margin=1.05,
            **common_params
        )
    elif scene_type == "outdoor":
        # Outdoor scenes: There may be sky, distant mountains, etc., using more aggressive compression
        return AdaptiveDepthCompressor(
            cv_thresholds=(0.4, 1.0),
            compression_quantiles=(0.98, 0.955, 0.93),
            fg_bg_depth_margin=1.15,
            **common_params
        )
    elif scene_type == "mixed":
        # Mixed Scene: Balanced Settings
        return AdaptiveDepthCompressor(
            cv_thresholds=(0.3, 0.8),
            compression_quantiles=(0.99, 0.97, 0.95),
            fg_bg_depth_margin=1.1,
            **common_params
        )
    else:  # auto
        # Automatic mode: Use default settings
        return AdaptiveDepthCompressor(**common_params)
