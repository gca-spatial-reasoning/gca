import io
import os
from typing import Dict, List, Optional

import cv2
import matplotlib as mpl
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh
import torch
import torch.nn.functional as F

from tools.utils.misc import add_sys_path

third_party_dir = os.path.join(os.path.dirname(__file__), '..', 'third_party')
with add_sys_path(os.path.join(third_party_dir, 'vggt')):
    from visual_util import integrate_camera_into_scene


def add_label_to_image(image: Image.Image, label: str) -> Image.Image:
    font_size = max(20, int(image.height / 27))
    font = ImageFont.truetype(
        os.path.join(os.path.dirname(__file__), 'OpenSans-Bold.ttf'), 
        font_size
    )

    padding = int(font_size * 0.4)

    try:
        text_bbox = font.getbbox(label)
        text_height = text_bbox[3] - text_bbox[1]
    except AttributeError:
        _, text_height = font.getsize(label)

    title_height = text_height + padding * 2

    new_width = image.width
    new_height = image.height + title_height
    new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))

    draw = ImageDraw.Draw(new_image)
    draw.rectangle([(0, 0), (new_width, title_height)], fill=(220, 220, 220))

    text_position = (padding, padding)
    draw.text(text_position, label, fill=(0, 0, 0), font=font)
    new_image.paste(image, (0, title_height))
    
    return new_image


def fig2img(fig: Figure):
    with io.BytesIO() as buf:
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = Image.open(buf).copy()
        plt.close(fig)
    return img


def show_box(box, ax, color='red'):
    x0, y0 = box[0], box[1]
    w, h = box[2] - x0, box[3] - y0
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=2))


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)


def visualize_detection(
    image: Image.Image,
    boxes: torch.Tensor,
    labels: List[str],
    scores: Optional[torch.Tensor] = None,
) -> Image.Image:
    """
    Visualizes detection results with indexed and labeled bounding boxes.
    Each box is colored differently and includes its index, label, and score.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    num_boxes = len(boxes)
    colors = plt.cm.get_cmap('gist_rainbow', num_boxes) if num_boxes > 0 else []

    for i in range(num_boxes):
        box, label = boxes[i].cpu(), labels[i]
        color = colors(i)
        show_box(box, ax, color=color)

        text = f'{i}: {label}'
        x0, y0 = box[0], box[1]
        ax.text(
            x0, y0 - 5, text, 
            bbox=dict(facecolor=color, alpha=0.7), 
            fontsize=14, 
            color='white'
        )
    ax.axis('off')

    return fig2img(fig)


def visualize_segmentation(
    image: Image.Image,
    mask: torch.Tensor | List[torch.Tensor],
    prompt_box: Optional[torch.Tensor] = None,
) -> Image.Image:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    if prompt_box is not None:
        show_box(prompt_box.cpu(), ax)
    show_mask(mask.cpu().numpy(), ax, borders=True)
    ax.axis('off')

    return fig2img(fig)


def visualize_ocr(
    image: Image.Image,
    boxes: torch.Tensor,
    texts: List[str],
    scores: Optional[torch.Tensor] = None,
) -> Image.Image:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.axis('off')

    num_boxes = len(boxes)
    colors = plt.cm.get_cmap('gist_rainbow', num_boxes) if num_boxes > 0 else []

    for i in range(num_boxes):
        box, text = boxes[i].cpu(), texts[i]
        color = colors(i)
        show_box(box, ax, color=color)

        if scores is not None:
            score = scores[i].cpu().item()
            display_text = f'"{text}" ({score:.2f})'
        else:
            display_text = f'"{text}"'
        
        x0, y0 = box[0], box[1]
        ax.text(x0, y0 - 5, display_text, 
                bbox=dict(facecolor=color, alpha=0.8), 
                fontsize=14, color='white')

    return fig2img(fig)


def _plot_camera_frustum(ax: Axes3D, frustum: trimesh.Trimesh):
    for edge in frustum.edges_unique:
        # Get the 3D coordinates of the two vertices that form the edge
        p1, p2 = frustum.vertices[edge]
        color = (frustum.visual.face_colors[0, :3] / 255.).tolist()
        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
            color=color, 
            zorder=100
        )


def _plot_box_3d(ax: Axes3D, box: trimesh.primitives.Box):
    for edge in box.edges_unique:
        p1, p2 = box.vertices[edge]
        color = (box.visual.face_colors[0, :3] / 255.).tolist()
        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
            color=color,
            alpha=0.5, 
            linestyle='--',
            linewidth=1,
            zorder=100
        )


def _apply_scene_transform(
    scene: trimesh.Scene,
    transform: np.ndarray,
) -> trimesh.Scene:
    for geom_name, geom in scene.geometry.items():
        transformed_geom = geom.copy()
        transformed_geom.apply_transform(transform)
        scene.geometry[geom_name] = transformed_geom
    return scene


def _apply_scene_alignment_mpl(
    scene: trimesh.Scene, 
    extrinsics_matrices: np.ndarray
) -> trimesh.Scene:
    # Only apply the transform to make the first camera the origin.
    # We remove the OpenGL conversion and the 180-degree rotation.
    transform = np.linalg.inv(extrinsics_matrices[0])
    return _apply_scene_transform(scene, transform)


def visualize_3d_object(
    points: torch.Tensor,
    colors: torch.Tensor,
) -> Image.Image:
    # 1. --- Data Preparation ---
    points_np = points.squeeze().reshape(-1, 3).cpu().numpy()
    # Convert colors from [0, 1] range to [0, 255] for trimesh
    colors_np = colors.squeeze().reshape(-1, 3).cpu().numpy()

    # 2. --- Matplotlib Visualization ---
    data_range_x = points_np[:, 0].max() - points_np[:, 0].min()
    data_range_y = points_np[:, 1].max() - points_np[:, 1].min()
    aspect_ratio = data_range_x / (data_range_y + 1e-6)

    fig_height = 8
    fig_width = fig_height * (2 * aspect_ratio)
    fig_width = np.clip(fig_width, 10, 24)

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Place the subplot at the specified location in GridSpec.
    gs = gridspec.GridSpec(
        3, 3, 
        figure=fig, 
        width_ratios=[1, 1, 0.08], 
        height_ratios=[0.2, 0.6, 0.2]
    )
    ax1 = fig.add_subplot(gs[:, 0], projection='3d')
    ax2 = fig.add_subplot(gs[:, 1], projection='3d')
    cax = fig.add_subplot(gs[1, 2]) 

    # Left subplot: point cloud with RGB colors
    ax1.set_title('3D Points with RGB Color', fontsize=16)
    ax1.scatter(
        points_np[:, 0],
        points_np[:, 1],
        points_np[:, 2],
        c=colors_np,
        s=1.0
    )

    # Right subplot: Point cloud colored by height/depth
    ax2.set_title('3D Points Colored by Depth', fontsize=16)
    scatter = ax2.scatter(
        points_np[:, 0], 
        points_np[:, 1], 
        points_np[:, 2],
        c=points_np[:, 2], 
        cmap='viridis', 
        s=1.0
    )
    fig.colorbar(scatter, cax=cax, label='Depth')

    # Set labels, view, and ticks for both subplots
    for ax in [ax1, ax2]:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_aspect('equal', adjustable='box')
        # Set a default isometric view
        ax.view_init(elev=-85, azim=-90) 
        
        # Use MaxNLocator to optimize tick display and prevent overlapping
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))

    return fig2img(fig)


def visualize_obj_orientation(
    image: Image.Image,
    azimuth: float,
    polar: float,
    rotation: float,
) -> Image.Image:
    phi   = np.radians(azimuth)
    theta = np.radians(polar)
    gamma = rotation
    
    with add_sys_path(os.path.join(third_party_dir, 'Orient-Anything')):
        from utils import render_3D_axis, overlay_images_with_scaling

    render_axis = render_3D_axis(phi, theta, gamma)
    res_img = overlay_images_with_scaling(render_axis, image)
    return res_img
    

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def visualize_obj_pose(
    image: Image.Image,
    points_3d: torch.Tensor,
    image_tensor: torch.Tensor,
    obb: List[torch.Tensor],
    axes: Dict[str, torch.Tensor],
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
    text_label: str,
    show_cam: bool = True
):
    # 1. --- Data Preparation ---
    centroid, _, corners = [d.cpu().numpy() for d in obb]
    axes = {key: value.cpu().numpy() for key, value in axes.items()}
    extrinsic = extrinsic.cpu().numpy()
    intrinsic = intrinsic.cpu().numpy()

    points_3d = points_3d.cpu().numpy().reshape(-1, 3)
    image_tensor = image_tensor.cpu().numpy()
    if image_tensor.ndim == 4 and image_tensor.shape[1] == 3:  # NCHW format
        colors_rgb = np.transpose(image_tensor, (0, 2, 3, 1))
    else:  # Assume already in NHWC format
        colors_rgb = image_tensor
    colors_rgb = colors_rgb.reshape(-1, 3)

    # 2. --- Project Corners for OBB visualization ---
    corners_world_h = np.hstack([corners, np.ones((8, 1))])
    corners_cam = (extrinsic @ corners_world_h.T).T
    valid_corners_mask = corners_cam[:, 2] > 1e-6
    corners_img = np.empty((0, 2))
    if np.any(valid_corners_mask):
        corners_img_h = (intrinsic @ corners_cam[:, :3].T).T
        corners_img = corners_img_h[valid_corners_mask, :2] / corners_img_h[valid_corners_mask, 2, np.newaxis]

    # 3. --- Matplotlib Visualization ---
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f'Pose Analysis for {text_label}', fontsize=16)

    # 3.1 --- Left Panel: 2D Camera View ---
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image)
    ax1.set_title('2D Camera Perspective')
    ax1.axis('off')

    # 3.1.1 --- OBB Visualization ---
    if corners_img.shape[0] > 0:
        edges = [
            (0, 1), (0, 2), (0, 3), (1, 4), 
            (2, 4), (3, 5), (1, 5), (2, 6),
            (3, 6), (4, 7), (5, 7), (6, 7),
        ]

        for i, j in edges:
            if valid_corners_mask[i] and valid_corners_mask[j]:
                idx_i = np.where(np.where(valid_corners_mask)[0] == i)[0][0]
                idx_j = np.where(np.where(valid_corners_mask)[0] == j)[0][0]
                ax1.plot(
                    [corners_img[idx_i, 0], corners_img[idx_j, 0]],
                    [corners_img[idx_i, 1], corners_img[idx_j, 1]],
                    c='cyan', lw=2, ls='--', zorder=5
                )

    # 3.1.2 --- Axes Visualization ---
    arrow_length_3d = 0.2
    centroid_world_h = np.append(centroid, 1)
    centroid_cam = extrinsic @ centroid_world_h
    legends = []
    for i, (name, vec) in enumerate(axes.items()):
        arrow_end_world_h = np.append(centroid + vec * arrow_length_3d, 1)
        arrow_end_cam = extrinsic @ arrow_end_world_h
        if centroid_cam[2] < 1e-6 or arrow_end_cam[2] < 1e-6: 
            continue

        p_start_h = intrinsic @ centroid_cam[:3]
        p_end_h = intrinsic @ arrow_end_cam[:3]
        p_start = p_start_h[:2] / p_start_h[2]
        p_end = p_end_h[:2] / p_end_h[2]

        min_pixel_length = 60
        arrow_vec_2d = p_end - p_start
        current_pixel_length = np.linalg.norm(arrow_vec_2d)
        if 0 < current_pixel_length < min_pixel_length:
            arrow_vec_2d = (arrow_vec_2d / current_pixel_length) * min_pixel_length
            p_end = p_start + arrow_vec_2d

        color = ['red', 'green', 'blue'][i]
        ax1.arrow(
            p_start[0], p_start[1], p_end[0] - p_start[0], p_end[1] - p_start[1],
            head_width=15, 
            head_length=15, 
            linewidth=3, 
            fc=color, 
            ec=color, 
            length_includes_head=True, zorder=90,
        )

        arrow_vec_2d = p_end - p_start
        norm = np.linalg.norm(arrow_vec_2d)
        if norm > 1e-6:
            unit_arrow_vec_2d = arrow_vec_2d / norm
        else:
            unit_arrow_vec_2d = np.array([0.707, -0.707])

        axis_label_pos = p_end + 25 * unit_arrow_vec_2d
        ax1.text(
            axis_label_pos[0], axis_label_pos[1], name, color='white',
            fontsize=14,
            bbox=dict(facecolor=color, alpha=0.9, pad=1), 
            ha='center', va='center', zorder=100,
        )

        if '+Y' in name:
            legends.append(Line2D([0], [0], color=color, lw=2, label='+Y (points object\'s bottom)'))
        if '+Z' in name:
            legends.append(Line2D([0], [0], color=color, lw=2, label='+Z (points object\'s front)'))
        if '+X' in name:
            legends.append(Line2D([0], [0], color=color, lw=2, label='+X'))
    if legends:
        ax1.legend(handles=legends, loc='upper right', bbox_to_anchor=(0.95, 0.95))

    # 3.1.3 --- Text Label Visualization ---
    p_label_anchor = None
    img_h, img_w = image.height, image.width
    visible_corners_mask = (
        (corners_img[:, 0] >= 0) & 
        (corners_img[:, 0] < img_w) & 
        (corners_img[:, 1] >= 0) & 
        (corners_img[:, 1] < img_h)
    )
    visible_corners = corners_img[visible_corners_mask]

    if visible_corners.shape[0] > 0:
        top_corner_idx = np.argmin(visible_corners[:, 1])
        p_label_anchor = visible_corners[top_corner_idx]
    elif centroid_cam[2] > 1e-6:
        p_centroid_h = intrinsic @ centroid_cam[:3]
        p_label_anchor = p_centroid_h[:2] / p_centroid_h[2]

    if p_label_anchor is not None:
        ax1.text(
            p_label_anchor[0], p_label_anchor[1] - 20, text_label, 
            color='black', fontsize=14,
            ha='center', va='bottom', zorder=15,
            bbox=dict(facecolor='yellow', alpha=0.8, pad=3)
        )

    # 3.2 --- Right Panel: 3D Bird's-Eye View ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_title("3D Bird's-Eye View")

    # 3.2.1 --- Point Cloud Visualization ---
    ax2.scatter(
        points_3d[:, 0], 
        points_3d[:, 1], 
        points_3d[:, 2], 
        c=colors_rgb[:, :3], 
        s=1.0, 
        alpha=0.8,
    )

    # 3.2.2 --- 3D OBB Visualization ---
    legend_handles = []
    for i, j in edges:
        p1, p2 = corners[i], corners[j]
        ax2.plot(
            [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
            c='cyan', lw=1.5, ls='--', zorder=10,
        )
    legend_handles.append(
        Line2D([0], [0], color='cyan', lw=2, linestyle='--', label='OBB')
    )
    
    # 3.2.3 --- 3D Axes Visualization ---
    arrow_length_3d = 0.4
    artist_on_top = []
    for i, (name, vec) in enumerate(axes.items()):
        color = ['red', 'green', 'blue'][i]
        arrow = Arrow3D(
            [centroid[0], centroid[0] + vec[0] * arrow_length_3d],
            [centroid[1], centroid[1] + vec[1] * arrow_length_3d],
            [centroid[2], centroid[2] + vec[2] * arrow_length_3d],
            mutation_scale=20, lw=3, arrowstyle="-|>", color=color,
            zorder=100,
        )
        arrow.set_animated(True)
        ax2.add_artist(arrow)
        artist_on_top.append(arrow)

        text_pos = np.array([
            centroid[0] + vec[0] * arrow_length_3d * 1.2,
            centroid[1] + vec[1] * arrow_length_3d * 1.2,
            centroid[2] + vec[2] * arrow_length_3d * 1.2,
        ])
        if np.linalg.norm(vec[np.array([0, 2])]) < 0.2:  # If X and Z components are small
            other_vecs = [v for j, v in enumerate(list(axes.values())) if i != j]
            offset_dir = -(other_vecs[0] + other_vecs[1])
            offset_dir /= np.linalg.norm(offset_dir)
            text_pos += offset_dir * arrow_length_3d * 0.35

        text_artist = ax2.text(
            text_pos[0], text_pos[1], text_pos[2], name.upper(), 
            color='white', fontsize=14, ha='center', zorder=125,
            bbox=dict(facecolor=color, alpha=0.9, pad=1), 
        )
        text_artist.set_animated(True)
        artist_on_top.append(text_artist)
    
    # 3.2.4 --- 3D Camera Visualization ---
    if show_cam:
        scene_3d = trimesh.Scene()
        if points_3d.size == 0:
            points_3d = np.array([[1, 0, 0]])
            colors_rgb = np.array([[255, 255, 255]])
            scene_scale = 1.0
        else:
            lower_percentile = np.percentile(points_3d, 5, axis=0)
            upper_percentile = np.percentile(points_3d, 95, axis=0)
            scene_scale = np.linalg.norm(upper_percentile - lower_percentile)

        color = (255, 165, 0)  # orange
        camera_to_world = np.linalg.inv(extrinsic)
        integrate_camera_into_scene(
            scene_3d,
            camera_to_world,
            color,
            scene_scale
        )
        for _, geom in scene_3d.geometry.items():
            _plot_camera_frustum(ax2, geom)

        legend_handles.append(
            Line2D([0], [0], color=tuple(float(x / 255.) for x in color), lw=2, label='Camera')
        )

    ax2.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(0.95, 0.95))
    ax2.view_init(elev=0, azim=-90)
    ax2.set_aspect('equal')

    def draw_top_artist(event):
        for artist in artist_on_top:
            ax2.draw_artist(artist)
    
    fig.canvas.mpl_connect('draw_event', draw_top_artist)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig2img(fig)


def visualize_optical_flow(
    image: Image.Image,
    flow: torch.Tensor,
    step: int = 16,
) -> Image.Image:
    # 1. --- Data Preparation ---
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].astype(int)

    flow = flow.cpu().numpy()
    fx, fy = flow[y, x].T

    # Calculate mean flow
    mean_flow = np.mean(flow, axis=(0, 1))
    mean_dx, mean_dy = mean_flow

    # 2. --- Matplotlib Visualization ---
    fig, ax = plt.subplots(figsize=(12, 12 * h / w))
    ax.imshow(img_np)

    # Plot local flow vectors (as before)
    magnitude = np.sqrt(fx**2 + fy**2)
    ax.quiver(x, y, fx, fy, magnitude, cmap='viridis', angles='xy', scale_units='xy', scale=0.5, alpha=0.8)
    
    ax.set_title(f'Pixel Avg Motion: (dx={mean_dx:.2f}, dy={mean_dy:.2f})')
    ax.axis('off')

    return fig2img(fig)


def visualize_3d_scene(
    points: torch.Tensor,
    points_conf: torch.Tensor,
    image_tensor: torch.Tensor,
    camera_extrinsics: Optional[torch.Tensor],
    conf_thres: float = 20.0,
    show_cam: bool = True,
    mask_white_bg: bool = True,
    mask_black_bg: bool = True,
) -> Image.Image:
    # 1. --- Data Preparation ---
    vertices_3d = points.cpu().numpy().reshape(-1, 3)
    images = image_tensor.cpu().numpy()
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:  # Assume already in NHWC format
        colors_rgb = images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    conf = points_conf.cpu().numpy().reshape(-1)
    # Convert percentage threshold to actual confidence value
    if conf_thres == 0.0:
        conf_threshold = 0.0
    else:
        conf_threshold = np.percentile(conf, conf_thres)
    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)
    
    if mask_black_bg:
        black_bg_mask = colors_rgb.sum(axis=1) >= 16
        conf_mask = conf_mask & black_bg_mask

    if mask_white_bg:
        white_bg_mask = ~((colors_rgb > 240).all(axis=1))
        conf_mask = conf_mask & white_bg_mask

    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]
    
    if vertices_3d is None or np.asarray(vertices_3d).size == 0:
        vertices_3d = np.array([[1, 0, 0]])
        colors_rgb = np.array([[255, 255, 255]])
        scene_scale = 1
    else:
        # Calculate the 5th and 95th percentiles along each axis
        lower_percentile = np.percentile(vertices_3d, 5, axis=0)
        upper_percentile = np.percentile(vertices_3d, 95, axis=0)

        # Calculate the diagonal length of the percentile bounding box
        scene_scale = np.linalg.norm(upper_percentile - lower_percentile)

    # 2. --- Create Trimesh Scene ---
    scene_3d = trimesh.Scene()

    # Add point cloud data to the scene
    point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)
    scene_3d.add_geometry(point_cloud_data)

    # Prepare 4x4 matrices for camera extrinsics
    extrinsics_matrices = camera_extrinsics.cpu().numpy()
    num_cameras = len(camera_extrinsics)

    cam_legend_handles = []
    if show_cam:
        colormap = mpl.colormaps.get_cmap('gist_rainbow')
        
        # Add camera models to the scene
        for i in range(num_cameras):
            world_to_camera = extrinsics_matrices[i]
            camera_to_world = np.linalg.inv(world_to_camera)
            rgba_color = colormap(i / num_cameras)
            current_color = tuple(int(255 * x) for x in rgba_color[:3])

            integrate_camera_into_scene(scene_3d, camera_to_world, current_color, scene_scale)
            cam_legend_handles.append(
                Line2D([0], [0], color=tuple(x for x in rgba_color[:3]), lw=2, label=f'Camera {i}')
            )

    scene_3d = _apply_scene_alignment_mpl(scene_3d, extrinsics_matrices)

    # 3. --- Matplotlib Visualization ---
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 8), facecolor='white', subplot_kw={'projection': '3d'}
    )
    axes = axes.flatten()
    fig.suptitle('3D Reconstruction', fontsize=20)

    # Extract point cloud and camera frustum
    pcd = scene_3d.geometry['geometry_0']
    cameras = [scene_3d.geometry[f'geometry_{i + 1}'] for i in range(num_cameras)]

    # Plot 6 degree multi-view perspective
    titles = ['Camera 0\'s Perspective', 'Bird\'s Eye View (BEV)\nRelative to Camera 0']
    views = [(-85, -90), (0, -90)]
    locator = [(5, 5, 3), (5, 3, 5)]
    for i, ax in enumerate(axes):
        ax.scatter(
            pcd.vertices[:, 0],
            pcd.vertices[:, 1],
            pcd.vertices[:, 2],
            c=pcd.colors[:, :3] / 255.0, # Use colors from trimesh object
            s=1.0,
            zorder=1
        )

        if show_cam:
            for cam in cameras: 
                _plot_camera_frustum(ax, cam)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.xaxis.set_major_locator(MaxNLocator(nbins=locator[i][0], prune='both'))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=locator[i][1], prune='both'))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=locator[i][2], prune='both'))
        
        # Align camera0 (elev=-90) with a slight downward angle
        ax.view_init(elev=views[i][0], azim=views[i][1])
        ax.set_title(titles[i], fontsize=14)

    if cam_legend_handles:
        fig.legend(handles=cam_legend_handles, loc='upper right', bbox_to_anchor=(0.95, 0.95))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig2img(fig)


def visualize_aligned_3d_scene(
    ref_points: torch.Tensor,
    ref_conf: torch.Tensor,
    ref_extrinsic: torch.Tensor,
    ref_colors: torch.Tensor,
    ref_static_pts: List[torch.Tensor],
    src_points: torch.Tensor,
    src_conf: torch.Tensor,
    src_extrinsic: torch.Tensor,
    src_colors: torch.Tensor,
    src_static_pts: List[torch.Tensor],
    transform_matrix: torch.Tensor,
    conf_thres: float = 20.0,
    show_cam: bool = True,
    show_static_boxes: bool = True,
) -> Image.Image:
    # 1. --- Data Preparation ---
    ref_points = ref_points.cpu().reshape(-1, 3)
    if ref_colors.ndim == 4 and ref_colors.shape[1] == 3:
        ref_colors = ref_colors.permute(0, 2, 3, 1)
    ref_colors = ref_colors.cpu().reshape(-1, 3)
    
    src_points = src_points.reshape(-1, 3)
    if src_colors.ndim == 4 and src_colors.shape[1] == 3:
        src_colors = src_colors.permute(0, 2, 3, 1)
    src_colors = src_colors.cpu().reshape(-1, 3)

    ref_conf = ref_conf.cpu().reshape(-1)
    ref_conf_mask = ref_conf > np.percentile(ref_conf.numpy(), conf_thres)
    src_conf = src_conf.cpu().reshape(-1)
    src_conf_mask = src_conf > np.percentile(src_conf.numpy(), conf_thres)

    ref_extrinsic = ref_extrinsic.cpu()
    src_extrinsic = src_extrinsic.cpu()

    ref_static_pts = [pts.cpu().numpy() for pts in ref_static_pts]
    src_static_pts = [pts.cpu().numpy() for pts in src_static_pts]

    # 2. --- Align and combine point clouds ---
    p1, c1 = ref_points[ref_conf_mask], ref_colors[ref_conf_mask]
    p2, c2 = src_points[src_conf_mask], src_colors[src_conf_mask]
    transform_matrix = transform_matrix.cpu().float()

    p2_h = F.pad(p2, (0, 1), mode='constant', value=1.0)
    p2_aligned_h = (transform_matrix @ p2_h.T).T
    p2_aligned = p2_aligned_h[:, :3] / p2_aligned_h[:, 3, None]
    vertices_3d = torch.cat([p1, p2_aligned], dim=0).cpu().numpy()
    colors_rgb = torch.cat([c1, c2], dim=0).cpu().numpy()

    if vertices_3d.size == 0:
        vertices_3d = np.array([[1, 0, 0]])
        colors_rgb = np.array([[255, 255, 255]])
        scene_scale = 1.0
    else:
        lower_percentile = np.percentile(vertices_3d, 5, axis=0)
        upper_percentile = np.percentile(vertices_3d, 95, axis=0)
        scene_scale = np.linalg.norm(upper_percentile - lower_percentile)

    p1_static_sample = np.concatenate(ref_static_pts).reshape(-1, 3)
    p2_static_sample = np.concatenate(src_static_pts).reshape(-1, 3)
    p2_static_h = F.pad(torch.from_numpy(p2_static_sample.copy()).float(), (0, 1), mode='constant', value=1.0)
    p2_aligned_static_h = (transform_matrix @ p2_static_h.T).T
    p2_aligned_static_sample = (p2_aligned_static_h[:, :3] / p2_aligned_static_h[:, 3, None]).numpy()

    # 3. --- Create Trimesh Scene ---
    scene_3d = trimesh.Scene()

    colormap = mpl.colormaps.get_cmap('gist_rainbow')
    total_legend = 0
    if show_cam:
        total_legend += 2
    if show_static_boxes:
        total_legend += len(ref_static_pts)

    legend_handles = []
    if show_cam:
        ref_cam_ext = ref_extrinsic[0]
        ref_cam_pose = torch.inverse(ref_cam_ext)

        rgba_color = colormap(0)
        integrate_camera_into_scene(
            scene_3d, 
            ref_cam_pose.numpy(), 
            tuple(int(255 * x) for x in rgba_color[:3]), 
            scene_scale
        )
        legend_handles.append(
            Line2D([0], [0], color=tuple(x for x in rgba_color[:3]), lw=2, label='Reference Cam')
        )

        src_cam_ext = src_extrinsic[0]
        src_cam_pose_local = torch.inverse(src_cam_ext)
        aligned_src_cam_pose = transform_matrix @ src_cam_pose_local

        rgba_color = colormap(1 / total_legend)
        integrate_camera_into_scene(
            scene_3d, 
            aligned_src_cam_pose.numpy(), 
            tuple(int(255 * x) for x in rgba_color[:3]), 
            scene_scale
        )
        legend_handles.append(
            Line2D([0], [0], color=tuple(x for x in rgba_color[:3]), lw=2, label='Source Cam')
        )
    
    if show_static_boxes:
        for i, (ref_pts_i, src_pts_i) in enumerate(zip(ref_static_pts, src_static_pts)):
            src_pts_torch = torch.from_numpy(src_pts_i.copy())
            src_pts_h = F.pad(src_pts_torch, (0, 1), mode='constant', value=1.0)
            src_aligned_h = (transform_matrix @ src_pts_h.T).T
            src_aligned_pts_i = (src_aligned_h[:, :3] / src_aligned_h[:, 3, None]).numpy()

            combined_static_pts = np.concatenate([ref_pts_i, src_aligned_pts_i], axis=0)
            if combined_static_pts.shape[0] > 0:
                bbox = trimesh.PointCloud(combined_static_pts).bounding_box
                rgba_color = colormap((i + 2) / total_legend)
                bbox.visual.face_colors[:, :3] = tuple(int(255 * x) for x in rgba_color[:3])
                scene_3d.add_geometry(bbox, geom_name=f'box_{i}')
                legend_handles.append(
                    Line2D([0], [0], color=tuple(x for x in rgba_color[:3]), lw=2, label=f'Static {i}')
                )
    
    # --- 4. Matplotlib Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 8), subplot_kw={'projection': '3d'})
    axes = axes.flatten()
    fig.suptitle('Alignment', fontsize=20)

    # Left and middle subplots: Aligned scenes in camera 0's perspective and BEV
    titles = ['Camera 0\'s Perspective', 'Bird\'s Eye View (BEV)\nRelative to Camera 0']
    views = [(-85, -90), (0, -90)]
    locator = [(5, 5, 3), (5, 3, 5)]
    for i in range(2):
        ax = axes[i]

        # Plot point cloud with color
        ax.scatter(
            vertices_3d[:, 0], 
            vertices_3d[:, 1], 
            vertices_3d[:, 2],
            c=colors_rgb, 
            s=1.0, 
            zorder=1
        )

        # Plot cameras and bboxes from trimesh scene
        for geom_name, geom in scene_3d.geometry.items():
            if 'box' in geom_name:
                _plot_box_3d(ax, geom)
            else:
                _plot_camera_frustum(ax, geom)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=locator[i][0], prune='both'))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=locator[i][1], prune='both'))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=locator[i][2], prune='both'))

        ax.view_init(elev=views[i][0], azim=views[i][1])
        ax.set_title(titles[i], fontsize=14)
    
    if total_legend > 0:
        axes[1].legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(0.95, 0.95))

    # Right subplot: Static objects
    axes[-1].scatter(
        p1_static_sample[:, 0], 
        p1_static_sample[:, 1], 
        p1_static_sample[:, 2],
        c='blue', 
        s=1.5, 
        alpha=0.6, 
        label='Reference'
    )
    axes[-1].scatter(
        p2_aligned_static_sample[:, 0], 
        p2_aligned_static_sample[:, 1], 
        p2_aligned_static_sample[:, 2],
        c='red', 
        s=1.5, 
        alpha=0.6, 
        label='Source Aligned'
    )
    axes[-1].legend()
    axes[-1].set_xlabel('X')
    axes[-1].set_ylabel('Y')
    axes[-1].set_zlabel('Z')
    axes[-1].xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
    axes[-1].yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
    axes[-1].zaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))

    axes[-1].set_title('Provided Static Objects after Alignment')
    axes[-1].view_init(elev=-85, azim=-90)
    
    fig.tight_layout()

    return fig2img(fig)


def visualize_alignment(
    points1: torch.Tensor,  # (H, W, 3) or (N, 3)
    points2: torch.Tensor, 
    transform_matrix: torch.Tensor,
    points1_conf: torch.Tensor = None,
    points2_conf: torch.Tensor = None,
    sample_size: int = 10000,
) -> Image.Image:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    quantile = 0.8  
    if points1_conf is not None:
        threshold1 = torch.quantile(points1_conf.float(), quantile)
        valid_mask1 = points1_conf > threshold1
        points1 = points1[valid_mask1]
    if points2_conf is not None:
        threshold2 = torch.quantile(points2_conf.float(), quantile)
        valid_mask2 = points2_conf > threshold2
        points2 = points2[valid_mask2]

    p1_sample = points1.reshape(-1, 3)
    p2_sample = points2.reshape(-1, 3)

    if sample_size > 0 and p1_sample.shape[0] > sample_size:
        p1_sample = p1_sample[torch.randperm(p1_sample.shape[0])[:sample_size]]
    if sample_size > 0 and p2_sample.shape[0] > sample_size:
        p2_sample = p2_sample[torch.randperm(p2_sample.shape[0])[:sample_size]]

    p2_h = F.pad(p2_sample, (0, 1), mode='constant', value=1.0)
    p2_aligned_h = (transform_matrix.float() @ p2_h.T).T
    p2_aligned_sample = p2_aligned_h[:, :3] / p2_aligned_h[:, 3, None]

    p1_np = p1_sample.cpu().numpy()
    p2_aligned_np = p2_aligned_sample.cpu().numpy()

    # Camera X -> Plot X
    # Camera Z -> Plot Y
    # Camera -Y -> Plot Z (to make it upright)
    ax.scatter(
        p1_np[:, 0],
        p1_np[:, 2],
        -p1_np[:, 1],
        c='blue',
        s=0.5,
        alpha=0.4,
        label='Scene 1 (Reference)'
    )
    ax.scatter(
        p2_aligned_np[:, 0],
        p2_aligned_np[:, 2],
        -p2_aligned_np[:, 1],
        c='red',
        s=0.5,
        alpha=0.4,
        label='Scene 2 (Aligned)'
    )

    ax.set_xlabel('X Axis (Right)')
    ax.set_ylabel('Y Axis (Depth)')
    ax.set_zlabel('Z Axis (Height)')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    return fig2img(fig)
