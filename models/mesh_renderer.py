import torch
import torch.nn as nn
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
)
from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams
from pytorch3d.renderer.mesh.textures import TexturesVertex

from pytorch3d.structures import Meshes


class SimpleShader(nn.Module):
    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images  # (N, H, W, 3) RGBA image


class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs):
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        depth = fragments.zbuf[:, :, :, 0].unsqueeze(0)
        image = images.permute(0, 3, 1, 2)[:, :3, :, :]
        mask = images.permute(0, 3, 1, 2)[:, [3], :, :]
        closest_faces = fragments.pix_to_face[0, :, :, 0]
        return image, mask, depth, closest_faces, fragments


class Renderer:
    def __init__(self, config, image_size=512, antialiasing_factor=1):
        self.device = config["device"]
        fl = antialiasing_factor * config["init_focal_length"]
        principal_point = image_size / 2
        self.K = torch.tensor(
            [[fl, 0.0, principal_point], [0.0, fl, principal_point], [0.0, 0.0, 1.0]], device=self.device
        )
        self.image_size = image_size
        self.raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=config["blur_radius"],
            faces_per_pixel=5,
            bin_size=0,
        )
        self.renderer = MeshRendererWithDepth(
            rasterizer=MeshRasterizer(raster_settings=self.raster_settings),
            shader=SimpleShader(),
        )

    def sample_points(self, points_3d, triangles, colors, extrinsic):
        points_3d = torch.cat((points_3d, torch.ones(points_3d.shape[0], 1, device=points_3d.device)), dim=1).T
        points_3d_transformed = extrinsic @ points_3d

        cameras = PerspectiveCameras(
            device=self.device,
            R=torch.eye(3).unsqueeze(0),
            in_ndc=False,
            T=torch.zeros(1, 3),
            focal_length=-self.K.diag()[:2].unsqueeze(0),
            principal_point=self.K[:2, 2].unsqueeze(0),
            image_size=torch.ones(1, 2) * self.image_size,
        )

        mesh = Meshes(
            points_3d_transformed[:3].T.unsqueeze(0),
            triangles.unsqueeze(0),
            textures=TexturesVertex(verts_features=colors.unsqueeze(0).float()),
        )

        return self.renderer(mesh, cameras=cameras)

    @staticmethod
    def get_normals(vertices, triangles):
        triangles_coords = vertices[triangles]
        a = torch.nn.functional.normalize(triangles_coords[:, 1] - triangles_coords[:, 0], dim=-1)
        b = torch.nn.functional.normalize(triangles_coords[:, 2] - triangles_coords[:, 0], dim=-1)
        normals = torch.nn.functional.normalize(torch.cross(a, b, dim=-1), dim=-1)
        return normals

    @staticmethod
    def get_triangles_min_angle_degree(vertices, triangles):
        triangles_coords = vertices[triangles]
        a = (triangles_coords[:, 0, :] - triangles_coords[:, 1, :]).norm(dim=-1)
        b = (triangles_coords[:, 0, :] - triangles_coords[:, 2, :]).norm(dim=-1)
        c = (triangles_coords[:, 1, :] - triangles_coords[:, 2, :]).norm(dim=-1)
        gamma = torch.arccos((c ** 2 - a ** 2 - b ** 2) / (-2 * a * b))
        alpha = torch.arccos((a ** 2 - c ** 2 - b ** 2) / (-2 * c * b))
        beta = torch.arccos((b ** 2 - a ** 2 - c ** 2) / (-2 * a * c))
        return torch.stack((alpha, beta, gamma)).min(dim=0)[0] * 180 / torch.pi

    def unproject_points(
        self, depth, image, extrinsic, closest_boundary_points_data=None, starting_index=0, depth_boundaries_mask=None
    ):

        depth = depth[0, 0]
        image = image[0].permute(1, 2, 0)

        yss, xss = torch.where(depth != -1)
        indss = torch.linspace(
            starting_index, starting_index + yss.shape[0] - 1, yss.shape[0], device=depth.device, dtype=torch.long
        )
        im_mask = torch.ones((self.image_size, self.image_size), device=indss.device, dtype=torch.long) * -1
        im_mask[yss, xss] = indss
        if closest_boundary_points_data is not None:
            boundary_xs, boundary_ys, boundary_closest_points = (
                closest_boundary_points_data["xs"],
                closest_boundary_points_data["ys"],
                closest_boundary_points_data["closest_points"],
            )
            im_mask[boundary_xs, boundary_ys] = boundary_closest_points

        if depth_boundaries_mask is not None:
            depth_discontinuity_points = im_mask[depth_boundaries_mask[0, 0]]

        triangles1 = torch.stack((im_mask[1:, :-1], im_mask[1:, 1:], im_mask[:-1, 1:]))
        triangles2 = torch.stack((im_mask[:-1, :-1], im_mask[1:, :-1], im_mask[:-1, 1:]))
        triangles1 = triangles1[:, (triangles1 >= 0).all(dim=0)]
        triangles2 = triangles2[:, (triangles2 >= 0).all(dim=0)]
        triangles = torch.cat((triangles1, triangles2), dim=1).long().T

        z_points = depth[yss, xss]
        colors = image[yss, xss]

        x_points = ((xss + 0.5) / self.K[0, 0] - self.K[0, 2] / self.K[0, 0]) * (z_points.squeeze())
        y_points = ((yss + 0.5) / self.K[1, 1] - self.K[1, 2] / self.K[1, 1]) * (z_points.squeeze())

        points_3d = extrinsic.inverse() @ torch.stack((x_points, y_points, z_points, torch.ones_like(z_points)))

        return_dict = {
            "colors": colors,
            "points_3d": points_3d[:3].T,
            "triangles": triangles,
        }
        if depth_boundaries_mask is not None:
            return_dict["depth_discontinuity_points"] = depth_discontinuity_points

        return return_dict
