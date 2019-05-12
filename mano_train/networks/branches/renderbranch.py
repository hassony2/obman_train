import os

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as torch_f
import trimesh

from neural_renderer import Renderer
from handobjectdatasets.queries import TransQueries, BaseQueries

from mano_train.objectutils.objectio import load_obj


class RenderBranch(nn.Module):
    def __init__(self, obj_ico_divisions=3, mano_root='misc/mano', modes=['depth']):
        """
        Args:
            mano_root (path): dir containing mano pickle files
        """
        super(RenderBranch, self).__init__()
        intrinsic = torch.Tensor([[[600., 0., 160., 0.], [0., 600., 120., 0.],
                                   [0., 0., 1., 0.]]]).cuda()
        intrinsic = torch.Tensor([[[480., 0., 128., 0.], [0., -480., 128., 0.],
                                   [0., 0., 1., 0.]]]).cuda()
        intrinsic.requires_grad = False
        self.renderer = Renderer(orig_size=256, light=False, far=1000)
        _, faces_right = load_obj(os.path.join(mano_root, 'mano_right.obj'))
        _, faces_left = load_obj(os.path.join(mano_root, 'mano_left.obj'))
        self.hand_faces_right = torch.Tensor(faces_right).cuda().int()
        self.hand_faces_left = torch.Tensor(faces_left).cuda().int()
        self.hand_texture = torch.Tensor([1, 0, 0]).cuda().expand(
            1, self.hand_faces_right.shape[0], 2, 2, 2, 3)

        # Initialize objfaces
        test_mesh = trimesh.creation.icosphere(subdivisions=obj_ico_divisions)
        self.objfaces = torch.Tensor(
            np.array(test_mesh.faces)).int().cuda().unsqueeze(0)
        self.objtexture = torch.Tensor([0, 1, 0]).cuda().expand(
            1, self.objfaces.shape[1], 2, 2, 2, 3)
        self.modes = modes

    def forward(self,
                center3d=None,
                camintr=None,
                affinetrans=None,
                handverts3d=None,
                hand_sides=None,
                objverts3d=None):
        if handverts3d is not None:
            batch_size = handverts3d.shape[0]
        elif objverts3d is not None:
            batch_size = objverts3d.shape[0]

        results = {}
        verts = None
        faces = None
        textures = None
        if handverts3d is not None:
            # Uncenter hands and back to meters
            handverts3d = center3d.unsqueeze(1) / 1000 + handverts3d / 1000

            # Insure hand texture is of right size
            if self.hand_texture.shape[0]:
                self.hand_texture = self.hand_texture[0].unsqueeze(0).repeat(
                    batch_size, 1, 1, 1, 1, 1)

            # Get correct right and left hand faces
            is_rights = handverts3d.new_tensor(
                [side == 'right' for side in hand_sides]).byte()
            is_lefts = 1 - is_rights
            hand_faces = self.hand_faces_right.new_empty(
                (batch_size, self.hand_faces_right.shape[0],
                 self.hand_faces_right.shape[1]))
            if is_rights.sum():
                hand_faces[is_rights] = self.hand_faces_right
            if is_lefts.sum():
                hand_faces[is_lefts] = self.hand_faces_left

            verts = handverts3d
            faces = hand_faces
            textures = self.hand_texture

        if objverts3d is not None:
            # Uncenter objects and back to meters
            objverts3d = center3d.unsqueeze(1) / 1000 + objverts3d / 1000
            if self.objtexture.shape[0] != batch_size:
                self.objtexture = self.objtexture[0].unsqueeze(0).repeat(
                    batch_size, 1, 1, 1, 1, 1)
            if self.objfaces.shape[0] != batch_size:
                self.objfaces = self.objfaces[0].unsqueeze(0).repeat(
                    batch_size, 1, 1)

            if verts is None:
                verts = objverts3d
                faces = self.objfaces
                textures = self.objtexture
            else:
                faces = torch.cat(
                    [faces, self.objfaces + verts.shape[1]], dim=1)
                verts = torch.cat([verts, objverts3d], dim=1)
                textures = torch.cat([textures, self.objtexture], dim=1)

        render_flag = False
        # Render segmentation
        if 'segm' in self.modes:
            segm = self.renderer.forward(
                verts, faces, textures, K=camintr)
            segm = torch.flip(segm, [2])
            results['render_segms'] = segm
            render_flag = True

        if 'depth' in self.modes:
            # Render detph
            depth = self.renderer.forward(
                verts, faces, textures, K=camintr, mode='depth')
            # zero extreme depths

            # Countours of rendering have large values, rendering artifact ?
            depth[depth > 10] = 0
            # Flip along y axis
            depth = torch.flip(depth, [1])
            results['render_depth'] = depth
            render_flag = True
        if not render_flag:
            warnings.warn('No renderings produced although renderer initialized')
        return results


class RenderLoss():
    def __init__(self, lambda_render=1):
        self.lambda_render = lambda_render

    def compute_loss(self, preds, target):
        render_losses = {}
        final_loss = torch.Tensor([0]).cuda()
        loss_flag  = False  # Keep track of actually computing losses
        if (TransQueries.depth in target) and ('render_depth' in preds):
            depth_loss = torch_f.mse_loss(preds['render_depth'],
                                          target[TransQueries.depth])
            render_losses['render_depth_loss'] = depth_loss
            final_loss += depth_loss
            loss_flag = True
        if (TransQueries.segms in target) and self.lambda_render and (
                'render_segms' in preds):
            segm_loss = torch_f.mse_loss(preds['render_segms'],
                                         target[TransQueries.segms])
            render_losses['render_segm_loss'] = segm_loss
            final_loss += segm_loss
            loss_flag = True
        if not loss_flag:
            warnings.warn('No render loss computed although render loss active!')
        return final_loss, render_losses
