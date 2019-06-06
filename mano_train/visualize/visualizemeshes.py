import pickle
import pythreejs as p3js
import numpy as np

from matplotlib import pyplot as plt


def hand_obj_children(
    obj_verts=None,
    obj_faces=None,
    gt_obj_verts=None,
    gt_obj_faces=None,
    hand_verts=None,
    mano_faces_left=None,
    display_wireframe=False,
    inside_face_colors=True,
    hand_opacity=1,
    obj_opacity=0.2,
):
    """Args:
        obj_verts(numpy.ndarray): vertices of object
        hand_verts(numpy.ndarray): vertices of handect
        *_faces(numpy.ndarray): faces
    """

    scene_children = []
    if obj_verts is not None:
        geo_obj = p3js.Geometry(vertices=obj_verts.tolist(), faces=obj_faces.tolist())
        geo_obj.exec_three_obj_method("computeFaceNormals")
        mat = p3js.MeshLambertMaterial(color="red", side="FrontSide", transparent=True)
        mat.opacity = obj_opacity  # obj_opacity
        surf_obj = p3js.Mesh(geometry=geo_obj, material=mat)
        if inside_face_colors:
            back_color = "#a91818"
        else:
            back_color = "red"
        mat_bak = p3js.MeshLambertMaterial(
            color=back_color, side="BackSide", transparent=True
        )
        mat_bak.opacity = obj_opacity
        surf_obj_back = p3js.Mesh(geometry=geo_obj, material=mat_bak)
        scene_children.append(surf_obj)
        scene_children.append(surf_obj_back)
        if display_wireframe:
            obj_edges = p3js.Mesh(
                geometry=geo_obj,
                material=p3js.MeshBasicMaterial(color="black", wireframe=True),
            )
            scene_children.append(obj_edges)
    if gt_obj_verts is not None:
        geo_obj = p3js.Geometry(
            vertices=gt_obj_verts.tolist(), faces=gt_obj_faces.tolist()
        )
        geo_obj.exec_three_obj_method("computeFaceNormals")
        mat = p3js.MeshLambertMaterial(
            color="orange", side="FrontSide", transparent=True
        )
        mat.opacity = obj_opacity
        surf_obj = p3js.Mesh(geometry=geo_obj, material=mat)
        mat_back = p3js.MeshLambertMaterial(
            color="#a91818", side="BackSide", transparent=True
        )
        mat_back.opacity = obj_opacity
        surf_obj_back = p3js.Mesh(geometry=geo_obj, material=mat_bak)
        scene_children.append(surf_obj)
        scene_children.append(surf_obj_back)
        if display_wireframe:
            obj_edges = p3js.Mesh(
                geometry=geo_obj,
                material=p3js.MeshBasicMaterial(color="black", wireframe=True),
            )
            scene_children.append(obj_edges)
    if hand_verts is not None:
        geo_hand = p3js.Geometry(
            vertices=hand_verts.tolist(), faces=mano_faces_left.tolist()
        )
        geo_hand.exec_three_obj_method("computeFaceNormals")
        mat = p3js.MeshLambertMaterial(color="blue", side="FrontSide", transparent=True)
        mat.opacity = hand_opacity
        surf_hand = p3js.Mesh(geometry=geo_hand, material=mat)
        bak_mat = p3js.MeshLambertMaterial(
            color="blue", side="BackSide", transparent=True
        )
        bak_mat.opacity = hand_opacity
        surf_hand_bak = p3js.Mesh(geometry=geo_hand, material=bak_mat)

        scene_children.append(surf_hand)
        scene_children.append(surf_hand_bak)
        if display_wireframe:
            hand_edges = p3js.Mesh(
                geometry=geo_hand,
                material=p3js.MeshBasicMaterial(color="black", wireframe=True),
            )
            scene_children.append(hand_edges)

    return scene_children


def scatter_children(points, color="red", size=4):
    scene_children = []
    geometry_point = p3js.BufferGeometry(
        attributes={"position": p3js.BufferAttribute(array=points)}
    )
    material_point = p3js.PointsMaterial(color=color, size=size)
    pts = p3js.Points(geometry_point, material_point)
    scene_children.append(pts)
    return scene_children


def lines_children(origins, targets, color="blue"):
    material = p3js.LineBasicMaterial(color=color, linewidth=4)

    scene_children = []
    # For each 24 joint
    for origin, target in zip(origins, targets):
        geometry = p3js.Geometry(vertices=np.array([origin, target]).tolist())
        line = p3js.Line(geometry, material)
        scene_children.append(line)
    return scene_children


def joint_children(joints3D, color="blue", links=None):
    material = p3js.LineBasicMaterial(color=color, linewidth=4)

    scene_children = []
    # For each 24 joint
    if links is None:
        links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]
    for link in links:
        for j1, j2 in zip(link[0:-1], link[1:]):
            geometry = p3js.Geometry(vertices=joints3D[(j1, j2), :].tolist())
            line = p3js.Line(geometry, material)
            scene_children.append(line)
    return scene_children
