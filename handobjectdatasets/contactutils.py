from functools import lru_cache
import pickle
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mano_train.visualize import displaymano


@lru_cache(maxsize=128)
def load_contacts(
    save_contact_paths="assets/contact_zones.pkl", display=False
):
    with open(save_contact_paths, "rb") as p_f:
        contact_data = pickle.load(p_f)
    hand_verts = contact_data["verts"]
    if display:
        colors = [
            "#f04e36",
            "#f36e27",
            ["#f3d430"],
            ["#1eb19d"],
            ["#ed1683"],
            ["#37bad6"],
        ]
        hand_faces = contact_data["faces"]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # Display hand and object meshes
        hand_mesh_disp = Poly3DCollection(hand_verts[hand_faces], alpha=0.1)
        hand_mesh_disp.set_edgecolor("k")
        hand_mesh_disp.set_facecolor([[1, 1, 1], [1, 0, 0]])
        ax.add_collection3d(hand_mesh_disp)
        idx_1, idx_2, idx_3 = 0, 1, 2
        ax.axis("off")
        # ax.scatter(hand_verts[:, idx_1], hand_verts[:, idx_2])
        for zone_idx, zone_vert_idxs in contact_data["contact_zones"].items():
            ax.scatter(
                hand_verts[zone_vert_idxs, idx_1],
                hand_verts[zone_vert_idxs, idx_2],
                hand_verts[zone_vert_idxs, idx_3],
                s=100,
                c=colors[zone_idx],
            )
        displaymano.cam_equal_aspect_3d(ax, hand_verts)
        plt.show()
    return hand_verts, contact_data["contact_zones"]
