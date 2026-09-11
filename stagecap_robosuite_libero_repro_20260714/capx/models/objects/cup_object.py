"""CupObject: a parametric open cylindrical cup that can physically contain
small free bodies (used as "water bead" stand-ins for liquid in the MannerCap
pouring/serving task).

Built the same way robosuite's own HollowCylinderObject approximates a
cylindrical shell with a ring of box geoms, but with an additional flat
bottom-cap geom so the cup actually holds objects placed inside it instead of
letting them fall through (HollowCylinderObject is an open tube, top AND
bottom).
"""

from __future__ import annotations

import numpy as np

from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import RED, CustomMaterial, add_to_dict


class CupObject(CompositeObject):
    """Open-top cylindrical cup made of a ring of box "wall" geoms plus one
    flat "bottom" geom, generated procedurally (no external mesh needed).

    Args:
        name: Name of this cup object.
        outer_radius: Outer radius of the cup wall (m).
        inner_radius: Inner radius of the cup cavity (m).
        height: Height of the cup wall (m).
        ngeoms: Number of box geoms approximating the cylindrical wall.
        rgba: Optional RGBA color override.
        material: Optional CustomMaterial.
        density: Geom density (kg/m^3).
    """

    def __init__(
        self,
        name,
        outer_radius: float = 0.035,
        inner_radius: float = 0.028,
        height: float = 0.06,
        ngeoms: int = 10,
        rgba=None,
        material: CustomMaterial | None = None,
        density: float = 400.0,
        solref=(0.02, 1.0),
        solimp=(0.9, 0.95, 0.001),
        friction=None,
    ):
        self._name = name
        self.rgba = rgba
        self.density = density
        self.friction = friction if friction is None else np.array(friction)
        self.solref = solref
        self.solimp = solimp

        self.has_material = material is not None
        if self.has_material:
            assert isinstance(material, CustomMaterial)
            self.material = material

        self._important_sites = {}

        self.r1 = inner_radius
        self.r2 = outer_radius
        self.n = ngeoms
        self.height = height

        self.unit_box_width = self.r2 * np.sin(np.pi / self.n)
        self.unit_box_height = (self.r2 - self.r1) * np.cos(np.pi / self.n) / 2.0
        # NOTE: MuJoCo box geom "size" is HALF-extent, so unlike robosuite's own
        # HollowCylinderObject (which treats its `height` ctor arg as an
        # already-halved value), here `self.height` is the TRUE full wall
        # height, so we must halve it to get the box's half-height (depth).
        self.unit_box_depth = self.height / 2.0
        self.int_r = (self.r1 * np.cos(np.pi / self.n)) + self.unit_box_height

        # bottom cap thickness
        self.bottom_thickness = 0.005

        super().__init__(**self._get_geom_attrs())

        if self.has_material:
            self.append_material(self.material)

    def _get_geom_attrs(self):
        base_args = {
            "total_size": [self.r2, self.r2, self.height / 2.0 + self.bottom_thickness],
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
            "density": self.density,
            "solref": self.solref,
            "solimp": self.solimp,
        }
        obj_args = {}

        angle_step = 2.0 * np.pi / self.n
        for i in range(self.n):
            geom_angle = np.pi - i * angle_step
            geom_center = np.array(
                [self.int_r * np.cos(geom_angle), self.int_r * np.sin(geom_angle), 0.0]
            )
            geom_quat = np.array(
                [np.cos(geom_angle / 2.0), 0.0, 0.0, np.sin(geom_angle / 2.0)]
            )
            geom_size = np.array(
                [self.unit_box_height, self.unit_box_width, self.unit_box_depth]
            )
            add_to_dict(
                dic=obj_args,
                geom_types="box",
                geom_locations=tuple(geom_center),
                geom_quats=tuple(geom_quat),
                geom_sizes=tuple(geom_size),
                geom_names=f"wall_{i}",
                geom_rgbas=self.rgba,
                geom_materials=self.material.mat_attrib["name"] if self.has_material else None,
                geom_frictions=self.friction,
                geom_condims=4,
            )

        # Bottom cap: flat cylinder-ish box spanning the inner radius, placed
        # just below the wall ring so beads can rest on it without leaking.
        bottom_z = -self.height / 2.0 - self.bottom_thickness / 2.0 + 0.001
        add_to_dict(
            dic=obj_args,
            geom_types="cylinder",
            geom_locations=(0.0, 0.0, bottom_z),
            geom_quats=(1.0, 0.0, 0.0, 0.0),
            geom_sizes=(self.r1 + 0.002, self.bottom_thickness / 2.0, 0.0),
            geom_names="bottom",
            geom_rgbas=self.rgba,
            geom_materials=self.material.mat_attrib["name"] if self.has_material else None,
            geom_frictions=self.friction,
            geom_condims=4,
        )

        obj_args["sites"] = [
            {
                "name": "center",
                "pos": (0, 0, 0),
                "size": "0.002",
                "rgba": RED,
                "type": "sphere",
            }
        ]

        obj_args.update(base_args)
        return obj_args
