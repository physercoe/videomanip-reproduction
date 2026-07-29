#!/usr/bin/env python3
"""Create the DRO-Grasp 'extended' Inspire-hand URDF (6 virtual root joints + hand)
and register the 'inspire' robot in drograsp's data dirs.

Input : third_party/xr_teleoperate/assets/inspire_hand/inspire_hand_right.urdf
Output: third_party/drograsp/data/data_urdf/robot/inspire/inspire_hand_right_extended.urdf
        third_party/drograsp/data/data_urdf/robot/inspire/meshes/*.STL
        (patched) urdf_assets_meta.json, removed_links.json
"""
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "third_party/xr_teleoperate/assets/inspire_hand"
DST = ROOT / "third_party/drograsp/data/data_urdf/robot/inspire"
META = ROOT / "third_party/drograsp/data/data_urdf/robot/urdf_assets_meta.json"
REMOVED = ROOT / "third_party/drograsp/data_utils/removed_links.json"

VIRTUAL = """
  <link name="world" />
  <link name="virtual_link_dummy">
    <inertial><mass value="0.01"/><inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/></inertial>
  </link>
  <!-- sacrificial zero-limit joint: the Isaac Sim URDF importer consumes the first
       joint of the chain (replaced by the articulation root joint) -->
  <joint name="virtual_joint_dummy" type="prismatic">
      <parent link="world" /><child link="virtual_link_dummy" />
      <origin rpy="0 0 0" xyz="0 0 0" /><axis xyz="1 0 0" />
      <limit effort="0" lower="0" upper="0" velocity="0" />
  </joint>
  <link name="virtual_link_x">
    <inertial><mass value="0.01"/><inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/></inertial>
  </link>
  <joint name="virtual_joint_x" type="prismatic">
      <parent link="virtual_link_dummy" /><child link="virtual_link_x" />
      <origin rpy="0 0 0" xyz="0 0 0" /><axis xyz="1 0 0" />
      <limit effort="300" lower="-10" upper="10" velocity="2" />
  </joint>
  <link name="virtual_link_y">
    <inertial><mass value="0.01"/><inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/></inertial>
  </link>
  <joint name="virtual_joint_y" type="prismatic">
      <parent link="virtual_link_x" /><child link="virtual_link_y" />
      <origin rpy="0 0 0" xyz="0 0 0" /><axis xyz="0 1 0" />
      <limit effort="300" lower="-10" upper="10" velocity="2" />
  </joint>
  <link name="virtual_link_z">
    <inertial><mass value="0.01"/><inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/></inertial>
  </link>
  <joint name="virtual_joint_z" type="prismatic">
      <parent link="virtual_link_y" /><child link="virtual_link_z" />
      <origin rpy="0 0 0" xyz="0 0 0" /><axis xyz="0 0 1" />
      <limit effort="300" lower="-10" upper="10" velocity="2" />
  </joint>
  <link name="virtual_link_roll">
    <inertial><mass value="0.01"/><inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/></inertial>
  </link>
  <joint name="virtual_joint_roll" type="revolute">
      <parent link="virtual_link_z" /><child link="virtual_link_roll" />
      <origin rpy="0 0 0" xyz="0 0 0" /><axis xyz="1 0 0" />
      <limit effort="100" velocity="100" lower="-6.283185" upper="6.283185" />
  </joint>
  <link name="virtual_link_pitch">
    <inertial><mass value="0.01"/><inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/></inertial>
  </link>
  <joint name="virtual_joint_pitch" type="revolute">
      <parent link="virtual_link_roll" /><child link="virtual_link_pitch" />
      <origin rpy="0 0 0" xyz="0 0 0" /><axis xyz="0 1 0" />
      <limit effort="100" velocity="100" lower="-6.283185" upper="6.283185" />
  </joint>
  <link name="virtual_link_yaw">
    <inertial><mass value="0.01"/><inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/></inertial>
  </link>
  <joint name="virtual_joint_yaw" type="revolute">
      <parent link="virtual_link_pitch" /><child link="virtual_link_yaw" />
      <origin rpy="0 0 0" xyz="0 0 0" /><axis xyz="0 0 1" />
      <limit effort="100" velocity="100" lower="-6.283185" upper="6.283185" />
  </joint>
  <joint name="virtual_robot" type="fixed">
      <parent link="virtual_link_yaw" /><child link="R_hand_base_link" />
      <origin rpy="0 0 0" xyz="0 0 0" />
  </joint>
"""


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    # meshes
    (DST / "meshes").mkdir(exist_ok=True)
    for stl in (SRC / "meshes").glob("*_R.STL"):
        shutil.copy2(stl, DST / "meshes" / stl.name)
    shutil.copy2(SRC / "meshes/R_hand_base_link.STL", DST / "meshes/R_hand_base_link.STL")

    # urdf: strip <robot> wrapper, prepend virtual chain
    text = (SRC / "inspire_hand_right.urdf").read_text()
    tree = ET.fromstring(text)
    assert tree.tag == "robot"
    inner = text[text.index(">", text.index("<robot")) + 1:text.rindex("</robot>")].strip()
    out = '<?xml version="1.0" ?>\n<robot name="inspire_right">\n' + VIRTUAL + "\n" + inner + "\n</robot>\n"
    # normalize mesh paths to drograsp layout (relative to urdf dir)
    out = out.replace("./meshes/", "meshes/")
    (DST / "inspire_hand_right_extended.urdf").write_text(out)

    meta = json.loads(META.read_text())
    meta["urdf_path"]["inspire"] = "data/data_urdf/robot/inspire/inspire_hand_right_extended.urdf"
    meta["meshes_path"]["inspire"] = "data/data_urdf/robot/inspire/meshes"
    META.write_text(json.dumps(meta, indent=4))

    removed = json.loads(REMOVED.read_text())
    removed["inspire"] = []
    REMOVED.write_text(json.dumps(removed, indent=4))

    print("[ok] wrote", DST / "inspire_hand_right_extended.urdf")
    print("[ok] patched", META.name, "and", REMOVED.name)


if __name__ == "__main__":
    main()
