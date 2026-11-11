"""Plotly figures for the GUI layout tab: 3D layout and per-camera views. No bpy."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from synthrig import layout as lay
from synthrig.opticsrig import RigModel

CLASS_COLOURS = {
    "wheel_tread": "#b08d5f", "wheel_flange": "#c99a62", "wheel_rim_face": "#9c7b55",
    "wheel_plate": "#8a8a8a", "wheel_hub": "#6a6a6a", "rail_head": "#5b7fb8", "rail_web": "#4a6795",
    "rail_foot": "#3c5378", "tie": "#7a5a36", "ballast": "#c8c3b5",
}
CAM_COLOURS = ["#00a6d6", "#0072b2", "#009e73", "#56b4e9", "#cc79a7", "#e69f00"]


def _rgb(c) -> str:
    c = np.clip(np.asarray(c, float), 0, 1)
    c = 0.25 + 0.75 * c / max(c.max(), 1e-6)  # keep laser colours visible on white
    return "rgb({},{},{})".format(*(int(255 * v) for v in c))


def _lines(points_list, colour, name, width=3, showlegend=True, dash=None):
    xs, ys, zs = [], [], []
    for pts in points_list:
        pts = np.asarray(pts)
        xs += list(pts[:, 0]) + [None]
        ys += list(pts[:, 1]) + [None]
        zs += list(pts[:, 2]) + [None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=colour, width=width, dash=dash),
                        name=name, showlegend=showlegend, hoverinfo="name")


def layout_3d(rig: RigModel, L: lay.Layout, key_points: dict[str, list[float]], height: int = 720) -> go.Figure:
    fig = go.Figure()
    for cls, colour in CLASS_COLOURS.items():
        ms = [m for m in L.meshes if m.cls == cls]
        if not ms:
            continue
        V = np.concatenate([m.V for m in ms])
        off = np.cumsum([0] + [len(m.V) for m in ms[:-1]])
        F = np.concatenate([m.F + o for m, o in zip(ms, off)])
        opacity = 0.35 if cls == "ballast" else 1.0
        fig.add_trace(go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2],
                                color=colour, opacity=opacity, name=cls, flatshading=False,
                                lighting=dict(ambient=0.55, diffuse=0.6, specular=0.15), hoverinfo="name"))
    for k, (name, cam) in enumerate(rig.cameras.items()):
        colour = CAM_COLOURS[k % len(CAM_COLOURS)]
        aim = cam.spec.pose.look_at
        depth = float(np.linalg.norm(np.asarray(aim) - cam.T_rig_cam[:3, 3])) * 1.15 if aim else 1.0
        c, q = lay.camera_frustum(cam, depth)
        rays = [[c, q[i]] for i in range(4)] + [np.vstack([q, q[:1]])]
        fig.add_trace(_lines(rays, colour, f"{name} ({lay.hfov_deg(cam):.0f} deg)", 4))
        fig.add_trace(go.Scatter3d(x=[c[0]], y=[c[1]], z=[c[2]], mode="markers+text", text=[name],
                                   textposition="top center", marker=dict(size=6, color=colour, symbol="square"),
                                   showlegend=False, hoverinfo="text"))
    for name, Lm in rig.lasers.items():
        colour = _rgb(Lm.color_rgb)
        pts, _ = L.laser_hits[name]
        ok = ~np.isnan(pts[:, 0])
        o = Lm.T_rig_proj[:3, 3]
        aim = Lm.spec.pose.look_at
        reach = float(np.linalg.norm(np.asarray(aim) - o)) * 1.1 if aim else 1.0
        ends = lay.laser_fan_outline(Lm, reach)
        fig.add_trace(_lines([[o, e] for e in ends], colour, f"{name} fan", 3, dash="dot"))
        fig.add_trace(go.Scatter3d(x=pts[ok, 0], y=pts[ok, 1], z=pts[ok, 2], mode="markers",
                                   marker=dict(size=2.5, color=colour), name=f"{name} hits"))
        fig.add_trace(go.Scatter3d(x=[o[0]], y=[o[1]], z=[o[2]], mode="markers+text", text=[name],
                                   textposition="top center", marker=dict(size=6, color=colour, symbol="diamond"),
                                   showlegend=False, hoverinfo="text"))
    kp = np.array(list(key_points.values()))
    fig.add_trace(go.Scatter3d(x=kp[:, 0], y=kp[:, 1], z=kp[:, 2], mode="markers", name="aim points",
                               text=list(key_points), marker=dict(size=4, color="black", symbol="cross"),
                               hoverinfo="text"))
    # view box: the wheel plus every device, padded; the track itself is clipped to it
    devs = [c.T_rig_cam[:3, 3] for c in rig.cameras.values()] + [l.T_rig_proj[:3, 3] for l in rig.lasers.values()]
    pts = np.vstack([kp, *devs]) if devs else kp
    lo, hi = pts.min(0) - 0.3, pts.max(0) + 0.3
    lo[2], hi[2] = min(lo[2], -0.35), max(hi[2], 1.0)
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0), uirevision="layout",
        legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0.7)", font=dict(size=11)),
        scene=dict(aspectmode="data", xaxis_title="x along track (m)", yaxis_title="y lateral (m)",
                   zaxis_title="z up (m)", camera=dict(eye=dict(x=-1.1, y=-1.6, z=0.8)),
                   xaxis=dict(range=[lo[0], hi[0]]), yaxis=dict(range=[lo[1], hi[1]]),
                   zaxis=dict(range=[lo[2], hi[2]])))
    return fig


def camera_view(rig: RigModel, L: lay.Layout, cam_name: str, height: int = 260) -> go.Figure:
    """What a camera sees, approximately: projected mesh vertices + laser hits."""
    cam = rig.cameras[cam_name]
    w, h = cam.size
    fig = go.Figure()
    for cls, colour in CLASS_COLOURS.items():
        V = [m.V for m in L.meshes if m.cls == cls]
        if not V:
            continue
        V = np.concatenate(V)
        if len(V) > 4000:
            V = V[np.linspace(0, len(V) - 1, 4000).astype(int)]
        uv, ok = lay.project(cam, V)
        fig.add_trace(go.Scattergl(x=uv[ok, 0], y=uv[ok, 1], mode="markers", name=cls, showlegend=False,
                                   marker=dict(size=2, color=colour, opacity=0.5), hoverinfo="name"))
    for name, Lm in rig.lasers.items():
        pts, _ = L.laser_hits[name]
        okp = ~np.isnan(pts[:, 0])
        uv, ok = lay.project(cam, pts[okp])
        fig.add_trace(go.Scattergl(x=uv[ok, 0], y=uv[ok, 1], mode="markers", name=name, showlegend=False,
                                   marker=dict(size=4, color=_rgb(Lm.color_rgb)), hoverinfo="name"))
    cov = ", ".join(f"{k} {v:.0%}" for k, v in L.coverage[cam_name].items())
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=24, b=0), title=dict(text=f"{cam_name}: {cov}", font=dict(size=12)),
        xaxis=dict(range=[0, w], showticklabels=False, constrain="domain"),
        yaxis=dict(range=[h, 0], showticklabels=False, scaleanchor="x"),
        plot_bgcolor="#202020", uirevision=cam_name)
    return fig
