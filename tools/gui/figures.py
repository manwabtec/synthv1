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
SET_COLOURS = {"outside": ["#1f77b4", "#6baed6"], "inside": ["#e6550d", "#fdae6b"]}  # blue / orange


def _f32(a):
    """Compact arrays for the browser (plotly sends numpy arrays as binary)."""
    return np.asarray(a, np.float32)


def cam_colours(sets) -> dict:
    """Camera -> colour: blue shades outside the rail, orange shades inside."""
    out = {}
    for s in sets:
        shades = SET_COLOURS[s.label]
        for i, c in enumerate(s.cameras):
            out[c] = shades[i % len(shades)]
    return out


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


def layout_3d(rig: RigModel, L: lay.Layout, key_points: dict[str, list[float]], height: int = 720,
              sets=None, profile: "lay.ProfileCoverage | None" = None) -> go.Figure:
    fig = go.Figure()
    colours = cam_colours(sets or [])
    for cls, colour in CLASS_COLOURS.items():
        ms = [m for m in L.meshes if m.cls == cls]
        if not ms:
            continue
        V = np.concatenate([m.V for m in ms])
        off = np.cumsum([0] + [len(m.V) for m in ms[:-1]])
        F = np.concatenate([m.F + o for m, o in zip(ms, off)])
        opacity = 0.35 if cls == "ballast" else 1.0
        V, F = _f32(V), F.astype(np.int32)
        fig.add_trace(go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2],
                                color=colour, opacity=opacity, name=cls, flatshading=False,
                                lighting=dict(ambient=0.55, diffuse=0.6, specular=0.15), hoverinfo="name"))
    for k, (name, cam) in enumerate(rig.cameras.items()):
        colour = colours.get(name, CAM_COLOURS[k % len(CAM_COLOURS)])
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
        fig.add_trace(go.Scatter3d(x=_f32(pts[ok, 0]), y=_f32(pts[ok, 1]), z=_f32(pts[ok, 2]), mode="markers",
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
    if profile is not None and sets:
        add_profile_inset(fig, profile, sets)
    return fig


def add_profile_inset(fig: go.Figure, pc: "lay.ProfileCoverage", sets) -> None:
    """2D inset (top right of the 3D view): the wheel profile, with the parts seen by both
    cameras of each set as thick bands (outside = blue, inside = orange) and parts seen by
    only one camera as thin lines. Offsets keep the bands apart."""
    a, dr = pc.a * 1000, pc.dr * 1000
    tng = np.gradient(np.stack([a, dr], 1), axis=0)
    nrm = np.stack([-tng[:, 1], tng[:, 0]], 1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    total = pc.arc.sum()
    fig.add_trace(go.Scatter(x=_f32(a), y=_f32(dr), mode="lines", line=dict(color="#555", width=1.5),
                             xaxis="x", yaxis="y", name="profile", showlegend=False, hoverinfo="skip"))
    lines = []
    for k_, s in enumerate(sets):
        if s.label not in SET_COLOURS:
            continue
        seen = [pc.seen[c] for c in s.cameras if c in pc.seen]
        both = np.logical_and.reduce(seen)
        anyc = np.logical_or.reduce(seen)
        off = 3.0 + 3.5 * k_
        xo, yo = a + off * nrm[:, 0], dr + off * nrm[:, 1]
        for mask, width, opacity in ((anyc & ~both, 1.5, 0.6), (both, 4.5, 1.0)):
            fig.add_trace(go.Scatter(x=_f32(np.where(mask, xo, np.nan)), y=_f32(np.where(mask, yo, np.nan)),
                                     mode="lines", line=dict(color=SET_COLOURS[s.label][0], width=width),
                                     opacity=opacity, xaxis="x", yaxis="y", showlegend=False, connectgaps=False,
                                     hoverinfo="skip"))
        lines.append(f"{s.label} {'+'.join(s.cameras)}: {pc.arc[both].sum() / total:.0%} both, "
                     f"{pc.arc[anyc].sum() / total:.0%} any")
    fig.update_layout(
        xaxis=dict(domain=[0.66, 0.995], anchor="y", title=dict(text="lateral from flange back (mm)", font=dict(size=10)),
                   tickfont=dict(size=9), showgrid=False, zeroline=False),
        yaxis=dict(domain=[0.64, 0.995], anchor="x", title=dict(text="radial (mm)", font=dict(size=10)),
                   tickfont=dict(size=9), showgrid=False, scaleanchor="x", zeroline=False),
        annotations=[dict(text=f"<b>Profile coverage</b> (section {pc.angle_deg:.0f} deg from bottom)<br>" + "<br>".join(lines),
                          xref="paper", yref="paper", x=0.665, y=0.63, xanchor="left", yanchor="top", showarrow=False,
                          align="left", font=dict(size=11), bgcolor="rgba(255,255,255,0.85)")],
        shapes=[dict(type="rect", xref="paper", yref="paper", x0=0.62, x1=1.0, y0=0.55, y1=1.0,
                     fillcolor="rgba(255,255,255,0.88)", line=dict(color="#999", width=1), layer="below")],
        plot_bgcolor="rgba(0,0,0,0)")


def camera_view(rig: RigModel, L: lay.Layout, cam_name: str, height: int = 260, budget: int = 1500) -> go.Figure:
    """What a camera sees, approximately: projected mesh vertices + laser hits."""
    cam = rig.cameras[cam_name]
    w, h = cam.size
    fig = go.Figure()
    for cls, colour in CLASS_COLOURS.items():
        V = [m.V for m in L.meshes if m.cls == cls]
        if not V:
            continue
        V = np.concatenate(V)
        uv, ok = lay.project(cam, V)
        idx = np.nonzero(ok)[0]
        if len(idx) > budget // 4:
            idx = idx[np.linspace(0, len(idx) - 1, budget // 4).astype(int)]
        fig.add_trace(go.Scattergl(x=_f32(uv[idx, 0]), y=_f32(uv[idx, 1]), mode="markers", name=cls, showlegend=False,
                                   marker=dict(size=2, color=colour, opacity=0.5), hoverinfo="name"))
    for name, Lm in rig.lasers.items():
        pts, _ = L.laser_hits[name]
        okp = ~np.isnan(pts[:, 0])
        uv, ok = lay.project(cam, pts[okp])
        fig.add_trace(go.Scattergl(x=_f32(uv[ok, 0]), y=_f32(uv[ok, 1]), mode="markers", name=name, showlegend=False,
                                   marker=dict(size=4, color=_rgb(Lm.color_rgb)), hoverinfo="name"))
    cov = ", ".join(f"{k} {v:.0%}" for k, v in L.coverage[cam_name].items())
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=24, b=0), title=dict(text=f"{cam_name}: {cov}", font=dict(size=12)),
        xaxis=dict(range=[0, w], showticklabels=False, constrain="domain"),
        yaxis=dict(range=[h, 0], showticklabels=False, scaleanchor="x"),
        plot_bgcolor="#202020", uirevision=cam_name)
    return fig
