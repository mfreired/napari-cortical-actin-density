"""
Napari Actin Density Plugin
Cortical actin density from a surface mesh (.vtp) + actin mask (.mrc).
"""

import os
import numpy as np
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QDoubleSpinBox, QSpinBox,
    QGroupBox, QProgressBar, QTextEdit,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline — yields dicts with "type": "progress" or "type": "result"
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline(params):
    p = params
    voxel_nm = p["voxel_size"] / 10.0

    def prog(step, msg):
        return {"type": "progress", "step": step, "msg": msg}

    # STEP 1
    import pyvista as pv
    yield prog(1, "Loading mesh…")
    mesh = pv.read(p["vtp_path"])
    mesh = mesh.triangulate()
    vertices = np.asarray(mesh.points)
    faces    = mesh.faces.reshape(-1, 4)[:, 1:]

    # STEP 2
    import mrcfile
    from skimage.morphology import skeletonize
    yield prog(2, "Loading actin mask…")
    with mrcfile.open(p["actin_path"], permissive=True) as mrc:
        actin = mrc.data.copy().astype(bool)
    yield prog(2, f"Actin voxels: {actin.sum()} — skeletonizing (may take 1-2 min)…")
    skeleton = skeletonize(actin)
    yield prog(2, f"Skeleton voxels: {skeleton.sum()}")

    # STEP 3
    from skimage.measure import label as sk_label
    yield prog(3, "Cleaning skeleton…")
    if skeleton.sum() == 0:
        xyz_clean = np.empty((0, 3), dtype=float)
    else:
        labeled = sk_label(skeleton)
        sizes   = np.bincount(labeled.ravel())
        sizes[0] = 0
        skeleton_clean = np.zeros_like(skeleton)
        for lid, sz in enumerate(sizes):
            if lid > 0 and sz >= p["min_size"]:
                skeleton_clean[labeled == lid] = 1
        zyx_clean = np.argwhere(skeleton_clean).astype(float)
        xyz_clean = zyx_clean[:, ::-1] * voxel_nm
    yield prog(3, f"Skeleton points: {len(xyz_clean)}")

    # STEP 4/5: density
    from scipy.spatial import cKDTree
    yield prog(4, "Computing cortical actin density…")

    CORTEX_DEPTH   = p["cortex_depth"]
    LATERAL_RADIUS = p["lateral_radius"]
    has_actin      = len(xyz_clean) > 0

    normals = np.asarray(
        mesh.compute_normals(
            point_normals=True, cell_normals=False,
            auto_orient_normals=False, consistent_normals=False,
        ).point_normals
    ).copy()

    bodies        = mesh.split_bodies()
    vertex_labels = np.full(len(vertices), -1, dtype=int)
    tree_v        = cKDTree(vertices)
    big_comp      = []
    for cid, body in enumerate(bodies):
        if body.n_points >= 1000:
            big_comp.append(cid)
            _, idx = tree_v.query(np.asarray(body.points), k=1)
            vertex_labels[idx] = cid

    if len(big_comp) == 2 and has_actin:
        labA, labB = big_comp
        mA, mB = vertex_labels == labA, vertex_labels == labB
        vA, vB = vertices[mA], vertices[mB]
        iA, iB = np.where(mA)[0], np.where(mB)[0]
        _, nb = cKDTree(vB).query(vA, k=1)
        d = vA - vB[nb]; d /= np.linalg.norm(d, axis=1, keepdims=True)
        normals[iA[np.einsum('ij,ij->i', normals[iA], d) < 0]] *= -1
        _, na = cKDTree(vA).query(vB, k=1)
        d = vB - vA[na]; d /= np.linalg.norm(d, axis=1, keepdims=True)
        normals[iB[np.einsum('ij,ij->i', normals[iB], d) < 0]] *= -1
    else:
        for cid in big_comp:
            mask    = vertex_labels == cid
            verts_c = vertices[mask]
            idx_g   = np.where(mask)[0]
            if has_actin:
                tmp = cKDTree(xyz_clean)
                nearby = set()
                for v in verts_c[::max(1, len(verts_c)//500)]:
                    nearby.update(tmp.query_ball_point(v, r=1000.0))
                ref = xyz_clean[list(nearby)].mean(axis=0) if nearby else verts_c.mean(axis=0)
            else:
                ref = verts_c.mean(axis=0)
            iv = ref - verts_c
            iv /= np.linalg.norm(iv, axis=1, keepdims=True)
            normals[idx_g[np.einsum('ij,ij->i', normals[idx_g], iv) < 0]] *= -1

    cell_areas      = np.asarray(
        mesh.compute_cell_sizes(length=False, area=True, volume=False).cell_data["Area"]
    )
    faces_arr       = mesh.faces.reshape(-1, 4)[:, 1:]
    vertex_area_nm2 = np.zeros(len(vertices))
    np.add.at(vertex_area_nm2, faces_arr, (cell_areas / 3.0)[:, None])
    vertex_area_um2 = vertex_area_nm2 * 1e-6

    cortex_counts = np.zeros(len(vertices), dtype=int)
    if has_actin:
        tree_actin = cKDTree(xyz_clean)
        search_r   = max(LATERAL_RADIUS, CORTEX_DEPTH) * 1.1
        for i, (v, n) in enumerate(zip(vertices, normals)):
            if vertex_labels[i] < 0:
                continue
            idx = tree_actin.query_ball_point(v, r=search_r)
            if not idx:
                continue
            pts    = xyz_clean[idx]
            d      = pts - v
            along  = d @ n
            lat_sq = np.einsum('ij,ij->i', d, d) - along**2
            cortex_counts[i] = (
                (along >= 0) & (along <= CORTEX_DEPTH) & (lat_sq <= LATERAL_RADIUS**2)
            ).sum()

    cortex_length_nm = cortex_counts * voxel_nm
    patch_area_um2   = np.pi * (LATERAL_RADIUS * 1e-3)**2
    density_um2      = (cortex_length_nm * 1e-3) / patch_area_um2

    # STEP 6: save
    yield prog(6, "Saving results…")
    import pandas as pd, json
    os.makedirs(p["output_path"], exist_ok=True)
    mesh_out = pv.PolyData(vertices, mesh.faces)
    mesh_out.point_data["cortex_density_um_per_um2"] = density_um2
    mesh_out.point_data["vertex_area_um2"]           = vertex_area_um2
    mesh_out.save(os.path.join(p["output_path"], "cortical_actin_density.vtp"))
    pd.DataFrame({
        "x_nm": vertices[:, 0], "y_nm": vertices[:, 1], "z_nm": vertices[:, 2],
        "vertex_area_um2": vertex_area_um2,
        "cortex_density_um_per_um2": density_um2,
    }).to_csv(os.path.join(p["output_path"], "cortical_actin_density.csv"), index=False)

    data_nz = density_um2[density_um2 > 0]
    stats = {
        "pm_area_um2":    float(vertex_area_um2.sum()),
        "total_actin_um": float(len(xyz_clean) * voxel_nm / 1000.0),
        "mean_density":   float(density_um2.mean()),
        "median_density": float(np.median(data_nz)) if len(data_nz) else 0.0,
        "coverage_pct":   float((density_um2 > 0).sum() / len(density_um2) * 100),
        "vertices":       vertices,
        "faces":          faces,
        "mesh_faces":     mesh.faces,
        "xyz_clean":      xyz_clean,
        "density_um2":    density_um2,
    }
    with open(os.path.join(p["output_path"], "cortical_actin_density_params.json"), "w") as f:
        json.dump({k: v for k, v in stats.items()
                   if not isinstance(v, np.ndarray)}, f, indent=2)

    # yield result as final message
    yield {"type": "result", **stats}


# ─────────────────────────────────────────────────────────────────────────────
#  Main widget
# ─────────────────────────────────────────────────────────────────────────────

class ActinDensityWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)

        title = QLabel("Cortical Actin Density")
        title.setStyleSheet("font-size:14px; font-weight:bold; padding:4px 0;")
        root.addWidget(title)

        grp = QGroupBox("Input Files")
        gl  = QVBoxLayout(grp)
        self.le_vtp = QLineEdit()
        self.le_vtp.setPlaceholderText("*.surface.vtp")
        btn = QPushButton("Browse…")
        btn.clicked.connect(lambda: self._browse_file(self.le_vtp, "VTP (*.vtp)"))
        gl.addWidget(QLabel("Mesh (.vtp):")); gl.addLayout(self._hrow(self.le_vtp, btn))

        self.le_actin = QLineEdit()
        self.le_actin.setPlaceholderText("actin mask *.mrc")
        btn2 = QPushButton("Browse…")
        btn2.clicked.connect(lambda: self._browse_file(self.le_actin, "MRC (*.mrc)"))
        gl.addWidget(QLabel("Actin mask (.mrc):")); gl.addLayout(self._hrow(self.le_actin, btn2))

        self.le_output = QLineEdit()
        self.le_output.setPlaceholderText("Output folder")
        btn3 = QPushButton("Browse…")
        btn3.clicked.connect(lambda: self._browse_dir(self.le_output))
        gl.addWidget(QLabel("Output folder:")); gl.addLayout(self._hrow(self.le_output, btn3))
        root.addWidget(grp)

        grp2 = QGroupBox("Parameters")
        pl   = QVBoxLayout(grp2)
        self.sb_voxel    = self._dspin(12.4,  1.0,   100.0,  0.1,  " Å")
        self.sb_min_size = self._spin( 20,    1,     10000)
        self.sb_cortex   = self._dspin(200.0, 10.0,  2000.0, 10.0, " nm")
        self.sb_lateral  = self._dspin(100.0, 10.0,  1000.0, 10.0, " nm")
        pl.addLayout(self._prow("Voxel size:",             self.sb_voxel))
        pl.addLayout(self._prow("Min skeleton component:", self.sb_min_size))
        pl.addLayout(self._prow("Cortex depth:",           self.sb_cortex))
        pl.addLayout(self._prow("Lateral radius:",         self.sb_lateral))
        root.addWidget(grp2)

        self.btn_run = QPushButton("▶  Run Pipeline")
        self.btn_run.setStyleSheet(
            "QPushButton{background:#0078d4;color:white;font-weight:bold;"
            "padding:8px;border-radius:4px;}"
            "QPushButton:hover{background:#006cbf;}"
            "QPushButton:disabled{background:#555;}"
        )
        self.btn_run.clicked.connect(self._run)
        root.addWidget(self.btn_run)

        self.progress = QProgressBar()
        self.progress.setMaximum(6)
        root.addWidget(self.progress)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(110)
        self.log.setStyleSheet("font-size:11px; font-family:monospace;")
        root.addWidget(self.log)

        grp3 = QGroupBox("Results")
        rl   = QVBoxLayout(grp3)
        self.lbl_area     = QLabel("PM area:        —")
        self.lbl_actin    = QLabel("Total actin:    —")
        self.lbl_mean     = QLabel("Mean density:   —")
        self.lbl_median   = QLabel("Median density: —")
        self.lbl_coverage = QLabel("PM coverage:    —")
        for lbl in [self.lbl_area, self.lbl_actin, self.lbl_mean,
                    self.lbl_median, self.lbl_coverage]:
            lbl.setStyleSheet("font-family:monospace; font-size:12px;")
            rl.addWidget(lbl)
        root.addWidget(grp3)
        root.addStretch()

    def _hrow(self, *widgets):
        row = QHBoxLayout()
        for w in widgets: row.addWidget(w)
        return row

    def _prow(self, label, widget):
        row = QHBoxLayout()
        lbl = QLabel(label); lbl.setFixedWidth(180)
        row.addWidget(lbl); row.addWidget(widget)
        return row

    def _dspin(self, val, mn, mx, step, suffix=""):
        w = QDoubleSpinBox()
        w.setRange(mn, mx); w.setValue(val); w.setSingleStep(step)
        if suffix: w.setSuffix(suffix)
        return w

    def _spin(self, val, mn, mx):
        w = QSpinBox(); w.setRange(mn, mx); w.setValue(val); return w

    def _browse_file(self, le, filt):
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "", filt)
        if path: le.setText(path)

    def _browse_dir(self, le):
        path = QFileDialog.getExistingDirectory(self, "Select folder")
        if path: le.setText(path)

    def _log(self, msg):
        self.log.append(msg)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _run(self):
        if not self.le_vtp.text():
            self._log("⚠ Please select a .vtp mesh file."); return
        if not self.le_actin.text():
            self._log("⚠ Please select an actin .mrc file."); return
        if not self.le_output.text():
            self._log("⚠ Please select an output folder."); return

        params = {
            "vtp_path":       self.le_vtp.text(),
            "actin_path":     self.le_actin.text(),
            "output_path":    self.le_output.text(),
            "voxel_size":     self.sb_voxel.value(),
            "min_size":       self.sb_min_size.value(),
            "cortex_depth":   self.sb_cortex.value(),
            "lateral_radius": self.sb_lateral.value(),
        }

        self.btn_run.setEnabled(False)
        self.progress.setValue(0)
        self.log.clear()
        self._log("Starting pipeline…")

        @thread_worker(connect={"yielded": self._on_yield,
                                "errored": self._on_error})
        def _worker():
            yield from _run_pipeline(params)

        _worker()

    def _on_yield(self, update):
        if update["type"] == "progress":
            self.progress.setValue(update["step"])
            self._log(f"[{update['step']}/6] {update['msg']}")
        elif update["type"] == "result":
            self._on_finished(update)

    def _on_finished(self, stats):
        self.btn_run.setEnabled(True)
        self.progress.setValue(6)

        v = self.viewer
        v.dims.ndisplay = 3
        for name in ["OMM mesh", "Actin skeleton", "Actin density"]:
            for layer in list(v.layers):
                if layer.name == name:
                    v.layers.remove(layer)

        v.add_surface(
            (stats["vertices"], stats["faces"]),
            name="OMM mesh", shading="smooth", colormap="gray", opacity=0.4,
        )
        if len(stats["xyz_clean"]) > 0:
            v.add_points(
                stats["xyz_clean"], name="Actin skeleton",
                size=3, face_color="yellow", opacity=0.8,
            )
        v.add_surface(
            (stats["vertices"], stats["faces"], stats["density_um2"]),
            name="Actin density", colormap="turbo", opacity=0.9,
        )

        self.lbl_area.setText(    f"PM area:        {stats['pm_area_um2']:.3f} µm²")
        self.lbl_actin.setText(   f"Total actin:    {stats['total_actin_um']:.3f} µm")
        self.lbl_mean.setText(    f"Mean density:   {stats['mean_density']:.4f} µm/µm²")
        self.lbl_median.setText(  f"Median density: {stats['median_density']:.4f} µm/µm²")
        self.lbl_coverage.setText(f"PM coverage:    {stats['coverage_pct']:.1f}%")
        self._log("✓ Results saved to output folder.")

    def _on_error(self, exc):
        self.btn_run.setEnabled(True)
        import traceback
        self._log(f"ERROR:\n{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}")
