import numpy as np
from tqdm import tqdm
from ...physics import C_SOUND


class FDTDSolver:
    """
    Finite Difference Time Domain (FDTD) solver for low-frequency room acoustics.
    Solves the scalar wave equation on a 3D grid.
    """

    def __init__(self, room, max_freq=1000.0, ppw=6.0, cfl=1.0 / np.sqrt(3)):
        """
        Initialize the FDTD solver.

        :param room: RayRoom Room object.
        :param max_freq: Maximum frequency to simulate (determines grid size).
        :param ppw: Points per wavelength (dispersion control). Typically 6-10.
        :param cfl: Courant number (stability). Max 1/sqrt(3) for 3D.
        """
        self.room = room
        self.max_freq = max_freq

        # 1. Grid Setup
        # lambda_min = c / f_max
        # dx <= lambda_min / ppw
        lambda_min = C_SOUND / max_freq
        self.dx = lambda_min / ppw

        # dt <= dx * CFL / c
        self.dt = (self.dx * cfl) / C_SOUND

        print(f"FDTD Setup: f_max={max_freq}Hz, dx={self.dx * 100:.2f}cm, dt={self.dt * 1000:.4f}ms")

        # 2. Voxelize Room
        self._voxelize_room()

        # 3. State Arrays
        # p[n+1], p[n], p[n-1]
        self.p = np.zeros(self.grid_shape, dtype=np.float32)
        self.p_prev = np.zeros(self.grid_shape, dtype=np.float32)
        self.p_next = np.zeros(self.grid_shape, dtype=np.float32)

        # Precompute coefficients
        # Standard wave eq update:
        # p_next = 2*p - p_prev + (c*dt/dx)^2 * Laplacian(p)
        self.courant_sq = (C_SOUND * self.dt / self.dx) ** 2

        # Estimate boundary damping factor from room materials
        # Average absorption coefficient of walls
        abs_coeffs = []
        for w in self.room.walls:
            mat = w.material
            val = np.mean(mat.absorption) if np.ndim(mat.absorption) > 0 else mat.absorption
            abs_coeffs.append(val)

        # Impedance boundary condition approximation for FDTD
        # Reflection R = sqrt(1 - alpha)
        # Damping per step ~ R? No.
        # Ideally, we use an impedance update equation at boundaries.
        # Simplified "lossy layer": apply a decay factor at the boundary mask.
        # If is_air is 1 (inside), factor is 1.0.
        # If is_air is 0 (boundary), factor is R?
        # No, standard FDTD puts nodes ON the boundary.
        # Our mask `is_air` is binary.
        # Let's create a `damping_grid`.

        # Simple "Lossy Wall" approximation:
        # Inside air: damping = 1.0 (no loss)
        # At wall interface (neighbors of air that are not air): damping = R
        # But we don't simulate inside walls.
        # We simulate Air.
        # So we need to dampen the pressure *at the edge of air*.

        # Let's refine the voxelization to identify boundary nodes.
        # For MVP: Just apply a global decay term to the wave equation to simulate absorption.
        # T60 approx?
        # Better: Apply decay to p_next everywhere? That simulates air absorption.
        # We want wall absorption.

        # Let's use a simplified approach:
        # Apply a uniform damping factor corresponding to expected T60 of the LF band.
        # This prevents infinite ringing (tube effect).

        # Sabine T60 = 0.161 * V / (Sum(S*alpha))
        # decay_constant = 6.91 / T60
        # per_step_decay = exp(-decay_constant * dt)

        # Calculate V and A
        # More precise:
        # Volume = sum(is_air) * dx^3
        volume = np.sum(self.is_air) * (self.dx ** 3)

        # Surface Area?
        # Hard to estimate from voxels without marching cubes.
        # Use room walls directly.
        surface_area = 0.0
        total_absorption_area = 0.0

        for w in self.room.walls:
            # Area of polygon
            # Shoebox walls are rectangles: width * height
            # Generic polygon area: 0.5 * |cross sum|
            # Let's assume rectangular for simplicity or calc generic
            # Generic area of 3D planar polygon:
            # Project to 2D plane (max normal component) and calc 2D area / normal_component?
            # Or 0.5 * norm(sum(cross(vi, vi+1)))

            # Vertices
            v = w.vertices
            n = len(v)
            area_vec = np.array([0.0, 0.0, 0.0])
            for i in range(n):
                area_vec += np.cross(v[i], v[(i + 1) % n])
            area = 0.5 * np.linalg.norm(area_vec)

            mat = w.material
            alpha = np.mean(mat.absorption) if np.ndim(mat.absorption) > 0 else mat.absorption

            surface_area += area
            total_absorption_area += area * alpha

        if total_absorption_area > 0:
            t60 = 0.161 * volume / total_absorption_area
            decay_constant = 6.91 / t60
            self.damping_factor = np.exp(-decay_constant * self.dt)
        else:
            self.damping_factor = 1.0
            t60 = float('inf')

        print(f"Estimated LF T60: {t60:.2f}s. Damping factor: {self.damping_factor:.6f}")

    def _voxelize_room(self):
        """
        Discretize the room geometry into a boolean grid (Air/Solid).
        """
        # Find bounds
        all_verts = []
        for w in self.room.walls:
            all_verts.extend(w.vertices)
        all_verts = np.array(all_verts)

        min_bounds = np.min(all_verts, axis=0) - 2 * self.dx  # Padding
        max_bounds = np.max(all_verts, axis=0) + 2 * self.dx

        self.origin = min_bounds
        size = max_bounds - min_bounds
        self.grid_shape = np.ceil(size / self.dx).astype(int)

        grid_volume_million = self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2] / 1e6
        print(f"Grid shape: {self.grid_shape} ({grid_volume_million:.2f} M voxels)")

        x = np.arange(self.grid_shape[0]) * self.dx + self.origin[0]
        y = np.arange(self.grid_shape[1]) * self.dx + self.origin[1]
        z = np.arange(self.grid_shape[2]) * self.dx + self.origin[2]

        self.is_air = np.zeros(self.grid_shape, dtype=bool)

        print("Voxelizing room (assuming convex for speed)...")

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        points = np.stack((X, Y, Z), axis=-1)

        mask = np.ones(self.grid_shape, dtype=bool)

        for wall in self.room.walls:
            v0 = wall.vertices[0]
            n = wall.normal

            dot_val = points[:, :, :, 0] * n[0] + points[:, :, :, 1] * n[1] + points[:, :, :, 2] * n[2]
            threshold = np.dot(v0, n)

            mask_wall = dot_val >= threshold - 1e-5
            mask = np.logical_and(mask, mask_wall)

        self.is_air = mask

        # Also handle furniture (as obstacles/solids)
        # for furn in self.room.furniture:
        #     pass

        print(f"Voxelization complete. Air volume fraction: {np.mean(self.is_air):.2%}")

    def _world_to_grid(self, pos):
        idx = (np.array(pos) - self.origin) / self.dx
        return np.round(idx).astype(int)

    def run(self, duration, sources, receivers, source_signals=None):
        steps = int(duration / self.dt)

        src_indices = []
        for src in sources:
            idx = self._world_to_grid(src.position)
            idx = np.clip(idx, 0, np.array(self.grid_shape) - 1)
            src_indices.append((src, idx))

        rec_indices = []
        rec_signals = {r: np.zeros(steps) for r in receivers}
        for r in receivers:
            idx = self._world_to_grid(r.position)
            idx = np.clip(idx, 0, np.array(self.grid_shape) - 1)
            rec_indices.append((r, idx))

        # C^2
        C2 = self.courant_sq

        print(f"Running FDTD for {steps} steps...")

        for t in tqdm(range(steps)):

            # 1. Inject Source
            for i, (src, idx) in enumerate(src_indices):
                val = 0.0
                if source_signals and src in source_signals:
                    sig = source_signals[src]
                    if t < len(sig):
                        val = sig[t]
                else:
                    # Gaussian Pulse
                    width = 5
                    val = np.exp(-((t - 20) ** 2) / (2 * width ** 2))

                self.p[idx[0], idx[1], idx[2]] += val

            # 2. Update Field (Vectorized)
            p_c = self.p[1:-1, 1:-1, 1:-1]
            p_prev_c = self.p_prev[1:-1, 1:-1, 1:-1]

            p_x1 = self.p[2:, 1:-1, 1:-1]
            p_x0 = self.p[:-2, 1:-1, 1:-1]
            p_y1 = self.p[1:-1, 2:, 1:-1]
            p_y0 = self.p[1:-1, :-2, 1:-1]
            p_z1 = self.p[1:-1, 1:-1, 2:]
            p_z0 = self.p[1:-1, 1:-1, :-2]

            lap_sum = (p_x1 + p_x0 + p_y1 + p_y0 + p_z1 + p_z0 - 6.0 * p_c)

            self.p_next[1:-1, 1:-1, 1:-1] = 2.0 * p_c - p_prev_c + C2 * lap_sum

            # 3. Apply Boundary Conditions / Damping
            # Mask: Forces 0 outside
            self.p_next *= self.is_air

            # Damping: Apply global exponential decay to simulate wall absorption
            # This is an approximation to avoid implementing impedance boundaries.
            self.p_next *= self.damping_factor

            # 4. Record Receivers
            for r, idx in rec_indices:
                rec_signals[r][t] = self.p_next[idx[0], idx[1], idx[2]]

            # Cycle buffers
            self.p_prev[:] = self.p[:]
            self.p[:] = self.p_next[:]

        return rec_signals, (1.0 / self.dt)
