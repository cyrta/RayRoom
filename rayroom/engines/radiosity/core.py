import numpy as np
from tqdm import tqdm
from ...geometry import normalize
from ...physics import C_SOUND

class RadiositySolver:
    """
    Acoustic Radiosity Solver for late diffuse reverberation.
    Divides room surfaces into patches and solves energy exchange.
    """

    def __init__(self, room, patch_size=0.5):
        """
        :param room: The Room object.
        :param patch_size: Approximate side length of a square patch (meters).
        """
        self.room = room
        self.patch_size = patch_size
        self.patches = []
        self.view_factors = None
        
        # 1. Discretize Room into Patches
        self._create_patches()
        
        # 2. Compute Form Factors (View Factors)
        # This is an N x N matrix calculation. Can be slow.
        self._compute_view_factors()

    def _create_patches(self):
        """
        Subdivide all walls into smaller patches.
        """
        print(f"Radiosity: Meshing walls with patch_size={self.patch_size}m...")
        self.patches = []
        
        for wall in self.room.walls:
            # Assuming rectangular walls for simple subdivision
            # Generic polygon subdivision is harder (Delaunay or grid clipping)
            # We'll implement a simple grid based on the first 3 vertices
            
            v0 = wall.vertices[0]
            v1 = wall.vertices[1]
            v2 = wall.vertices[2]
            
            # Vectors
            vec_u = v1 - v0
            vec_v = v3 = wall.vertices[3] - v0 # Assuming ordered quad v0,v1,v2,v3?
            # RayRoom Wall creates: 0,1,2,3. 
            # Let's verify: v[1]-v[0] is one edge. v[3]-v[0] is other edge?
            # Box creation: 0->1 (x), 0->3 (y). Yes.
            
            # Dimensions
            len_u = np.linalg.norm(vec_u)
            len_v = np.linalg.norm(wall.vertices[3] - v0)
            
            u_dir = vec_u / len_u
            v_dir = (wall.vertices[3] - v0) / len_v
            
            n_u = int(np.ceil(len_u / self.patch_size))
            n_v = int(np.ceil(len_v / self.patch_size))
            
            du = len_u / n_u
            dv = len_v / n_v
            
            # Material
            mat = wall.material
            # Using mean scattering/absorption
            scattering = np.mean(mat.scattering) if np.ndim(mat.scattering) > 0 else mat.scattering
            absorption = np.mean(mat.absorption) if np.ndim(mat.absorption) > 0 else mat.absorption
            
            # Create patches
            for i in range(n_u):
                for j in range(n_v):
                    # Center
                    center = v0 + (i + 0.5) * du * u_dir + (j + 0.5) * dv * v_dir
                    area = du * dv
                    
                    patch = {
                        'center': center,
                        'normal': wall.normal,
                        'area': area,
                        'absorption': absorption,
                        'scattering': scattering, # Radiosity handles diffuse energy
                        'energy': 0.0,
                        'id': len(self.patches)
                    }
                    self.patches.append(patch)
                    
        print(f"  Created {len(self.patches)} patches.")

    def _compute_view_factors(self):
        """
        Compute Form Factor Matrix F_ij (fraction of energy leaving i that hits j).
        Approximation:
        F_ij = (cos(theta_i) * cos(theta_j) * Area_j) / (pi * r^2) + Visibility
        """
        n = len(self.patches)
        print(f"Radiosity: Computing {n}x{n} view factor matrix...")
        
        self.view_factors = np.zeros((n, n), dtype=np.float32)
        
        # Pre-extract arrays for vectorization
        centers = np.array([p['center'] for p in self.patches]) # (N, 3)
        normals = np.array([p['normal'] for p in self.patches]) # (N, 3)
        areas = np.array([p['area'] for p in self.patches])     # (N,)
        
        # Loop or Vectorize?
        # N=100 -> 10k checks. Fast.
        # N=1000 -> 1M checks. Slower but okay.
        
        for i in tqdm(range(n)):
            p_i = centers[i]
            n_i = normals[i]
            
            # Vector from i to all j
            # r_vec = centers - p_i  # (N, 3)
            # But we can ignore i=j (dist=0)
            
            for j in range(n):
                if i == j:
                    continue
                    
                p_j = centers[j]
                n_j = normals[j]
                area_j = areas[j]
                
                vec = p_j - p_i
                dist2 = np.dot(vec, vec)
                dist = np.sqrt(dist2)
                
                dir_ij = vec / dist
                
                # Cosines
                cos_i = np.dot(n_i, dir_ij)
                cos_j = np.dot(n_j, -dir_ij) # Normal j opposes ray
                
                if cos_i <= 0 or cos_j <= 0:
                    continue # Facing away
                    
                # Visibility Check (Occlusion)
                # Ray cast p_i -> p_j. Check walls/furniture.
                # This is the expensive part.
                # For Convex Room (Shoebox), all facing patches see each other.
                # Assuming Convex for MVP speed.
                
                # Form factor
                # F_dAi_dAj = (cos_i * cos_j) / (pi * r^2)
                # F_i_j approx F_dAi_dAj * Area_j
                
                val = (cos_i * cos_j * area_j) / (np.pi * dist2)
                
                # Cap sum? F_ii = 0. sum(F_ij) <= 1
                self.view_factors[i, j] = val

        # Normalize rows? Sum F_ij <= 1 (Closed enclosure = 1)
        # Our approximation is point-to-area.
        # Ideally, we check sum and warn.
        row_sums = np.sum(self.view_factors, axis=1)
        print(f"  Avg row sum (should be ~1 for closed room): {np.mean(row_sums):.3f}")

    def solve(self, source, duration=1.0, time_step=0.01):
        """
        Run time-dependent radiosity.
        Returns energy decay envelopes for each patch? 
        Or reconstructs IR at receiver?
        
        To reconstruct IR:
        We need to record "Impulse Response" of energy at each patch.
        Then Receiver samples energy from all visible patches.
        """
        steps = int(duration / time_step)
        n_patches = len(self.patches)
        
        # Energy history: (Steps, N_patches)
        # Or just current/next state
        
        # Initial Energy Injection (Direct Sound -> Patches)
        # Source illuminates patches at t = dist/c
        
        # We need a histogram of energy arrival at patches?
        # Time-dependent radiosity usually buckets energy into time bins.
        
        print("Radiosity: Solving energy propagation...")
        
        # Bins
        energy_history = np.zeros((steps, n_patches))
        
        # 1. Direct Illumination (Source -> Wall Patches)
        # Ray trace or analytic source->patch
        src_pos = source.position
        
        for i in range(n_patches):
            p = self.patches[i]
            vec = p['center'] - src_pos
            dist = np.linalg.norm(vec)
            dir_sp = vec / dist
            
            # Time bin
            t_idx = int((dist / C_SOUND) / time_step)
            if t_idx >= steps:
                continue
                
            # Cosine
            cos_theta = np.dot(p['normal'], -dir_sp)
            if cos_theta <= 0:
                continue
            
            # Energy: Power * SolidAngle / Area?
            # Irradiance E = Power * cos / (4 pi r^2)
            # Total Energy received = E * Area * dt? No, impulse.
            # Total Energy Packet = Power * (Area * cos / 4pi r^2)
            
            solid_angle_factor = (p['area'] * cos_theta) / (4 * np.pi * dist**2)
            
            # Source directivity?
            gain = 1.0 # (Implement if needed)
            
            energy_received = source.power * gain * solid_angle_factor
            
            # Absorb immediately?
            # Radiosity tracks "Radiosity" (Leaving Flux) usually, or Stored Energy.
            # Let's track "Leaving Power" (Radiosity B).
            # B = rho * H (Irradiance)
            # Energy Leaving = Energy Received * (1 - alpha) * scattering?
            # Standard radiosity: Diffuse reflection.
            # rho = 1 - alpha.
            
            # If material is specular, radiosity is wrong.
            # We assume patches are diffuse.
            # So we multiply by scattering coefficient?
            # Hybrid: Specular handled by ISM. Diffuse handled here.
            # Energy entering radiosity system = Energy * scattering_coeff * (1-alpha)
            
            rho = (1.0 - p['absorption']) * p['scattering']
            
            energy_history[t_idx, i] += energy_received * rho

        # 2. Propagation (Iterative)
        # At each step, energy shoots from patches to other patches.
        # But patches have distance delays.
        # We can't use simple matrix multiplication for time-dependent.
        # We must delay energy based on distance matrix.
        
        # Precompute distance matrix (in time steps)
        dist_matrix = np.zeros((n_patches, n_patches), dtype=int)
        centers = np.array([p['center'] for p in self.patches])
        for i in range(n_patches):
            dists = np.linalg.norm(centers - centers[i], axis=1)
            dist_matrix[i, :] = (dists / C_SOUND / time_step).astype(int)
            
        # Simulation Loop
        # We iterate through time.
        # At time t, patch i radiates energy_history[t, i].
        # This energy arrives at j at t + delay_ij.
        
        # Optimization: Don't loop t. Loop "Shooting" events?
        # With many steps, this is slow.
        # But typical RIR is 1-2 sec. 100-200 steps (10ms).
        
        for t in tqdm(range(steps)):
            # Patches radiate what they have in this bin
            current_energies = energy_history[t, :].copy()
            
            # Threshold to skip empty patches
            active_indices = np.where(current_energies > 1e-9)[0]
            
            for i in active_indices:
                E_leaving = current_energies[i]
                
                # Shoot to all j
                # Vectorized adds
                
                # View factors from i: F_ij
                # Delays from i: D_ij
                
                # Energy arriving at j: E_leaving * F_ij
                # But j reflects it: * rho_j
                
                factors = self.view_factors[i, :]
                delays = dist_matrix[i, :]
                
                # Arrival times
                t_arr = t + delays
                
                # Valid targets (t_arr < steps)
                valid = t_arr < steps
                
                if not np.any(valid):
                    continue
                    
                # Update
                # We need to add to specific (t_arrival, j)
                # Vectorized add at indices is tricky in numpy if indices repeat?
                # Here j is unique per i.
                
                indices_j = np.where(valid)[0]
                times_j = t_arr[indices_j]
                
                # Reflection factors for j
                rhos = np.array([(1 - self.patches[x]['absorption']) * self.patches[x]['scattering'] for x in indices_j])
                
                energies_arriving = E_leaving * factors[indices_j] * rhos
                
                # Add
                # energy_history[times_j, indices_j] += energies_arriving
                # Numpy advanced indexing
                energy_history[times_j, indices_j] += energies_arriving
                
        return energy_history

    def collect_at_receiver(self, receiver, energy_history, time_step):
        """
        Calculate energy received at a microphone from the radiating patches.
        """
        print(f"Radiosity: Collecting at receiver {receiver.name}...")
        steps, n_patches = energy_history.shape
        
        # Receiver histogram
        # We can map to exact times or bins
        histogram = [] # (time, energy)
        
        rec_pos = receiver.position
        
        for i in range(n_patches):
            p = self.patches[i]
            
            # Check visibility/angle
            vec = rec_pos - p['center']
            dist = np.linalg.norm(vec)
            dir_pr = vec / dist
            
            cos_theta = np.dot(p['normal'], dir_pr)
            if cos_theta <= 0:
                continue
                
            # Delay
            delay_t = dist / C_SOUND
            delay_steps = int(delay_t / time_step)
            
            # Form factor Patch->Receiver
            # Area * cos / r^2 ?
            # Or Solid Angle of Patch seen by Receiver?
            # Treating patch as point source now (since it's radiating).
            # Power_leaving = Energy_history (it's stored as energy packet).
            # Intensity at Rec = Power / 2pi r^2 (Lambertian)?
            # Lambertian source: I = P/pi * cos.
            # Flux at receiver = I * A_rec / r^2 ?
            
            # Receiver is omni sphere.
            # Energy Fraction = (Area_rec / (pi * dist^2)) * cos_theta?
            # (pi dist^2 because hemisphere radiation)
            
            geom_factor = (np.pi * receiver.radius**2 * cos_theta) / (np.pi * dist**2)
            
            # Collect all time bins from this patch
            patch_energies = energy_history[:, i]
            valid_indices = np.where(patch_energies > 1e-12)[0]
            
            for t_idx in valid_indices:
                E = patch_energies[t_idx]
                arrival_time = (t_idx * time_step) + delay_t
                
                received_E = E * geom_factor
                histogram.append((arrival_time, received_E))
                
        return histogram

