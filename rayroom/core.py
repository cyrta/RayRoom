import numpy as np
from tqdm import tqdm
from .geometry import ray_plane_intersection, is_point_in_polygon, reflect_vector, random_direction_hemisphere, normalize

C_SOUND = 343.0 # m/s

class RayTracer:
    def __init__(self, room):
        self.room = room
        
    def run(self, n_rays=10000, max_hops=50, energy_threshold=1e-6):
        """
        Run the simulation.
        """
        for source in self.room.sources:
            print(f"Simulating Source: {source.name}")
            self._trace_source(source, n_rays, max_hops, energy_threshold)
            
    def _trace_source(self, source, n_rays, max_hops, energy_threshold):
        # Generate rays
        # Uniform sphere sampling
        phi = np.random.uniform(0, 2*np.pi, n_rays)
        costheta = np.random.uniform(-1, 1, n_rays)
        
        theta = np.arccos(costheta)
        r = 1.0
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        directions = np.stack((x, y, z), axis=1)
        
        # Base energy per ray (uniform distribution)
        base_energy = source.power / n_rays 
        # Assuming scalar power for now. If array, handle accordingly.
        
        # Directivity Factors
        # Calculate angle between ray_dir and source.orientation
        if hasattr(source, 'directivity') and source.directivity != "omnidirectional":
             # Dot product: cos(theta) = (a . b) / (|a||b|)
             # Directions and orientation are normalized
             cos_theta = np.dot(directions, source.orientation)
             
             if source.directivity == "cardioid":
                 # 0.5 * (1 + cos(theta))
                 gain = 0.5 * (1.0 + cos_theta)
             elif source.directivity == "subcardioid":
                 # 0.7 + 0.3 * cos(theta)
                 gain = 0.7 + 0.3 * cos_theta
             elif source.directivity == "hypercardioid":
                 # 0.25 + 0.75 * cos(theta) -> take abs for back lobe? or just pattern
                 # Standard polar pattern: |A + B cos(theta)|
                 # Hypercardioid usually A=0.25, B=0.75
                 gain = np.abs(0.25 + 0.75 * cos_theta)
             elif source.directivity == "bidirectional":
                 # |cos(theta)|
                 gain = np.abs(cos_theta)
             else:
                 gain = np.ones(n_rays)
                 
             # Normalize gain so total energy is conserved relative to uniform source?
             # Ideally sum(gain * base_energy) = source.power
             # If we want to represent focusing power:
             # current sum = sum(gain) * base_energy
             # scaling_factor = n_rays / sum(gain)
             # energy_per_ray = base_energy * gain * scaling_factor
             
             scaling_factor = n_rays / (np.sum(gain) + 1e-9)
             initial_energies = base_energy * gain * scaling_factor
        else:
             initial_energies = np.full(n_rays, base_energy)
        
        for i in tqdm(range(n_rays)):
            ray_origin = source.position
            ray_dir = directions[i]
            current_energy = initial_energies[i]
            total_dist = 0.0
            
            if current_energy < energy_threshold:
                continue
            
            for hop in range(max_hops):
                if np.sum(current_energy) < energy_threshold:
                    break
                    
                # 1. Find nearest wall/furniture intersection
                t_min = float('inf')
                hit_obj = None
                hit_normal = None
                hit_point = None
                
                # Check walls
                for wall in self.room.walls:
                    t = ray_plane_intersection(ray_origin, ray_dir, wall.vertices[0], wall.normal)
                    if t is not None and t > 1e-4 and t < t_min:
                        p = ray_origin + t * ray_dir
                        if is_point_in_polygon(p, wall.vertices, wall.normal):
                            t_min = t
                            hit_obj = wall
                            hit_normal = wall.normal
                            hit_point = p
                            
                # Check furniture
                for furn in self.room.furniture:
                    # Check all faces (naive)
                    # Bounding box check could optimize
                    for f_idx, normal in enumerate(furn.face_normals):
                        plane_pt = furn.face_planes[f_idx]
                        t = ray_plane_intersection(ray_origin, ray_dir, plane_pt, normal)
                        if t is not None and t > 1e-4 and t < t_min:
                             # Check if inside face polygon
                             face_verts = furn.vertices[furn.faces[f_idx]]
                             p = ray_origin + t * ray_dir
                             if is_point_in_polygon(p, face_verts, normal):
                                 t_min = t
                                 hit_obj = furn
                                 hit_normal = normal
                                 hit_point = p

                # 2. Check Receivers (pass-through)
                # We check if the ray segment (ray_origin -> hit_point) intersects receiver spheres
                dist_to_wall = t_min if t_min != float('inf') else 1e9
                
                for receiver in self.room.receivers:
                    # Ray-Sphere intersection
                    # |Origin + t*Dir - Center|^2 = R^2
                    oc = ray_origin - receiver.position
                    b = np.dot(oc, ray_dir)
                    c = np.dot(oc, oc) - receiver.radius**2
                    delta = b*b - c
                    
                    if delta >= 0:
                        sqrt_delta = np.sqrt(delta)
                        t1 = -b - sqrt_delta
                        t2 = -b + sqrt_delta
                        
                        # We want entry point
                        t_rx = None
                        if t1 > 1e-4 and t1 < dist_to_wall:
                            t_rx = t1
                        elif t2 > 1e-4 and t2 < dist_to_wall:
                            t_rx = t2
                            
                        if t_rx is not None:
                            # Receiver hit!
                            dist = total_dist + t_rx
                            time = dist / C_SOUND
                            receiver.record(time, current_energy)

                # 3. Handle Wall Hit
                if hit_obj is None:
                    break # Lost ray
                
                total_dist += t_min
                
                # Material interaction
                mat = hit_obj.material
                
                # Material properties (handle scalar or array)
                abs_coeff = np.mean(mat.absorption) if np.ndim(mat.absorption) > 0 else mat.absorption
                trans_coeff = np.mean(mat.transmission) if np.ndim(mat.transmission) > 0 else mat.transmission
                scat_coeff = np.mean(mat.scattering) if np.ndim(mat.scattering) > 0 else mat.scattering

                # Energy loss due to absorption
                current_energy *= (1.0 - abs_coeff)
                
                # Determine fate: Transmit or Reflect?
                # Probability of transmission given we didn't absorb: T / (1 - A)
                
                if abs_coeff >= 1.0 - 1e-6:
                    break # Fully absorbed

                prob_transmission = trans_coeff / (1.0 - abs_coeff)
                
                if np.random.random() < prob_transmission:
                     # Transmit
                     # Simplified: No refraction, just pass through
                     ray_origin = hit_point + ray_dir * 1e-3
                else:
                     # Reflect
                     if np.random.random() < scat_coeff:
                         # Diffuse
                         ray_dir = random_direction_hemisphere(hit_normal)
                     else:
                         # Specular
                         ray_dir = reflect_vector(ray_dir, hit_normal)
                     
                     ray_origin = hit_point + hit_normal * 1e-3
