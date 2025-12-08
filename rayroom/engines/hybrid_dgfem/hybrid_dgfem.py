import numpy as np
from scipy.signal import butter, lfilter
from dataclasses import dataclass, field
from typing import Optional

from ...room.base import Room
from ..dgfem.dgfem import DGFEMSolver, DGFEMConfig
from ..ism.ism import ImageSourceEngine
from ..raytracer.core import RayTracer


@dataclass
class HybridDGFEMConfig:
    """Configuration for the Hybrid DG-FEM solver."""
    crossover_freq: float = 300.0  # Hz
    transition_time: float = 0.1  # Seconds for ISM to Raytracer fade
    dgfem_config: DGFEMConfig = field(default_factory=DGFEMConfig)
    ism_config: dict = field(default_factory=lambda: {'max_order': 5})
    raytracer_config: dict = field(default_factory=lambda: {'num_rays': 10000})


class HybridDGFEMMethod:
    """
    A hybrid room acoustics simulation method combining a wave-based solver (DG-FEM)
    for low frequencies and geometrical acoustics methods (ISM and Ray Tracing)
    for high frequencies.
    """

    def __init__(self, room: Room, config: Optional[HybridDGFEMConfig] = None):
        """
        Initializes the HybridDGFEMMethod with a given room and configuration.

        Parameters
        ----------
        room : Room
            The room to be simulated.
        config : HybridDGFEMConfig, optional
            A dictionary of configuration parameters.
        """
        self.room = room
        self.config = config or HybridDGFEMConfig()

        print("Initializing Hybrid DG-FEM sub-solvers...")
        self.dgfem_solver = DGFEMSolver(room, self.config.dgfem_config)
        self.ism_solver = ImageSourceEngine(
            room,
            temperature=self.config.ism_config.get("temperature", 20.0),
            humidity=self.config.ism_config.get("humidity", 50.0),
        )
        self.raytracer_solver = RayTracer(
            room,
            temperature=self.config.raytracer_config.get("temperature", 20.0),
            humidity=self.config.raytracer_config.get("humidity", 50.0)
        )
        print("Sub-solvers initialized.")

    def _reset_receivers(self):
        """Resets the state of all receivers in the room."""
        for receiver in self.room.receivers:
            # Handle mono receiver
            if hasattr(receiver, 'amplitude_histogram'):
                receiver.amplitude_histogram = []
            # Handle ambisonic receiver
            if hasattr(receiver, 'w_histogram'):
                receiver.w_histogram = []
                receiver.x_histogram = []
                receiver.y_histogram = []
                receiver.z_histogram = []

    def _get_rir_from_receivers(self, duration: float) -> np.ndarray:
        """
        Generates a single RIR from the first receiver in the room.
        Assumes the receiver has been populated by a simulation engine.
        """
        if not self.room.receivers:
            return np.zeros(int(duration * self.room.fs))

        receiver = self.room.receivers[0]
        num_samples = int(duration * self.room.fs)
        rir = np.zeros(num_samples)

        # Determine which histogram to use based on receiver type
        histogram = []
        if hasattr(receiver, 'amplitude_histogram'): # Mono receiver
            histogram = receiver.amplitude_histogram
        elif hasattr(receiver, 'w_histogram'): # Ambisonic, use W channel for now
            histogram = receiver.w_histogram

        for time, amplitude in histogram:
            sample_idx = int(time * self.room.fs)
            if 0 <= sample_idx < num_samples:
                rir[sample_idx] += amplitude

        return rir

    def compute_rir(self, duration: float = 1.0) -> np.ndarray:
        """
        Computes the Room Impulse Response (RIR) using the hybrid method.

        Returns
        -------
        np.ndarray
            The computed Room Impulse Response.
        """
        print("\nComputing RIR with Hybrid DG-FEM Method...")

        # 1. Compute low-frequency RIR with DG-FEM
        print("\n--- Running DG-FEM for low frequencies ---")
        rir_lf = self.dgfem_solver.compute_rir(duration=duration)

        # 2. Compute high-frequency RIR with geometrical methods
        print("\n--- Running Geometrical Acoustics for high frequencies ---")
        
        # Reset receivers and run ISM
        self._reset_receivers()
        for source in self.room.sources:
            self.ism_solver.run(
                source,
                max_order=self.config.ism_config.get("max_order", 5)
            )
        rir_ism = self._get_rir_from_receivers(duration)

        # Reset receivers and run RayTracer
        self._reset_receivers()
        for source in self.room.sources:
            self.raytracer_solver.run(
                source,
                n_rays=self.config.raytracer_config.get("num_rays", 10000)
            )
        rir_ray = self._get_rir_from_receivers(duration)
        
        rir_hf = self._combine_ism_raytracer(rir_ism, rir_ray)

        # 3. Combine low and high frequency RIRs
        print("\n--- Combining low and high frequency RIRs ---")
        rir_final = self._combine_lf_hf(rir_lf, rir_hf)

        print("\nHybrid DG-FEM simulation complete.")
        return rir_final

    def _combine_ism_raytracer(self, rir_ism: np.ndarray, rir_raytracer: np.ndarray) -> np.ndarray:
        """
        Combines the ISM and Ray Tracing RIRs with a smooth transition in the time domain.
        This is suitable for the high-frequency part of the spectrum.
        """
        sample_rate = self.room.fs
        transition_sample = int(self.config.transition_time * sample_rate)

        len_ism = len(rir_ism)
        len_ray = len(rir_raytracer)
        len_final = max(len_ism, len_ray)

        rir_ism = np.pad(rir_ism, (0, len_final - len_ism))
        rir_raytracer = np.pad(rir_raytracer, (0, len_final - len_ray))

        fade_len = int(0.01 * sample_rate)  # 10 ms fade
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = np.linspace(1, 0, fade_len)

        mask_ism = np.ones(len_final)
        mask_ray = np.zeros(len_final)

        start_fade = transition_sample - fade_len // 2
        end_fade = transition_sample + fade_len // 2

        if start_fade < 0:
            start_fade = 0
            end_fade = fade_len
        if end_fade > len_final:
            end_fade = len_final
            start_fade = len_final - fade_len

        mask_ism[start_fade:end_fade] = fade_out
        mask_ism[end_fade:] = 0
        mask_ray[start_fade:end_fade] = fade_in
        mask_ray[end_fade:] = 1

        rir_combined = rir_ism * mask_ism + rir_raytracer * mask_ray
        print("Combined ISM and Ray Tracing for early/late reflections.")
        return rir_combined

    def _combine_lf_hf(self, rir_lf: np.ndarray, rir_hf: np.ndarray) -> np.ndarray:
        """
        Combines the low-frequency (DG-FEM) and high-frequency (Geometrical) RIRs
        using a crossover filter.
        """
        crossover_freq = self.config.crossover_freq
        fs = self.room.fs
        # Design a 4th-order Linkwitz-Riley crossover filter
        nyquist = 0.5 * fs
        low_cutoff = crossover_freq / nyquist
        # Low-pass filter for DG-FEM RIR
        b_lp, a_lp = butter(4, low_cutoff, btype='low')
        # High-pass filter for Geometrical RIR
        b_hp, a_hp = butter(4, low_cutoff, btype='high')

        # Pad shorter RIR to match length of longer RIR
        if len(rir_lf) > len(rir_hf):
            rir_hf = np.pad(rir_hf, (0, len(rir_lf) - len(rir_hf)))
        elif len(rir_hf) > len(rir_lf):
            rir_lf = np.pad(rir_lf, (0, len(rir_hf) - len(rir_lf)))

        # Apply filters
        rir_lf_filtered = lfilter(b_lp, a_lp, rir_lf)
        rir_hf_filtered = lfilter(b_hp, a_hp, rir_hf)
        rir_final = rir_lf_filtered + rir_hf_filtered
        print(f"Combined LF and HF responses at {crossover_freq} Hz crossover.")
        return rir_final
