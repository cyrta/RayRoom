# Hybrid Geometric Renderer Methodology

The Hybrid Renderer in RayRoom combines two powerful techniques from geometric acoustics: the **Image Source Method (ISM)** for early, specular reflections and **Stochastic Ray Tracing** for late, diffuse reverberation. This hybrid approach leverages the strengths of each method to produce a complete and acoustically accurate Room Impulse Response (RIR) efficiently.

- **Early Reflections (ISM):** The first part of the RIR is dominated by distinct, high-energy specular reflections. The ISM is a deterministic method that is perfectly suited to find these paths by mirroring the sound source across room boundaries. It is computationally precise but becomes exponentially complex as the reflection order increases.

- **Late Reverberation (Ray Tracing):** The later part of the RIR consists of a dense, decaying field of reflections that are best modeled as a stochastic process. Ray tracing is ideal for this, simulating the behavior of thousands of acoustic rays as they lose energy and become more diffuse over time. It is computationally efficient for high reflection orders.

The final RIR is created by cross-fading from the ISM output to the ray tracing output at a determined transition time or reflection order.

## Core Equations & Principles

### 1. Image Source Method (ISM)

The position of an image source \\( \mathbf{s}' \\) of order \\( N \\) is found by recursively reflecting the source position \\( \mathbf{s} \\) across the planes of the room walls. For a single reflection across a wall with normal \\( \mathbf{n} \\) and a point \\( \mathbf{p} \\) on the wall, the image source position is:

\[ \mathbf{s}' = \mathbf{s} - 2 ((\mathbf{s} - \mathbf{p}) \cdot \mathbf{n}) \mathbf{n} \]

The distance from an image source to the receiver \\( \mathbf{r} \\) gives the travel time for a specific reflection path. The pressure contribution \\( p(t) \\) of each image source is attenuated by distance and wall absorption, and its arrival is delayed by the path length.

### 2. Stochastic Ray Tracing

Each ray carries a portion of the source's energy. When a ray hits a surface, its energy \\( E_{i} \\) is reduced by the wall's absorption coefficient \\( \alpha \\). The reflected energy \\( E_{r} \\) is:

\[ E_r = E_i (1 - \alpha) \]

The direction of the reflected ray is determined by a combination of specular reflection (using Snell's Law) and diffuse reflection (scattering), governed by the material's scattering coefficient \\( s \\). A random number determines whether the reflection is specular or diffuse. For diffuse reflections, the new direction is often chosen from a cosine-weighted distribution over the hemisphere (Lambert's Law).

## Implementation Details

- **Transition:** The renderer combines the RIR from ISM (up to a user-defined `ism_order`) with the RIR from the ray tracer for the late reverberation.
- **Air Absorption:** The energy loss due to air is modeled according to the **ISO 9613-1** standard, which accounts for frequency, temperature, humidity, and pressure.
- **Material Properties:** Wall materials have frequency-dependent absorption \\( \alpha(f) \\) and scattering \\( s(f) \\) coefficients, allowing for more realistic acoustic modeling.

## Seminal Papers

1.  **Allen, J. B., & Berkley, D. A. (1979).** *Image method for efficiently simulating small-room acoustics*. The Journal of the Acoustical Society of America, 65(4), 943-950.

```bibtex
@article{allen1979image,
  title={Image method for efficiently simulating small-room acoustics},
  author={Allen, Jont B and Berkley, David A},
  journal={The Journal of the Acoustical Society of America},
  volume={65},
  number={4},
  pages={943--950},
  year={1979},
  publisher={Acoustical Society of America}
}
```

2.  **Krokstad, A., Strøm, S., & Sørsdal, S. (1968).** *Calculating the acoustical room response by the use of a ray tracing technique*. Journal of Sound and Vibration, 8(1), 118-125.

```bibtex
@article{krokstad1968calculating,
  title={Calculating the acoustical room response by the use of a ray tracing technique},
  author={Krokstad, Asbj{\o}rn and Strom, Staffan and S{\o}rsdal, Svein},
  journal={Journal of Sound and Vibration},
  volume={8},
  number={1},
  pages={118--125},
  year={1968},
  publisher={Elsevier}
}
```

3.  **Vorländer, M. (1989).** *Simulation of the transient and steady-state sound propagation in rooms using a new combined ray-tracing/image-source algorithm*. The Journal of the Acoustical Society of America, 86(1), 172-178.

```bibtex
@article{vorlander1989simulation,
  title={Simulation of the transient and steady-state sound propagation in rooms using a new combined ray-tracing/image-source algorithm},
  author={Vorl{\"a}nder, Michael},
  journal={The Journal of the Acoustical Society of America},
  volume={86},
  number={1},
  pages={172--178},
  year={1989},
  publisher={Acoustical Society of America}
}
```
