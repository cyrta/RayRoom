# Image Source Method (ISM) Methodology

The Image Source Method (ISM) is a deterministic geometric acoustics technique used to find the exact paths of early specular reflections in a room. Unlike stochastic methods like ray tracing, ISM is not based on random sampling. Instead, it computes the precise geometric paths that sound takes from a source to a receiver by mirroring the source across each of the room's surfaces.

This method is particularly powerful for modeling the early part of a Room Impulse Response (RIR), which is perceptually dominated by strong, discrete echoes. The timing and direction of these early reflections are crucial for spatial perception. ISM is computationally exact for simple shoebox-shaped rooms but becomes more complex with irregular geometries and higher reflection orders.

## Core Equations & Principles

The fundamental principle of ISM is that a specular reflection path from a source $s$ to a receiver $r$ off a planar surface is equivalent to a straight-line path from a "virtual" or "image" source $s'$ to the receiver.

### 1. Image Source Calculation

An image source is found by reflecting the position of the real source across a planar wall. For a source at position $s$ and a wall represented by a point $p$ on its plane and a normal vector $n$, the first-order image source $s'$ is located at:

$\mathbf{s}' = \mathbf{s} - 2 ((\mathbf{s} - \mathbf{p}) \cdot \mathbf{n}) \mathbf{n}$

Higher-order reflections are found by recursively reflecting these image sources across other walls. An image source of order $N$ represents a sound path that has undergone $N$ reflections.

### 2. Path Validation and RIR Construction

Once an image source is created, a straight line is drawn from it to the receiver. For this path to be valid, this line segment must intersect the *actual* wall panel that created the image source (and for higher orders, all parent panels in the reflection sequence).

If the path is valid, its contribution to the RIR is calculated:
-   **Delay:** The arrival time $t$ is determined by the total path length $d$ from the image source to the receiver and the speed of sound $c$: $t = d / c$.
-   **Amplitude:** The initial amplitude is attenuated by geometric spreading ($1/d$) and the cumulative absorption of all the walls involved in the reflection path. The pressure $p$ is attenuated by the product of the reflection coefficients $\beta_k = \sqrt{1 - \alpha_k}$ of each wall $k$:

    $A = \frac{1}{d} \prod_{k=1}^{N} \beta_k$

## Implementation Details

-   **Maximum Order:** The simulation is typically run up to a user-defined maximum reflection order (`ism_order`). The number of potential image sources grows exponentially with the order, making high-order calculations computationally expensive.
-   **Visibility Check:** A crucial part of the algorithm is the visibility check, which ensures that the path from an image source to the receiver is physically plausible and not obstructed.
-   **Frequency-Dependence:** Wall absorption is frequency-dependent. The simulation is often run independently for different frequency bands, and the resulting impulse responses are combined to produce the final broadband RIR.

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

2.  **Lehmann, E. A., & Johansson, A. M. (2008).** *Prediction of energy decay in room impulse responses simulated with an image-source model*. The Journal of the Acoustical Society of America, 124(1), 269-277.

```bibtex
@article{lehmann2008prediction,
  title={Prediction of energy decay in room impulse responses simulated with an image-source model},
  author={Lehmann, Eric A and Johansson, Anders M},
  journal={The Journal of the Acoustical Society of America},
  volume={124},
  number={1},
  pages={269--277},
  year={2008},
  publisher={AIP Publishing}
}
```

3.  **Borish, J. (1984).** *Extension of the image model to arbitrary polyhedra*. The Journal of the Acoustical Society of America, 75(6), 1827-1836.

```bibtex
@article{borish1984extension,
  title={Extension of the image model to arbitrary polyhedra},
  author={Borish, Jeffrey},
  journal={The Journal of the Acoustical Society of America},
  volume={75},
  number={6},
  pages={1827--1836},
  year={1984},
  publisher={Acoustical Society of America}
}
```
