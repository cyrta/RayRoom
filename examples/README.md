## Comparison & Demonstrations

Below we compare the output of **RayRoom**'s various rendering engines against **PyRoomAcoustics** (with ray tracing on).

The audio samples illustrate how RayRoom captures richer late reverberation, diffuse energy, and frequency-dependent phenomena compared to standard ISM.

### PyRoomAcoustics (Baseline)

Uses standard Ray Tracing. This serves as a baseline for comparison.

https://github.com/user-attachments/assets/a5c78ee2-2e4b-43b2-aeb8-5315d720d094

### RayRoom Hybrid

Combines the Image Source Method (ISM) for early reflections with Ray Tracing for late reverberation. This approach balances accuracy and performance.

https://github.com/user-attachments/assets/1695b957-fdaf-48b2-846d-78cf1a8a425f

### RayRoom Spectral (FDTD)

Utilizes a Spectral approach, combining Wave physics (FDTD) for low frequencies and Geometric methods (ISM + Ray Tracing) for high frequencies. This provides the highest fidelity across the frequency spectrum.

https://github.com/user-attachments/assets/148f07b0-f041-44f4-abe8-b7ca2780a4e9

### RayRoom Radiosity

Focuses on Diffuse Energy modeling. This method is excellent for simulating the diffuse reverberation field using energy exchange between surface patches.

https://github.com/user-attachments/assets/1084cc69-9597-4522-b40c-09758bef3f5c

### RayRoom Small Room

Demonstrates Ray Tracing in a smaller acoustic space.

https://github.com/user-attachments/assets/81080910-9084-4c62-bf27-41f130f15366
