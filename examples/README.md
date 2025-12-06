## Comparison & Demonstrations

Below we compare the output of **RayRoom**'s various rendering engines against **PyRoomAcoustics** (with ray tracing on).

The audio samples illustrate how RayRoom captures richer late reverberation, diffuse energy, and frequency-dependent phenomena compared to standard ISM.

### PyRoomAcoustics (Baseline)

Uses standard Ray Tracing. This serves as a baseline for comparison.


https://github.com/user-attachments/assets/1d038380-1b98-4c1b-842a-4db41e154a38


### RayRoom Hybrid

Combines the Image Source Method (ISM) for early reflections with Ray Tracing for late reverberation. This approach balances accuracy and performance.


https://github.com/user-attachments/assets/90b2481c-ef0e-4a43-8631-04f9c8201498



### RayRoom Spectral (FDTD)

Utilizes a Spectral approach, combining Wave physics (FDTD) for low frequencies and Geometric methods (ISM + Ray Tracing) for high frequencies. This provides the highest fidelity across the frequency spectrum.


https://github.com/user-attachments/assets/832d1b1a-911a-4bdd-8024-c4a775ec457d



### RayRoom Radiosity

Focuses on Diffuse Energy modeling. This method is excellent for simulating the diffuse reverberation field using energy exchange between surface patches.


https://github.com/user-attachments/assets/9f75a889-ceb0-40d1-8df2-8580792f4be2



### RayRoom Small Room

Demonstrates Ray Tracing in a smaller acoustic space.


https://github.com/user-attachments/assets/e967b2c5-fa9d-4a8c-ba53-49e2e1afb995


