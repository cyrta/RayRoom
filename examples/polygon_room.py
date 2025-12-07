import os
import sys
import matplotlib.pyplot as plt

from rayroom import Room, Source, Receiver, RayTracer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def main():
    # Define an L-shaped room using corners
    # (0,0) -> (4,0) -> (4,2) -> (2,2) -> (2,4) -> (0,4)
    corners = [
        (0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)
    ]

    room = Room.create_from_corners(corners, height=3.0)

    # Add Source in one leg of L
    source = Source("Speaker", [1, 1, 1.5])
    room.add_source(source)

    # Add Receiver in other leg of L (no direct line of sight if wall was there, but here it is open)
    # Wait, (2,2) is the inner corner. Line from (1,1) to (3,3) passes through (2,2)?
    # (1,1) -> (3,3) crosses boundary?
    # (1,1) is inside. (3,3) is inside?
    # (3,3) is outside the L-shape defined above?
    # 0-4 x, 0-2 y covers (x,y)
    # 0-2 x, 2-4 y covers (x,y)
    # (3,3) is x=3, y=3.
    # Bounds: x>2? y must be <2.
    # So (3,3) is OUTSIDE.

    # Let's pick a valid point in the other leg.
    # Leg 1: [0,4]x[0,2]
    # Leg 2: [0,2]x[2,4]
    # Point (3, 1) is in Leg 1.
    # Point (1, 3) is in Leg 2.
    # Line (3,1) to (1,3) goes through (2,2)?
    # Midpoint (2,2). Yes.
    # It grazes the corner.

    receiver = Receiver("Mic", [1, 3, 1.5], radius=0.2)
    room.add_receiver(receiver)

    # Add source
    room.sources = []  # Clear
    room.add_source(Source("Speaker", [3, 1, 1.5]))

    print("Running simulation for L-shaped room...")
    tracer = RayTracer(room)
    tracer.run(n_rays=20000, max_hops=20)

    print(f"Receiver recorded {len(receiver.energy_histogram)} hits.")

    if len(receiver.energy_histogram) > 0:
        times, energies = zip(*receiver.energy_histogram)
        plt.figure()
        plt.hist(times, bins=50, weights=energies, label="Energy")
        plt.title("L-Shaped Room Response")
        plt.savefig("l_shape_response.png")
        print("Saved l_shape_response.png")


if __name__ == "__main__":
    main()
