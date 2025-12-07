import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from rayroom import Room, Source, Receiver, Person, RayTracer, get_material

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def main():
    # 1. Create Room (Shoebox 5m x 4m x 3m)
    # Different materials for walls
    mats = {
        "floor": get_material("wood"),
        "ceiling": get_material("plaster"),
        "front": get_material("brick"),
        "back": get_material("brick"),
        "left": get_material("concrete"),
        "right": get_material("glass")
    }

    room = Room.create_shoebox([5, 4, 3], materials=mats)

    # 2. Add Objects
    # Source at (1, 1, 1.5)
    source = Source("Speaker", [1, 1, 1.5], power=1.0)
    room.add_source(source)

    # Receiver (Microphone) at (4, 3, 1.5)
    receiver = Receiver("Mic", [4, 3, 1.5], radius=0.2)
    room.add_receiver(receiver)

    # Person (blocker) at (2.5, 2, 0)
    person = Person("Human", [2.5, 2, 0])
    room.add_furniture(person)

    # Plot Room BEFORE Simulation (Check geometry)
    print("Saving room visualization...")
    room.plot("room_layout.png", show=False)

    # 3. Run Simulation
    tracer = RayTracer(room)
    print("Starting simulation...")
    tracer.run(n_rays=20000, max_hops=30)

    # 4. Analyze Results
    print(f"Receiver recorded {len(receiver.energy_histogram)} hits.")

    if len(receiver.energy_histogram) > 0:
        times, energies = zip(*receiver.energy_histogram)
        times = np.array(times)
        energies = np.array(energies)

        # Plot Impulse Response (Histogram)
        plt.figure(figsize=(10, 6))
        plt.hist(times, bins=50, weights=energies, alpha=0.7, label="Energy")
        plt.xlabel("Time (s)")
        plt.ylabel("Energy")
        plt.title("Room Impulse Response (Energy Time Curve)")
        plt.grid(True)
        plt.savefig("impulse_response.png")
        print("Saved impulse_response.png")
    else:
        print("No energy received at microphone.")


if __name__ == "__main__":
    main()
