import numpy as np
import matplotlib.pyplot as plt
from dronehover.optimization import Hover

from simevo.drone import Drone

num_propellers = 5
num_attributes = 5  # arm length, arm angle, phi, theta, ccw/cw

drone = Drone(num_propellers)
drone.get_genotype()
drone.get_phenotype()

sim = Hover(drone)

sim.compute_hover()

fig, ax = plt.subplots()
ax.set_aspect("equal", "box")
for prop in drone.props:
    loc = prop["loc"]
    ax.plot([0, loc[0]], [0, loc[1]], "k")
    ax.scatter(loc[0], loc[1], c="k")
    if prop["dir"][-1] =="ccw":
        col = "r"
    else:
        col = "b"
    ax.plot(0.1016/2*np.cos(np.linspace(0, 2*np.pi))+loc[0], 0.1016/2*np.sin(np.linspace(0, 2*np.pi))+loc[1], col)

ax.scatter(drone.cg[0], drone.cg[1], s=200, marker="o")

plt.show()


