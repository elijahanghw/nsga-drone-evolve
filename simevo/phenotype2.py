import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dronehover.bodies.custom_bodies import Custombody

from simevo.utils import linmap, quantmap
from simevo import min_props, max_props

class Phenotype2():
    def __init__(self, genotype, min_props=min_props, max_props=max_props):
        self.min_props = min_props
        self.max_props = max_props
        # self.num_props = 5*min_props + (5+1)*(min_props-max_props)
        self.num_att = 3    # arm length, arm angle, rotation, [optional prop]
        self.genotype = genotype

        self.generate_props()

    def generate_props(self):

        arm_map     = [0.08, 0.3]
        angle_map   = [-np.pi, np.pi]
        phi_map     = [0, 15/180*np.pi] # Inclination
        theta_map   = [-np.pi, np.pi] # Azimuth
        size_map    = [4, 5, 6] # Prop size

        self.props  = []

        # Create mandatory props
        for i in range(self.min_props):
            armG    = self.genotype[i*self.num_att]
            angleG  = self.genotype[i*self.num_att + 1]
            rotG    = self.genotype[i*self.num_att + 2]

            armP    = linmap(arm_map, armG)
            angleP  = linmap(angle_map, angleG)
            rotP    = "ccw" if np.sign(rotG) >= 0 else "cw"

            loc = [armP *np.cos(angleP), armP *np.sin(angleP), 0]
            dir = [0, 0, -1, rotP]

            prop = {"loc": loc, "dir": dir, "propsize": 5}
            self.props.append(prop)

        # Create optional props
        for j in range(self.max_props-self.min_props):
            armG    = self.genotype[self.min_props*3 + j*(self.num_att+1)]
            angleG  = self.genotype[self.min_props*3 + j*(self.num_att+1) + 1]
            rotG    = self.genotype[self.min_props*3 + j*(self.num_att+1) + 2]
            optG    = self.genotype[self.min_props*3 + j*(self.num_att+1) + 3]
            

            if optG > 0:
                armP    = linmap(arm_map, armG)
                angleP  = linmap(angle_map, angleG)
                rotP    = "ccw" if rotG >= 0 else "cw"

                loc = [armP *np.cos(angleP), armP *np.sin(angleP), 0]
                dir = [0, 0, -1, rotP]

                prop = {"loc": loc, "dir": dir, "propsize": 5}
                self.props.append(prop)
        # Update number of props
        self.num_props = len(self.props)
        self.adjust_scale()

        self.drone = Custombody(self.props)

        # self.get_props()
        # self.get_inertia()

    def adjust_scale(self):
        scale = 1
        # prop_size = 0.1016 + 0.02    # 4 inch diameter + extra tolerance
        for i in range(self.num_props):
            for j in range(i+1,self.num_props):
                size_i = self.props[i]["propsize"] * 0.0254
                loc_i = self.props[i]["loc"]
                loc_i = np.array(loc_i)

                size_j = self.props[j]["propsize"] * 0.0254
                loc_j = self.props[j]["loc"]
                loc_j = np.array(loc_j)

                dist = np.linalg.norm(loc_i - loc_j)
                min_dist = size_i/2 + size_j/2 + 0.01
                
                if dist < min_dist and min_dist/dist > scale:
                    scale = min_dist/dist

        for i in range(self.num_props):
            self.props[i]["loc"][0] *=  scale
            self.props[i]["loc"][1] *=  scale
            self.props[i]["loc"][2] *=  scale

        # return scale
    

    def plot_drone(self, quiver=False):
        fig, ax = plt.subplots()
        
        for i, prop in enumerate(self.props):
            size_label = prop["propsize"]
            size = prop["propsize"] * 0.0254
            loc = prop["loc"]
            dir = prop["dir"]
            ax.plot([0, loc[1]], [0, loc[0]], "k")
            ax.scatter(loc[1], loc[0], c="k")
            if quiver:
                ax.arrow(loc[1], loc[0], dir[1], dir[0], color="green", linestyle="--")
            if prop["dir"][-1] =="ccw":
                col = "r"
            else:
                col = "b"
            ax.plot(size/2*np.sin(np.linspace(0, 2*np.pi))+loc[1], size/2*np.cos(np.linspace(0, 2*np.pi))+loc[0], col)
            

            ax.text(loc[1], loc[0], f"{size_label}", fontsize=12, color='black')

        ax.scatter(self.drone.cg[1], self.drone.cg[0], s=200, marker="x", color="red")
        ax.text(self.drone.cg[1], self.drone.cg[0], "C.G.", fontsize=12, color='black')

        ccw = Line2D([0], [0], color='r', label="CCW")
        cw = Line2D([0], [0], color='b', label="CCW")
        arrow = Line2D([0], [0], linestyle="--", color="green")
        
        if quiver:
            ax.legend([ccw, cw, arrow], ["CCW", "CW", "Direction"], loc="best")
        else:
            ax.legend([ccw, cw, arrow], ["CCW", "CW"], loc="best")
        ax.set_xlabel("y")
        ax.set_ylabel("x")

        ax.set_aspect("equal", "box")