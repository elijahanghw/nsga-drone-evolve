import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import numba

from simevo.utils import linmap

class Phenotype():
    def __init__(self, genotype, num_attributes=5):
        self.num_props = int(len(genotype)/num_attributes)
        self.num_att = num_attributes
        self.genotype = genotype

        self.get_phenotype()

    def get_phenotype(self):
        arm_map     = [0.05, 0.3]
        angle_map   = [-np.pi, np.pi]
        phi_map     = [0, 15/180*np.pi] # Inclination
        theta_map   = [-np.pi, np.pi] # Azimuth

        self.props  = []

        for i in range(self.num_props):
            armG    = self.genotype[i*self.num_att]
            angleG  = self.genotype[i*self.num_att + 1]
            phiG    = self.genotype[i*self.num_att + 2]
            thetaG  = self.genotype[i*self.num_att + 3]
            rotG    = self.genotype[i*self.num_att + 4]

            armP    = linmap(arm_map, armG)
            angleP  = linmap(angle_map, angleG)
            phiP    = linmap(phi_map, phiG)
            thetaP  = linmap(theta_map, thetaG)
            rotP    = "ccw" if np.sign(rotG) >= 0 else "cw"

            loc = [armP *np.cos(angleP), armP *np.sin(angleP), 0]
            dir = [-np.sin(phiP)*np.cos(thetaP), -np.sin(phiP)*np.sin(thetaP), -np.cos(phiP), rotP]

            prop = {"loc": loc, "dir": dir, "constants": [7.24e-07, 8.20e-09], "wmax": 3927}
            self.props.append(prop)

        scale = self.adjust_scale()

        for i in range(self.num_props):
            self.props[i]["loc"][0] *=  scale
            self.props[i]["loc"][1] *=  scale
            self.props[i]["loc"][2] *=  scale

        self.get_inertia()

    def adjust_scale(self):
        scale = 1
        prop_size = 0.1016 + 0.02    # 4 inch diameter + extra tolerance
        for i in range(self.num_props):
            for j in range(i+1,self.num_props):
                loc_i = self.props[i]["loc"]
                loc_i = np.array(loc_i)
                loc_j = self.props[j]["loc"]
                loc_j = np.array(loc_j)

                dist = np.linalg.norm(loc_i - loc_j)
                
                if dist < prop_size and dist < prop_size/scale:
                    scale = prop_size/dist

        return scale
    
    def get_inertia(self):
        controller_mass = 0.3 # based on 4S, 3200 mAh lipo
        prop_mass = 0.018 # based on 1806 BLDC

        self.mass = controller_mass + prop_mass*self.num_props

        self.cg = np.zeros(3)
        for prop in self.props:
            self.cg += prop_mass/self.mass * np.array(prop["loc"])

        self.Ix = norm(np.cross(np.array([1,0,0]),self.cg))**2 * controller_mass
        self.Iy = norm(np.cross(np.array([0,1,0]),self.cg))**2 * controller_mass
        self.Iz = norm(np.cross(np.array([0,0,1]),self.cg))**2 * controller_mass

        for prop in self.props:
            r = np.asarray(prop["loc"]) - self.cg
            self.Ix += norm(np.cross(np.array([1,0,0]),r))**2  * prop_mass
            self.Iy += norm(np.cross(np.array([1,0,0]),r))**2 * prop_mass
            self.Iz += norm(np.cross(np.array([1,0,0]),r))**2 * prop_mass

        self.cg = self.cg.tolist()

    def plot_drone(self):
        fig, ax = plt.subplots()
        ax.set_aspect("equal", "box")
        for i, prop in enumerate(self.props):
            loc = prop["loc"]
            ax.plot([0, loc[0]], [0, loc[1]], "k")
            ax.scatter(loc[0], loc[1], c="k")
            if prop["dir"][-1] =="ccw":
                col = "r"
            else:
                col = "b"
            ax.plot(0.1016/2*np.cos(np.linspace(0, 2*np.pi))+loc[0], 0.1016/2*np.sin(np.linspace(0, 2*np.pi))+loc[1], col)

            ax.text(loc[0], loc[1], f"{i}", fontsize=12, color='black')

        ax.scatter(self.cg[0], self.cg[1], s=200, marker="x", color="red")
        ax.text(self.cg[0], self.cg[1], "C.G.", fontsize=12, color='black')

        plt.grid()
        plt.show()


        


