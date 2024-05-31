import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import numba

from simevo.utils import linmap
from simevo import min_props, max_props

class Phenotype():
    def __init__(self, genotype, min_props=min_props, max_props=max_props):
        self.min_props = min_props
        self.max_props = max_props
        # self.num_props = 5*min_props + (5+1)*(min_props-max_props)
        self.num_att = 5    # arm length, arm angle, phi, theta, rotation, [optional prop]
        self.genotype = genotype

        self.get_phenotype()

    def get_phenotype(self):
        # arm_map     = [0.05, 0.5]
        # angle_map   = [-np.pi, np.pi]
        # phi_map     = [0, np.pi/2] # Inclination
        # theta_map   = [-np.pi/2, np.pi/2] # Azimuth

        arm_map     = [0.05, 0.5]
        angle_map   = [-np.pi, np.pi]
        phi_map     = [0, 15/180*np.pi] # Inclination
        theta_map   = [-np.pi, np.pi] # Azimuth

        self.props  = []

        # Create mandatory props
        for i in range(self.min_props):
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
            dir = [np.sin(phiP)*np.cos(thetaP), np.sin(phiP)*np.sin(thetaP), -np.cos(phiP), rotP]
            # dir = [np.sin(phiP)*np.cos(thetaP+angleP), np.sin(phiP)*np.sin(thetaP+angleP), -np.cos(phiP), rotP]

            prop = {"loc": loc, "dir": dir, "constants": [7.24e-07, 8.20e-09], "wmax": 3927}
            self.props.append(prop)

        # Create optional props
        for j in range(self.max_props-self.min_props):
            armG    = self.genotype[self.min_props*5 + j*(self.num_att+1)]
            angleG  = self.genotype[self.min_props*5 + j*(self.num_att+1) + 1]
            phiG    = self.genotype[self.min_props*5 + j*(self.num_att+1) + 2]
            thetaG  = self.genotype[self.min_props*5 + j*(self.num_att+1) + 3]
            rotG    = self.genotype[self.min_props*5 + j*(self.num_att+1) + 4]
            optG    = self.genotype[self.min_props*5 + j*(self.num_att+1) + 5]

            if optG > 0:
                armP    = linmap(arm_map, armG)
                angleP  = linmap(angle_map, angleG)
                phiP    = linmap(phi_map, phiG)
                thetaP  = linmap(theta_map, thetaG)
                rotP    = "ccw" if rotG >= 0 else "cw"

                loc = [armP *np.cos(angleP), armP *np.sin(angleP), 0]
                dir = [np.sin(phiP)*np.cos(thetaP), np.sin(phiP)*np.sin(thetaP), -np.cos(phiP), rotP]
                # dir = [np.sin(phiP)*np.cos(thetaP+angleP), np.sin(phiP)*np.sin(thetaP+angleP), -np.cos(phiP), rotP]


                prop = {"loc": loc, "dir": dir, "constants": [7.24e-07, 8.20e-09], "wmax": 3927}
                self.props.append(prop)

        # Update number of props
        self.num_props = len(self.props)
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
        beam_density = 1650*0.005*0.01 # kg/m, carbon fiber plates, 5mm thickness, 10mm width

        self.mass = controller_mass + prop_mass*self.num_props

        self.cg = np.zeros(3)
        for prop in self.props:
            self.mass += beam_density*np.linalg.norm(np.array(prop["loc"])) # Add mass of carbon beam
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
            ax.plot([0, loc[1]], [0, loc[0]], "k")
            ax.scatter(loc[1], loc[0], c="k")
            if prop["dir"][-1] =="ccw":
                col = "r"
            else:
                col = "b"
            ax.plot( 0.1016/2*np.sin(np.linspace(0, 2*np.pi))+loc[1], 0.1016/2*np.cos(np.linspace(0, 2*np.pi))+loc[0], col)

            ax.text(loc[1], loc[0], f"{i}", fontsize=12, color='black')

        ax.scatter(self.cg[1], self.cg[0], s=200, marker="x", color="red")
        ax.text(self.cg[1], self.cg[0], "C.G.", fontsize=12, color='black')
        plt.xlabel("y")
        plt.ylabel("x")