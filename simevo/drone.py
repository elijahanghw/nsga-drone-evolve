import numpy as np
from numpy.linalg import norm
import numba

from simevo.utils import linmap

class Drone():
    def __init__(self, num_propellers, num_attributes=5):
        self.num_props = num_propellers
        self.num_att = num_attributes


    def get_genotype(self):
        self.genotype = np.random.uniform(low=-1, high=1, size=self.num_props*self.num_att)


    def get_phenotype(self):
        arm_map     = [0.05, 0.3]
        angle_map   = [-np.pi, np.pi]
        phi_map     = [0, 15/180*np.pi] # Inclination
        theta_map   = [-15/180*np.pi, 15/180*np.pi] # Azimuth

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
            dir = [np.sin(phiP)*np.cos(thetaP), np.sin(phiP)*np.sin(thetaP), np.cos(phiP), rotP]

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

        


