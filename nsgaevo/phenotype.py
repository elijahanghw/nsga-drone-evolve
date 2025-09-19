import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dronehover.bodies.custom_bodies import Custombody

from nsgaevo.utils import linmap, quantmap, vector_rotation_matrix
from nsgaevo import min_props, max_props

class PhenotypeBase:
    def adjust_scale(self):
        scale = 1
        for i in range(self.num_props):
            for j in range(i+1,self.num_props):
                size_i = self.props[i]["propsize"] * 0.0254
                loc_i = self.props[i]["loc"]
                loc_i = np.array(loc_i)

                size_j = self.props[j]["propsize"] * 0.0254
                loc_j = self.props[j]["loc"]
                loc_j = np.array(loc_j)

                dist = np.linalg.norm(loc_i - loc_j)
                min_dist = size_i/2 + size_j/2 + 0.02
                
                if dist < min_dist and min_dist/dist > scale:
                    scale = min_dist/dist

        for i in range(self.num_props):
            self.props[i]["loc"][0] *=  scale
            self.props[i]["loc"][1] *=  scale
            self.props[i]["loc"][2] *=  scale

        
    def plot_drone(self, legend=True, projection="3d"):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for i, prop in enumerate(self.props):
            size_label = prop["propsize"]
            size = prop["propsize"] * 0.0254
            loc = np.array(prop["loc"])
            dir = np.array(prop["dir"][0:3])
            ax.plot([0, loc[0]], [0, loc[1]], [0, loc[2]], "k")
            ax.scatter(loc[0], loc[1], loc[2], c="k")
            
            if prop["dir"][-1] =="cw":
                col = "r"
                ls = "-"
            else:
                col = "b"
                ls = "--"
            
            theta = np.linspace(0, 2 * np.pi, 100)
            circle = np.vstack((np.cos(theta), np.sin(theta), np.zeros_like(theta))) * (size / 2)
            z_axis = np.array([0, 0, 1])
            axis = np.cross(z_axis, dir)
            if norm(axis) > 1e-6:
                axis = axis / norm(axis)
                angle = np.arccos(np.clip(np.dot(z_axis, dir), -1.0, 1.0))
                K = np.array([[0, -axis[2], axis[1]],
                            [axis[2], 0, -axis[0]],
                            [-axis[1], axis[0], 0]])
                R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                circle = R @ circle
                
            x = circle[0, :] + loc[0]
            y = circle[1, :] + loc[1]
            z = circle[2, :] + loc[2]

            ax.plot(x, y, z, color=col, linestyle=ls)

        ccw = Line2D([0], [0], color='r', label="CW")
        cw = Line2D([0], [0], color='b', linestyle="--", label="CCW")
        arrow = Line2D([0], [0], linestyle=":", color="green")
        
        if legend:
            ax.legend([ccw, cw], ["CW", "CCW"], bbox_to_anchor=(1, 0.5))
            
        # Compute data bounds
        x_vals = [prop["loc"][0] for prop in self.props]
        y_vals = [prop["loc"][1] for prop in self.props]
        z_vals = [prop["loc"][2] for prop in self.props]

        # Add origin (0, 0, 0) to make sure center is included
        x_vals.append(0)
        y_vals.append(0)
        z_vals.append(0)

        # Get min/max ranges
        x_range = max(x_vals) - min(x_vals)
        y_range = max(y_vals) - min(y_vals)
        z_range = max(z_vals) - min(z_vals)
        max_range = max(x_range, y_range, z_range)

        # Compute midpoints
        x_mid = (max(x_vals) + min(x_vals)) / 2
        y_mid = (max(y_vals) + min(y_vals)) / 2
        z_mid = (max(z_vals) + min(z_vals)) / 2

        if projection == '3d':
            ax.set_box_aspect([1, 1, -1])
            ax.set_xlim(x_mid - max_range/2 - size/2, x_mid + max_range/2 + size/2)
            ax.set_ylim(y_mid - max_range/2 - size/2, y_mid + max_range/2 + size/2)
            ax.set_zlim(z_mid - max_range/2 - size/2, z_mid + max_range/2 + size/2) 
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_zlabel("z (m)")
        elif projection == '2d':
            ax.view_init(elev=90, azim=-90)  # Top-down view: looking straight down Z-axis
            ax.set_box_aspect([1, 1, -1])
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            
        plt.tight_layout()
        
    def plot_drone_2d(self, ax=None, quiver=False, legend=True, limits=None, axis_labels=True):
        if ax is None:
            fig, ax = plt.subplots()
        
        for i, prop in enumerate(self.props):
            size_label = prop["propsize"]
            size = prop["propsize"] * 0.0254
            loc = np.array(prop["loc"])
            dir = np.array(prop["dir"][0:3])
            ax.plot([0, loc[1]], [0, loc[0]], "k")
            ax.scatter(loc[1], loc[0], c="k")
            if quiver:
                dir_norm = dir[0:2] / norm(dir[0:2]) * 0.1
                ax.arrow(loc[1], loc[0], dir_norm[1], dir_norm[0], color="green", linestyle=":", head_width=0.01, length_includes_head=True)
                text_loc = dir_norm * 0.3
                ax.text(loc[1]+dir_norm[1]+text_loc[1], loc[0]+dir_norm[0]+text_loc[0], f"{np.arccos(-dir[2])/np.pi*180:.1f}"+r"$^{\circ}$", 
                        fontsize=10, color='black', horizontalalignment='center', verticalalignment='center')
            if prop["dir"][-1] =="cw":
                col = "r"
                ls = "-"
            else:
                col = "b"
                ls = "--"
            
            theta = np.arctan2(dir[0],dir[1])
            alpha1 = np.linspace(-np.pi/2, np.pi/2)
            alpha2 = np.linspace(np.pi/2, 3*np.pi/2)

            a = size/2
            b = -dir[2]*size/2
            r1 = a*b/(np.sqrt((b*np.sin(alpha1))**2 + (a*np.cos(alpha1))**2))
            r2 = a*b/(np.sqrt((b*np.sin(alpha2))**2 + (a*np.cos(alpha2))**2))
            
            ax.plot(r1*np.cos(alpha1+theta)+loc[1], r1*np.sin(alpha1+theta)+loc[0], col, linestyle=ls)
            ax.plot(r2*np.cos(alpha2+theta)+loc[1], r2*np.sin(alpha2+theta)+loc[0], col, linestyle=ls)
        
            # ax.text(loc[1], loc[0], f"{size_label}", fontsize=9, color='black')

        # ax.scatter(self.drone.cg[1], self.drone.cg[0], s=200, marker="x", color="red")
        # ax.text(self.drone.cg[1], self.drone.cg[0], "C.G.", fontsize=12, color='black')

        ccw = Line2D([0], [0], color='r', label="CW")
        cw = Line2D([0], [0], color='b', linestyle="--", label="CCW")
        arrow = Line2D([0], [0], linestyle=":", color="green")
        # cg = Line2D([], [], color="r", marker='x', linestyle='None')
        
        if legend:
            if quiver:
                ax.legend([ccw, cw, arrow], ["CW", "CCW", "Direction"])
            else:
                ax.legend([ccw, cw], ["CW", "CCW"], bbox_to_anchor=(1, 0.5))
        
        if axis_labels:
            ax.set_xlabel("y (m)")
            ax.set_ylabel("x (m)")

        if limits is not None:
            ax.set_xlim(-limits[0], limits[0])
            ax.set_ylim(-limits[1], limits[1])

        ax.set_aspect("equal", "box")
        
        

class SinglePheno_2D(PhenotypeBase):
    def __init__(self, genotype):
        self.num_att = 5
        self.genotype = genotype
        
        self.generate_props()
        
    def generate_props(self):
        length_map     = [0.1, 0.3]
        armtheta_map   = [-np.pi, np.pi] # Azimuth
        motorphi_map   = [0, 15*np.pi/180] # Inclination
        motortheta_map = [-np.pi/2, np.pi/2] # Azimuth
        
        self.props = []
        num_props = len(self.genotype) // self.num_att
        
        for i in range(num_props):
            lengthG     = self.genotype[i*self.num_att ]
            armthetaG   = self.genotype[i*self.num_att + 1]
            motorphiG   = self.genotype[i*self.num_att + 2]
            motorthetaG = self.genotype[i*self.num_att + 3]
            rotG        = self.genotype[i*self.num_att + 4]

            lengthP     = linmap(length_map, lengthG)
            armthetaP   = linmap(armtheta_map, armthetaG)
            motorphiP   = linmap(motorphi_map, motorphiG)
            motorthetaP = linmap(motortheta_map, motorthetaG)
            rotP        = "cw" if np.sign(rotG) >= 0 else "ccw"

            loc = [lengthP*np.cos(armthetaP), lengthP*np.sin(armthetaP), 0]
            # dir = [np.sin(motorphiP)*np.cos(motorthetaP), np.sin(motorphiP)*np.sin(motorthetaP), -np.cos(motorphiP), rotP]
            dir = [np.sin(motorphiP)*np.cos(motorthetaP+armthetaP), np.sin(motorphiP)*np.sin(motorthetaP+armthetaP), -np.cos(motorphiP), rotP]

            prop = {"loc": loc, "dir": dir, "propsize": 5}
            self.props.append(prop)


        # Update number of props
        self.num_props = len(self.props)
        self.adjust_scale()

        self.drone = Custombody(self.props)
            

class Phenotype_2D(PhenotypeBase):
    def __init__(self, genotype, min_props=min_props, max_props=max_props):
        self.min_props = min_props
        self.max_props = max_props
        self.num_att = 5    # arm length, arm angle, phi, theta, rotation
        self.genotype = genotype

        self.generate_props()

    def generate_props(self):
        prop_map       = [4, 5, 6, 7, 8]        
        length_map     = [0.1, 0.3]
        armtheta_map   = [-np.pi, np.pi] # Azimuth
        motorphi_map   = [0, 15*np.pi/180] # Inclination
        motortheta_map = [-np.pi/2, np.pi/2] # Azimuth

        self.props  = []
        num_props = quantmap(prop_map, self.genotype[0])
        # Create mandatory props
        for i in range(num_props):
            lengthG     = self.genotype[i*self.num_att + 1]
            armthetaG   = self.genotype[i*self.num_att + 2]
            motorphiG   = self.genotype[i*self.num_att + 3]
            motorthetaG = self.genotype[i*self.num_att + 4]
            rotG        = self.genotype[i*self.num_att + 5]

            lengthP     = linmap(length_map, lengthG)
            armthetaP   = linmap(armtheta_map, armthetaG)
            motorphiP   = linmap(motorphi_map, motorphiG)
            motorthetaP = linmap(motortheta_map, motorthetaG)
            rotP        = "cw" if np.sign(rotG) >= 0 else "ccw"

            loc = [lengthP*np.cos(armthetaP), lengthP*np.sin(armthetaP), 0]
            # dir = [np.sin(motorphiP)*np.cos(motorthetaP), np.sin(motorphiP)*np.sin(motorthetaP), -np.cos(motorphiP), rotP]
            dir = [np.sin(motorphiP)*np.cos(motorthetaP+armthetaP), np.sin(motorphiP)*np.sin(motorthetaP+armthetaP), -np.cos(motorphiP), rotP]

            prop = {"loc": loc, "dir": dir, "propsize": 5}
            self.props.append(prop)


        # Update number of props
        self.num_props = len(self.props)
        self.adjust_scale()

        self.drone = Custombody(self.props)