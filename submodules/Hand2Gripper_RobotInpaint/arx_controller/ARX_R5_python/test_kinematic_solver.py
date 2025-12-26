import bimanual
import numpy as np
np.set_printoptions(suppress=True)


if __name__ == "__main__":
	# joint 123456
	print(bimanual.forward_kinematics(np.array([0.00001145, 0.26783228, 0.39305905, -0.12518544, 0.00000951, -0.00001443])))  #joint2pos
	# pos xyzrpy
	print(bimanual.inverse_kinematics(np.array([0.09, 0.0, 0.2, 0.0, 0.0 ,0.0 ])))  #pos2joint
