import numpy as np
import vtk
import matplotlib.pyplot as plt
import vtk_visualizer as vis
from equation_solver import *
import math
from hocook import hocook,planecontact


def rotx(q):
	c = np.cos(q)
	s = np.sin(q)
	return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
	
def roty(q):
	c = np.cos(q)
	s = np.sin(q)
	return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rotz(q):
	c = np.cos(q)
	s = np.sin(q)
	return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

class tool():
	def __init__(self, scene):
		s = scene	
		self.finger1 = vis.cube(0.02, 0.01, 0.05)
		s.add_actor(self.finger1)
		self.finger2 = vis.cube(0.02, 0.01, 0.05)
		s.add_actor(self.finger2)
		self.palm = vis.cube(0.03, 0.08, 0.03)
		s.add_actor(self.palm)
		self.wrist = vis.cylinder(0.015, 0.04)
		s.add_actor(self.wrist)
		
	def set_configuration(self, g, TGS):	
		TF1G = np.identity(4)
		TF1G[:3,3] = np.array([0, -0.5*g-0.005, -0.025])
		TF1S = TGS @ TF1G
		vis.set_pose(self.finger1, TF1S)
		TF2G = np.identity(4)
		TF2G[:3,3] = np.array([0, 0.5*g+0.005, -0.025])	
		TF2S = TGS @ TF2G
		vis.set_pose(self.finger2, TF2S)
		TPG = np.identity(4)
		TPG[:3,3] = np.array([0, 0, -0.065])
		TPS = TGS @ TPG
		vis.set_pose(self.palm, TPS)
		TWG = np.block([[rotx(np.pi/2), np.array([[0], [0], [-0.1]])], [np.zeros((1, 3)), 1]])
		TWS = TGS @ TWG
		vis.set_pose(self.wrist, TWS)


def set_floor(s, size):
	floor = vis.cube(size[0], size[1], 0.01, (1,1,1))
	TFS = np.identity(4)
	TFS[2,3] = -0.005
	vis.set_pose(floor, TFS)
	s.add_actor(floor)

def visualize_gripper():
	# Scene
	s = vis.visualizer()

	# Floor.
	set_floor(s, [1, 1])

	# Cube.
	cube = vis.cube(0.03, 0.03, 0.03)
	TCS = np.identity(4)
	TCS[:3,3] = np.array([0, 0, 1.5])
	vis.set_pose(cube, TCS)
	s.add_actor(cube)
	
	# Tool.
	TGS = np.identity(4)
	TGS[:3,3] = np.array([0, 0, 1.0])
	tool_ = tool(s)
	tool_.set_configuration(0.015, TGS)
		
	# Render scene.
	s.run()


# TASK 1
class robot():
	def __init__(self, scene):
		q = np.zeros(6)
		d = np.array([0, 0, 0, 0.795, 0, 0.105])
		a = np.array([0.150, 0.760, 0.140, 0, 0, 0])
		al = np.array([-np.pi/2, 0, -np.pi/2,np.pi/2, -np.pi/2, 0])
		self.DH = np.stack((q, d, a, al), 1)
	
		s = scene
		
		#base
		self.base = vis.cylinder(0.26,0.35)
		s.add_actor(self.base)

		#link1
		self.link1 = vis.cylinder(0.15, 0.26)	
		s.add_actor(self.link1)
		
		#link2
		self.link2 = vis.cube(0.76,0.15,0.15)
		s.add_actor(self.link2)

		#link3
		self.link3 = vis.cylinder(0.1,0.26)
		s.add_actor(self.link3)
		
		#link4
		self.link4 = vis.cube(0.16,0.4,0.16)
		s.add_actor(self.link4)
		
		#Link4_2
		self.link4_2=vis.cube(0.1,0.795,0.1)
		s.add_actor(self.link4_2)

		#Link 5.
		self.link5 = vis.cube(0.05,0.05,0.105)
		s.add_actor(self.link5)		

		#Tool.
		self.tool = tool(s)
		
	def set_configuration(self, q, g, T0S):
		d = self.DH[:,1]
		a = self.DH[:,2]
		al = self.DH[:,3]
		
		# Base.
		TB0 = np.identity(4)
		TB0[:3,:3]=rotx(np.pi/2)
		TB0[2,3] = -0.29
		TBS = T0S @ TB0
		vis.set_pose(self.base, TBS)

		# Link 1.
		T10 = dh(q[0], d[0], a[0], al[0])
		T1S = T0S @ T10
		TL11 = np.identity(4)
		TL11[:3,:3]=rotx(np.pi/2)	
		TL1S = T1S @ TL11
		vis.set_pose(self.link1, TL1S)
		
		# Link 2.
		T21 = dh(q[1], d[1], a[1], al[1])
		T2S = T1S @ T21	
		TL22 = np.identity(4)
		TL22[0,3]=-0.38
		TL22[2,3]=-0.1
		TL2S = T2S @ TL22
		vis.set_pose(self.link2, TL2S)

		# Link 3.
		T32 = dh(q[2], d[2], a[2], al[2])
		T3S = T2S @ T32	
		TL33 = np.identity(4)
		TL33[0,3]=-0.14
		TL3S = T3S @ TL33
		vis.set_pose(self.link3, TL3S)
		
		# Link 4.
		T43 = dh(q[3], d[3], a[3], al[3])
		T4S = T3S @ T43	
		TL44 = np.identity(4)
		TL44[1,3] = -0.795
		TL4S = T4S @ TL44
		vis.set_pose(self.link4, TL4S)

		TL44_2 = np.identity(4)
		TL44_2[1,3]=-0.3975
		TL4S_2 = T4S @ TL44_2
		vis.set_pose(self.link4_2, TL4S_2)
		
		# Link 5.
		T54 = dh(q[4], d[4], a[4], al[4])
		T5S = T4S @ T54	
		TL55 = np.identity(4)
		TL5S = T5S @ TL55
		vis.set_pose(self.link5, TL5S)
		
		# Link 6.
		T65 = dh(q[5], d[5], a[5], al[5])
		T6S = T5S @ T65	
		self.tool.set_configuration(g, T6S)

		# tool - base position
		T60 = T10@T21@T32@T43@T54@T65 
		T6S = T0S@T10@T21@T32@T43@T54@T65  #already computed

		return T6S	

def dh(q, d, a, al):
	cq = np.cos(q)
	sq = np.sin(q)
	ca = np.cos(al)
	sa = np.sin(al)
	T = np.array([[cq, -sq*ca, sq*sa, a*cq],
		[sq, cq*ca, -cq*sa, a*sq],
		[0, sa, ca, d],
		[0, 0, 0, 1]])
	return T

def visualize_robotic_arm(q):
	# Scene.
	s = vis.visualizer()

	# Axes.
	axes = vtk.vtkAxesActor()
	s.add_actor(axes)

	# Floor.
	set_floor(s, [2, 2])

	# cube
	cube = vis.cube(0.01, 0.02, 0.03,(0,1,0))
	
	# Robot.
	T0S = np.identity(4)
	T0S[2,3] = 0.5
	T0S[:3,:3]=rotz(np.pi)
	rob = robot(s)
	T6S=rob.set_configuration(q, 0.03, T0S)

	#setting position of a cube
	TA6 = np.identity(4)
	TA6 = np.matrix([	[1,0,0,0],
						[0,-1,0,0],
						[0,0,-1,-0.02],
						[0,0,0,1]     ])
	
	TAS = T6S@TA6
	vis.set_pose(cube, TAS)
	s.add_actor(cube)
	
	# Render scene.
	s.run()

def inv_kin(DH,T60,solution):
	d=DH[:,1]
	a=DH[:,2]
	al=DH[:,3]

	th=np.zeros(6)

	p = T60 @ np.expand_dims(np.array([0, 0, -d[5], 1]), 1)
	x=p[0,0]
	y=p[1,0]
	z=p[2,0]
	r = x**2 + y**2 + z**2

	#find theta3
	#constants
	m=a[2]**2 + d[3]**2 + a[1]**2
	n=2*a[1]*a[2]
	o=2*a[1]*d[3]

	roots = solve_quartic_eq([m,n,o,z,r,a[0]])

	#check solutions
	for root in roots:
		check=quartic_eq(2*math.atan(roots[0]), m, n, o, z, r, a[0])
		if abs(check)>0.1:
			print("Not a good angle!")
	
	if 	 solution[0] == 0:
		th[2] = 2*math.atan(roots[0])
	elif solution[0] == 1:
		th[2] = 2*math.atan(roots[1])
	elif solution[0] == 2:
		th[2] = 2*math.atan(roots[2])
	else:
		th[2] = 2*math.atan(roots[3])

	#find theta2
	f1 = -d[3]*math.sin(th[2]) + a[2]*math.cos(th[2]) + a[1]
	f2 = a[2]*math.sin(th[2]) + d[3]*math.cos(th[2])

	k1=m+a[0]**2 + n*math.cos(th[2]) - o*math.sin(th[2])

	u1,u2 = solve_quadratic_eq([r,a[0],f1,f2,k1])

	th_21 =2*math.atan(u1)
	th_22 =2*math.atan(u2)
	check_1=math.sin(al[0])*(f1*math.sin(th_21)+f2*math.cos(th_21))
	check_2=math.sin(al[0])*(f1*math.sin(th_22)+f2*math.cos(th_22))

	if z>0 and check_1>0 or z<0 and check_1<0:
		th[1] = 2*math.atan(u1)
	elif z<0 and check_2<0 or z>0 and check_2>0:
		th[1] = 2*math.atan(u2)

	#find theta1
	g1 = math.cos(th[1])*f1 - math.sin(th[1])*f2 + a[0]

	th[0] = math.atan2(y/g1,x/g1)

	#find theta4,theta5,theta6
	T10 = dh(th[0], d[0], a[0], al[0])
	T21 = dh(th[1], d[1], a[1], al[1])
	T32 = dh(th[2], d[2], a[2], al[2])
	T30 = T10 @ T21 @ T32
	R30 = T30[:3,:3]
	R60 = T60[:3,:3]
	R63 = R30.T @ R60
	
	c5 = R63[2,2]

	th[4] = math.acos(c5) 
	if solution[1] == 0:
		th[4]= - th[4]

	s5 = math.sin(th[4])
	if np.abs(s5) > 1e-10:
		th[3] = math.atan2(-R63[1,2]/s5, -R63[0,2]/s5)
		th[5] = math.atan2(-R63[2,1]/s5, R63[2,0]/s5)
	else:
		c46 = R63[0,0]
		s46 = R63[0,1]
		q46 = math.atan2(s46, c46)
		th[3] = q46
		th[5] = 0
		
	return th


def calc_inverse_kinematics(solution):
	TTS = np.identity(4)
	TTS[0,3] = 0.6
	TTS[1,3] = 1.0
	TTS[2,3] = 0.5
	
	# Scene
	s = vis.visualizer()

	# Floor.
	set_floor(s, [2, 2])
	
	# Target object.
	target = vis.cube(0.03, 0.03, 0.03, (0,1,0)) 
	

	# Robot.
	rob = robot(s)

	T0S = np.identity(4)
	T0S[2,3] = 0.5
	T0S[:3,:3]=rotz(np.pi)

	vis.set_pose(target, TTS)
	s.add_actor(target)

	T6T	= np.matrix([[0,1,0,0],[1,0,0,0],[0,0,-1,0],[0,0,0,1]])

	T60 = np.linalg.inv(T0S) @ TTS @ T6T
	q = inv_kin(rob.DH, T60,solution)
	rob.set_configuration(q, 0.03, T0S)
	
	# Render scene.
	s.run()

class simulator():
	def __init__(self, robot, Qc):
		self.timer_count = 0
		self.robot = robot
		self.Qc = Qc
		self.trajW = []
		self.T0S = np.identity(4)
		self.T0S[2,3] = 0.5

	def execute(self,iren,event):
		T6S = self.robot.set_configuration(self.Qc[:,self.timer_count % self.Qc.shape[1]], 0.03, self.T0S)
		self.trajW.append(T6S)
		iren.GetRenderWindow().Render()
		self.timer_count += 1

def plan_trajectory():
	# Scene
	s = vis.visualizer()

	#robot was moved 0.5 up
	offset = 0.5

	# Floor.
	set_floor(s, [2, 2])
	
	# Robot.
	rob = robot(s)
	
	# Robot velocity and acceleration limits.
	dqgr=1/2.5*np.pi*np.ones((1,6))
	ddqgr=1/2.5*10*np.pi*np.ones((1,6))
	
	# Trajectory.
	q_home = np.array([-np.pi/2, -np.pi/2, 0, 0, 0, 0])
	T60_1 = np.identity(4)
	T60_1[:3,:3] = roty(np.pi)
	T60_1[:3,3] = np.array([0.4, -0.15, 0.2 - offset])
	q1 = inv_kin(rob.DH,T60_1,[1, 0])

	T60_3 = T60_1.copy()
	T60_3[:3,3] = np.array([0.4, -0.15, 0.02-offset])
	q3 = inv_kin(rob.DH,T60_3,[1, 0])

	T60_4 = T60_1.copy()
	T60_4[:3,3] = np.array([0.4, -0.1, 0.02-offset])
	q4 = inv_kin(rob.DH,T60_4,[1, 0])

	T60_5 = T60_1.copy()
	T60_5[:3,3] = np.array([0.4, -0.05, 0.02-offset])
	q5 = inv_kin(rob.DH,T60_5,[1, 0])

	T60_6 = T60_1.copy()
	T60_6[:3,3] = np.array([0.4, 0.0, 0.02-offset])
	q6 = inv_kin(rob.DH,T60_6,[1, 0])

	T60_7 = T60_1.copy()
	T60_7[:3,3] = np.array([0.45, -0.05, 0.02-offset])
	q7 = inv_kin(rob.DH,T60_7,[1, 0])

	T60_8 = T60_1.copy()
	T60_8[:3,3] = np.array([0.475, -0.075, 0.02-offset])
	q8 = inv_kin(rob.DH,T60_8,[1, 0])

	T60_9 = T60_1.copy()
	T60_9[:3,3] = np.array([0.5, -0.05, 0.02-offset])
	q9 = inv_kin(rob.DH,T60_9,[1, 0])

	T60_10 = T60_1.copy()
	T60_10[:3,3] = np.array([0.55, 0, 0.02-offset])
	q10 = inv_kin(rob.DH,T60_10,[1, 0, 0])

	T60_11 = T60_1.copy()
	T60_11[:3,3] = np.array([0.55, -0.05, 0.02-offset])
	q11 = inv_kin(rob.DH,T60_11,[1, 0, 0])

	T60_12 = T60_1.copy()
	T60_12[:3,3] = np.array([0.55, -0.1, 0.02-offset])
	q12 = inv_kin(rob.DH,T60_12,[1, 0, 0])

	T60_13 = T60_1.copy()
	T60_13[:3,3] = np.array([0.55, -0.15, 0.02-offset])
	q13 = inv_kin(rob.DH,T60_13,[1, 0, 0])

	T60_15 = T60_1.copy()
	T60_15[:3,3] = np.array([0.55, -0.15, 0.2-offset])
	q15 = inv_kin(rob.DH,T60_15,[1, 0, 0])

	Q = np.stack((q_home, q1,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q15, q_home), 1)
	Ts = 0.01
	Qc, dQc, ddQc, tc = hocook(Q, dqgr, ddqgr, Ts)
	
	# Display trajectory.
	plt.plot(tc,Qc[0,:],tc,Qc[1,:],tc,Qc[2,:],tc,Qc[3,:],tc,Qc[4,:],tc,Qc[5,:])
	plt.title("Odziv varijabli zglobova")
	plt.show()
	plt.plot(tc,dQc[0,:],tc,dQc[1,:],tc,dQc[2,:],tc,dQc[3,:],tc,dQc[4,:],tc,dQc[5,:])
	plt.title("Brzina")
	plt.show()
	plt.plot(tc,ddQc[0,:],tc,ddQc[1,:],tc,ddQc[2,:],tc,ddQc[3,:],tc,ddQc[4,:],tc,ddQc[5,:])
	plt.title("Ubrzanje")
	plt.show()

	
	# Create animation callback.
	sim = simulator(rob, Qc)
	
	# Start animation.
	s.run(animation_timer_callback=sim.execute)
	
	# Display tool trajectory in 3D.
	trajW = np.array(sim.trajW)
	tool_tip_W = trajW[:,:3,3]
	ax = plt.axes(projection='3d')
	ax.plot3D(tool_tip_W[:,0], tool_tip_W[:,1], tool_tip_W[:,2], 'b')
	plt.show()
	
	# Display plane contact.
	n_board = np.array([0, 0, 1])
	d_board = 0.02
	board_draw = planecontact(tool_tip_W, n_board, d_board)
	fig, ax = plt.subplots(1, 1)
	ax.plot(board_draw[:,0], board_draw[:,1], 'b.')
	ax.axis('equal')
	plt.show()
		
	return tool_tip_W

def main():

	#first parameter:  0-3 (theta_3)
	#second parameter: 0-1 (theta_5)
	#task3([3,1])

	plan_trajectory()


if __name__ == '__main__':
    main()