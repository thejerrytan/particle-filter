from Tkinter import *
from scipy import stats
import numpy as np
# import matplotlib.pyplot as plt

# Particle filter algorithm - finding position of robot in a 2 dimensional space using noisy sensors
# ===================================================== THEORY =========================================================================================
# First we need to model the problem as Hidden Markov Model (HMM)
# Notation: S - state space which can be discrete, continuous, defined on a range (smin, smax), t - time,
#           X - probability distribution of state, Y - probability distribution of observations, y(t) - observation at time t
#           B(X) - sampling distribution of X
# 1) State space, X which produces a sequence of hidden (unobserved) state x(t) i.e. true location of the robot
# 2) Transition model - P(x(t) | x(t-1)) i.e. what is the probability of robot being 1 step to the right at the next time step?
# 3) Sequence of observations - readings Y(t) from noisy sensor and P(Y | X) - what is my sensor error?
# What are we solving here, inference problem? P(X(t) | Y(t=0,1,2,3....t-1))
#
# SOLUTION
# Elapse time step - compute P(X(t) | y(1:t-1)) i.e. what is my probability distribution of X given history of observations?
#                    For every possible value of x in state space, P(x(t) | y(1:t-1)) = Summation over x(t-1) of P(x(t-1) | y(1:t-1)) * P(x(t) | x(t-1))
# Note the recurrence relation here, answer for current time step is dependent on answer for previous time step, so we can use dynamic programming here.
# 
# Observe step - Compute P(X(t) | y(1:t))
#                P(x(t) | y(1:t)) = P(x(t) | y(1:t-1)) * P(y(t) | x(t))
#
#
# ============================================ MOTIVATION FOR PARTICLE FILTER ==========================================================================
# Time complexity of elapse time step is |S|^2 because we have to perform the summation for every state to arrive at a distribution. 
# Thus the motivation for particle filters -> approximate solution to the above
# We use N particles (samples) to represent P(X)
# P(x) approximated by fraction of particles with value x, if N << |S|, we have many states with P(x) = 0 by pigeonhole principle
# Start with a prior distribution of where the robot is at time t = 0, if no clue at all, just use a uniform distribution
# 1) Elapse time step - each particle is moved by sampling its next position from the transition model
#                     x' = sample(P(X' | x))
#                     We approximate the new distribution using samples (particles) and thus the reduction in complexity
# 2) Observe step     - downweight samples based on the evidence.
#                     w(x) = P(y|x)
#                     B(X) = P(y|X) * B'(X)
#                     Normalize all the particles so sum of B(X) = 1
# If we iterate through these 2 steps, over time some of these particles are going to vanish to 0,
# which means we are getting coarser approximation of the true distribution. Thus step 3.
# 3) Resampling       - Rather than tracking weighted samples, we resample.
#                     N times, we choose from weighted sample distribution. Draw with replacement.
#                     Notice that we are sampling from the sampling distrubution, which is a reduced state space, thus reduction in complexity.
#                     weighted particles -> distribution -> unweighted particles
# Iterate till convergence.
# 
# So what is being filtered out and when? 
# 1) elapse time step, when we sample under the transition dynamics of the world, as N << |S| most states will end up with 0 or low probability
# 2) Resampling - we are drawing from a sample distribution which has reduced state space

# Constants

# State space is CANVAS_WIDTH * CANVAS_HEIGHT
CANVAS_WIDTH = 80
CANVAS_HEIGHT = 80
PARTICLE_RADIUS = 2
NUM_OF_PARTICLES = 200
ROBOT_RADIUS = 5
ROBOT_INIT_POS = (CANVAS_HEIGHT/2, CANVAS_WIDTH/2)
SAMPLE_SIZE = 50

# Global data structures
INIT_STATE_TABLE = None
TRANSITION_TABLE = None
OBS_ERROR_TABLE  = None
PARTICLE_LOCATION = {}

ERROR_TOLERANCE = 0.00001

# Initialize prior state distribution, distribution = ["uniform", "gaussian"]
def init_state(distribution, **vargs):
	global INIT_STATE_TABLE
	if distribution == "uniform":
		prob = 1.0 / (CANVAS_HEIGHT * CANVAS_WIDTH)
		INIT_STATE_TABLE = np.array([CANVAS_HEIGHT, CANVAS_WIDTH]).fill(prob)
		init_particles(distribution, **vargs)
	elif distribution == "gaussian":
		raise Exception("Not implemented")
	else:
		raise Exception("Invalid distribution - use one of uniform, gaussian")

# Construct the transition model probabilities, model = ["random", "gaussian", "gaussian-with-drift", "stationary"]
def init_transition_model(model, **vargs):
	global TRANSITION_TABLE
	if model == "random":
		TRANSITION_TABLE = np.random.rand(CANVAS_HEIGHT, CANVAS_WIDTH, CANVAS_HEIGHT, CANVAS_WIDTH)
		for x in range(TRANSITION_TABLE.shape[0]):
			for y in range(TRANSITION_TABLE.shape[1]):
				norm = np.sum(TRANSITION_TABLE[x,y])
				TRANSITION_TABLE[x,y,:,:] /= norm
	elif model == "gaussian":
		try:
			cov = vargs["covariance"]
		except KeyError:
			raise Exception("Please specify covariance matrix (standard deviation)")
		TRANSITION_TABLE = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, CANVAS_HEIGHT, CANVAS_WIDTH))
		if cov is None:
			# Default
			cov = np.array([[5,0],[0,5]])
		coords_x, coords_y = np.mgrid[0:CANVAS_HEIGHT, 0:CANVAS_WIDTH]
		coords = np.dstack((coords_x, coords_y))
		for x in range(TRANSITION_TABLE.shape[0]):
			for y in range(TRANSITION_TABLE.shape[1]):
				# Generate a multivariate truncated gaussian with mean (x,y) and bounded by (0,CANVAS_HEIGHT) 
				# in the x direction, bounded by (0, CANVAS_WIDTH) in the y direction, with covariance matrix cov
				# rescale by a, b
				# Note this is a hack, this is not a truncated multivariate norm distribution. 
				mean = np.array([x,y])
				rv = stats.multivariate_normal(mean, cov)
				TRANSITION_TABLE[x,y,:,:] = rv.pdf(coords)
				norm = np.sum(TRANSITION_TABLE[x,y])
				TRANSITION_TABLE[x,y] /= norm
				# plt.contourf(coords_x, coords_y, rv.pdf(coords))
				# plt.show()
				assert(abs(np.sum(TRANSITION_TABLE[x,y]) - 1.0) < ERROR_TOLERANCE)
	elif model == "gaussian-with-drift":
		raise Exception("Not implemented")
	elif model == "stationary":
		raise Exception("Not implemented")
	else:
		raise Exception("Invalid model - use one of random, gaussian, gaussian-with-drift, stationary")

# Construct the table of P(y|x), distribution = ["random", "gaussian"]
def init_obs_given_state(distribution, **vargs):
	global OBS_ERROR_TABLE
	if distribution == "random":
		raise Exception("not implemented")
	elif distribution == "gaussian":
		# Typical scenario, sensor gives a reading +- some degree of accuracy. So Y = X + error, error ~ N(0, sigma)
		try:
			cov = vargs["covariance"]
		except KeyError:
			raise Exception("Please specify covariance matrix (standard deviation)")
		OBS_ERROR_TABLE = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, CANVAS_HEIGHT, CANVAS_WIDTH))
		if cov is None:
			# Default
			cov = np.array([[2,0],[0,2]])
		coords_x, coords_y = np.mgrid[0:CANVAS_HEIGHT, 0:CANVAS_WIDTH]
		coords = np.dstack((coords_x, coords_y))
		for x in range(OBS_ERROR_TABLE.shape[0]):
			for y in range(OBS_ERROR_TABLE.shape[1]):
				# Generate a multivariate truncated gaussian with mean (x,y) and bounded by (0,CANVAS_HEIGHT) 
				# in the x direction, bounded by (0, CANVAS_WIDTH) in the y direction, with covariance matrix cov
				# rescale by a, b
				# Note this is a hack, this is not a truncated multivariate norm distribution. 
				mean = np.array([x,y])
				rv = stats.multivariate_normal(mean, cov)
				OBS_ERROR_TABLE[x,y,:,:] = rv.pdf(coords)
				norm = np.sum(OBS_ERROR_TABLE[x,y])
				OBS_ERROR_TABLE[x,y] /= norm
				# plt.contourf(coords_x, coords_y, rv.pdf(coords))
				# plt.show()
				assert(abs(np.sum(OBS_ERROR_TABLE[x,y]) - 1.0) < ERROR_TOLERANCE)
	else:
		raise Exception("Invalid model - use one of random, gaussian")
	pass

def init_particles(distribution, **vargs):
	if distribution == "uniform":
		x_samples = np.random.randint(low=0, high=CANVAS_HEIGHT, size=NUM_OF_PARTICLES)
		y_samples = np.random.randint(low=0, high=CANVAS_WIDTH, size=NUM_OF_PARTICLES)
		samples = np.dstack((x_samples, y_samples))
		for i in range(0, samples.shape[1]):
			PARTICLE_LOCATION[i] = samples[0,i]
	else:
		raise Exception("Invalid distribution - must be one of gaussian, uniform")

def elapse_time_step(distribution):
	for (idx, coords) in PARTICLE_LOCATION.iteritems():
		transition_model_given_x = TRANSITION_TABLE[coords[0], coords[1],:, :]
		if distribution == "gaussian":
			

	pass

def weight_particles():
	pass

def resample():
	pass

class Application(Canvas):
	def create_widgets(self):
		self.QUIT = Button(self)
		self.QUIT["text"] = "QUIT"
		self.QUIT["fg"]   = "red"
		self.QUIT["command"] =  self.quit

		self.QUIT.pack({"side": "left"})

		self.hi_there = Button(self)
		self.hi_there["text"] = "Hello",
		self.hi_there["command"] = self.say_hi

		self.hi_there.pack({"side": "left"})

	def add_robot(self):
		x1, y1 = (ROBOT_INIT_POS[0] - PARTICLE_RADIUS), (ROBOT_INIT_POS[1] - PARTICLE_RADIUS)
		x2, y2 = (ROBOT_INIT_POS[0] + PARTICLE_RADIUS), (ROBOT_INIT_POS[1] + PARTICLE_RADIUS)
		self.c.delete('robot')
		self.c.create_oval(x1, y1, x2, y2, fill="red", tag='robot')

	def add_particles(self):
		# x1, y1 = (event.x - PARTICLE_RADIUS), (event.y - PARTICLE_RADIUS)
		# x2, y2 = (event.x + PARTICLE_RADIUS), (event.y + PARTICLE_RADIUS)
		self.c.delete('particles')
		for (idx, coord) in PARTICLE_LOCATION.iteritems():
			x1, y1 = (coord[0] - PARTICLE_RADIUS), (coord[1] - PARTICLE_RADIUS)
			x2, y2 = (coord[0] + PARTICLE_RADIUS), (coord[1] + PARTICLE_RADIUS)
			self.c.create_oval(x1, y1, x2, y2, fill="green", tag='particles')

	def create_grid(self):
		width = self.c.winfo_width()
		height = self.c.winfo_height()
		self.c.delete('grid_line')

		for i in range(0, width, 10):
			self.c.create_line([(i,0), (i,height)], tag='grid_line')
			self.c.create_line([(0,i), (width,i)], tag='grid_line')

	def __init__(self, master=None):
		self.c = Canvas(master, height=CANVAS_HEIGHT, width=CANVAS_WIDTH, bg='white')
		self.c.pack()
		self.add_particles()
		self.add_robot()
		self.create_grid()

def main():
	init_state("uniform")
	init_transition_model("gaussian", covariance=None)
	init_obs_given_state("gaussian", covariance=None)
	root = Tk()
	app = Application(master=root)
	root.mainloop()
	# root.destroy()

if __name__ == "__main__":
	main()