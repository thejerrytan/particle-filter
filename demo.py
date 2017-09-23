from Tkinter import *

# Particle filter algorithm - finding position of robot in a 2 dimensional space using noisy sensors
# First we need to model the problem as Hidden Markov Model (HMM)
# Notation: S - state space which can be discrete, continuous, defined on a range (smin, smax), t - time,
#           X - probability distribution of state, Y - probability distribution of observations, y(t) - observation at time t 
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
# Time complexity of elapse time step is |S|^2 because we have to perform the summation for every state to arrive at a distribution. 
# Observe step - Compute P(X(t) | y(1:t))
#                P(x(t) | y(1:t)) = P(x(t) | y(1:t-1)) * P(y(t) | x(t))
# State space is CANVAS_WIDTH * CANVAS_HEIGHT

CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 600
PARTICLE_RADIUS = 2
NUM_OF_PARTICLES = 40000

def transition_model = 
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

    def add_particle(self, event):
        x1, y1 = (event.x - PARTICLE_RADIUS), (event.y - PARTICLE_RADIUS)
        x2, y2 = (event.x + PARTICLE_RADIUS), (event.y + PARTICLE_RADIUS)
        self.c.create_oval(x1, y1, x2, y2, fill="green")

    def create_grid(self, event=None):
        width = self.c.winfo_width()
        height = self.c.winfo_height()
        self.c.delete('grid_line')

        for i in range(0, width, 100):
            self.c.create_line([(i,0), (i,height)], tag='grid_line')
            self.c.create_line([(0,i), (width,i)], tag='grid_line')

    def __init__(self, master=None):
        self.c = Canvas(master, height=600, width=1200, bg='white')
        self.c.pack()
        self.c.bind('<Configure>', self.create_grid)
        self.c.bind('<B1-Motion>', self.add_particle)
        self.create_grid()

root = Tk()
app = Application(master=root)
root.mainloop()
root.destroy()