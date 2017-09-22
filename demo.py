from Tkinter import *

CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 1200
PARTICLE_RADIUS = 2

class Application(Canvas):
    def say_hi(self):
        print "hi there, everyone!"

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