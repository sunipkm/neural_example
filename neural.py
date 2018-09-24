import numpy as np
import matplotlib.pyplot as plt
from math import *
import matplotlib.animation as anim
from shapely.geometry import *

height = 100
width = 100
dt = 1
vlim = 10
goal = np.array([50,96])
goalboundx=[48,52,52,48,48]
goalboundy=[94,94,98,98,94]

wallco1 = ((0,48),(30,48),(30,52),(0,52),(0,48))
wallco2 = ((100,48),(70,48),(70,52),(100,52),(100,48))

goalcoords = ((48.,94.),(52.,94.),(52.,98.),(48.,98.),(48.,94.))
goalpoly = Polygon(goalcoords)

wallpoly = [Polygon(wallco1),Polygon(wallco2)]
#col = 'b' #default color

class Brain:
    def __init__(self,size):
        self.step = 0
        self.directions = [np.zeros((2),dtype=np.float) for i in xrange(size)]
        self.size=size
        for i in xrange(self.size):
            randomAngle = np.random.random()*2*pi
            #print randomAngle
            self.directions[i][0] = cos(randomAngle)
            self.directions[i][1] = sin(randomAngle)
        #print self.directions

class Dot:
    def __init__(self):
        self.brain = Brain(400)
        self.dead = False
        self.reachedGoal = False
        self.runOut = False
        self.pos = np.zeros((2),dtype=np.float)
        self.vel = np.zeros((2),dtype=np.float)
        self.acc = np.zeros((2),dtype=np.float)
        self.pos += (height/2.,10)
    def move(self):
        if self.dead or self.reachedGoal or self.runOut:
            #print "Dead"
            return
        if ( self.brain.step<len(self.brain.directions)):
            self.acc = self.brain.directions[self.brain.step]
            self.brain.step += 1
        else: 
            self.runOut = True
        self.vel+=self.acc*dt
        norm = np.sqrt(np.sum(self.vel*self.vel))
        if (norm>vlim):
            self.vel *= vlim/norm
        pos=self.pos+self.vel*dt*0.5
        # if goal[0]-2<pos[0]<goal[0]+2 and goal[1]-2<pos[1]<goal[1]+2:
        #     self.reachedGoal = True
        if goalpoly.contains(Point(pos[0],pos[1])):
            self.reachedGoal = True
        for wall in wallpoly:
            if wall.contains(Point(pos[0],pos[1])):
                self.dead = True
        if not (0<pos[0]<=height and 0<pos[1]<=width):
            self.dead = True
        if not self.dead :
            self.pos = pos
    
    def calcFitness(self):
        self.fitness = 1.0/np.sum((self.pos-goal)**2)

class Population:
    def __init__(self,size):
        self.size = size
        self.dots = [Dot() for i in xrange(self.size)]
        #self.px = [d.pos[0] for d in self.dots]
        #self.py = [d.pos[1] for d in self.dots]
    def update(self):
        #self.px = []
        #self.py = []
        for i in xrange(self.size):
            self.dots[i].move()
            #self.px.append(d.pos[0])
            #self.py.append(d.pos[1])
    def calcFitness(self):
        for i in xrange(self.size):
            self.dots[i].calcFitness()
a = Population(200)

fig,ax=plt.subplots(figsize=(10,10))

fig.suptitle(
    """
    Iteration: %d, Dots: %d, Step: %d, Reached: %d, Dead: %d, Runout: %d.
    """ % (1,a.size,0,0,0,0)
)


#ax = fig.add_subplot(1,1,1)
ax.set_xlim(0,height)
ax.set_ylim(0,width)
#ax.plot([a.pos[0]],[a.pos[1]],marker='o',ls='',markersize=2)
for i in xrange(a.size):
    ax.plot([a.dots[i].pos[0]],[a.dots[i].pos[1]],marker='o',ls='',markersize=2,color='b')
#ax.scatter([goal[0]],[goal [1]],marker='s',color='g',s=36*fig.dpi/72.)

ax.plot(goalboundx,goalboundy,color='g')
for wall in wallpoly:
    wx,wy=wall.exterior.xy
    ax.plot(wx,wy,color='r')


major_ticks = np.arange(0,101,10)
minor_ticks = np.arange(0,101,1)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks,minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks,minor=True)
end = 0
animstop = False
def update(i):
    global end
    a.update()
    ax.clear()
    ax.set_xlim(0,height)
    ax.set_ylim(0,width)
    major_ticks = np.arange(0,101,10)
    minor_ticks = np.arange(0,101,1)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks,minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks,minor=True)
    ax.grid(which='minor')
    #print a.pos
    #ax.scatter([goal[0]],[goal[1]],marker='s',color='g',s=81*(fig.dpi/72.))
    #ax.plot([a.pos[0]],[a.pos[1]],marker='o',ls='',markersize=2)
    ax.plot(goalboundx,goalboundy,color='g')
    for wall in wallpoly:
        wx,wy=wall.exterior.xy
        ax.plot(wx,wy,color='r')
    dead = 0
    reached = 0
    runout = 0
    for k in xrange(a.size):
        if a.dots[k].reachedGoal :
            col = 'r'
            reached += 1
        elif a.dots[k].runOut:
            col = 'cyan'
            runout += 1
        elif a.dots[k].dead:
            col = 'black'
            dead += 1
        else:
            col = 'b'
        ax.plot([a.dots[k].pos[0]],[a.dots[k].pos[1]],marker='o',ls='',markersize=2,color=col)
    if not ( dead + reached + runout == a.size ):
        end = i
    fig.suptitle(
    """
    Iteration: %d, Dots: %d, Step: %d, Reached: %d, Dead: %d, Runout: %d.
    """ % (1,a.size,end,reached,dead,runout)
    )

px = anim.FuncAnimation(fig,update,repeat=False)
plt.show()