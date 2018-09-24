import numpy as np
import matplotlib.pyplot as plt
from math import *
import matplotlib.animation as anim
from shapely.geometry import *
import copy
from multiprocessing import Pool

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

FinalReach = []
FinalDie = []
FinalRunOut = []
FinalEnd = []
FinalFitness = []

class Brain:
    def __init__(self,size):
        self.step = 0
        self.directions = [np.zeros((2),dtype=np.float) for i in range(size)]
        self.size=size
        for i in range(self.size):
            randomAngle = np.random.random()*2*pi
            #print randomAngle
            self.directions[i][0] = cos(randomAngle)
            self.directions[i][1] = sin(randomAngle)
        #print self.directions

    def mutate(self,mode='random',slewrate=0.5,signrate=0.5): #slewrate: deviation in % on the selected gene, signrate: sign flip rate of the derived direction
        mutationRate = 0.01 #1% chance of mutation
        for i in range(self.size):
            if mode=='random':
                if np.random.random()<mutationRate:
                    randomAngle = np.random.random()*2*pi
                    self.directions[i][0] = cos(randomAngle)
                    self.directions[i][1] = sin(randomAngle)
            if mode == 'slow':
                val = np.random.random()
                if val < 0.5 :
                    dev = np.random.random()*2*slewrate-slewrate #+/- slewrate
                    ax = 1.1
                    while ax*ax > 1.0:
                        ax = dev+self.directions[i][0]
                    ay = sqrt(1-ax*ax)
                    if np.random.random() <= signrate:
                        ay *= -1.0
                    self.directions[i][0]=ax
                    self.directions[i][1]=ay
                else:
                    dev = np.random.random()*2*slewrate-slewrate #+/- slewrate
                    ax = 1.1
                    while ax*ax > 1.0:
                        ax = dev+self.directions[i][1]
                    ay = sqrt(1-ax*ax)
                    if np.random.random() <= signrate:
                        ay *= -1.0
                    self.directions[i][1]=ax
                    self.directions[i][0]=ay

class Dot:
    def __init__(self,brain=True):
        if brain:
            self.brain = Brain(100)
        else:
            self.brain = None
        self.dead = False
        self.reachedGoal = False
        self.runOut = False
        self.pos = np.zeros((2),dtype=np.float)
        self.vel = np.zeros((2),dtype=np.float)
        self.acc = np.zeros((2),dtype=np.float)
        self.pos += (height/2.+np.random.random()*4,10+np.random.random()*4)
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
        dx = self.pos[0]-goal[0]
        dy = self.pos[1]-goal[1]
        dist = dx*dx+dy*dy
        if (dist) <= 0.0000001:
            dist = 0.0000001
        self.fitness = 1.0/dist
        #print self.fitness, self.pos[0], self.pos[1],dx,dy,dist

    def getBaby(self):
        baby = Dot()
        baby.brain = copy.deepcopy(self.brain)
        baby.brain.step = 0 #else baby will be too grown up
        return baby

class Population:
    step = 0
    def __init__(self,size):
        self.size = size
        self.dots = [Dot() for i in range(self.size)]
        self.generation = 0
        #self.px = [d.pos[0] for d in self.dots]
        #self.py = [d.pos[1] for d in self.dots]
    def update(self):
        #self.px = []
        #self.py = []
        self.step += 1
        for z in range(self.size):
            self.dots[z].move()
            #self.px.append(d.pos[0])
            #self.py.append(d.pos[1])
    def calcFitness(self):
        for x in range(self.size):
            self.dots[x].calcFitness()
        self.calcFitnessSum()

    def selectParent(self):
        #print "In select parent: ",
        val = np.random.random()*self.fitnessSum
        #print val
        runningsum = 0.
        #print self.size
        for j in range(self.size):
            runningsum += self.dots[j].fitness
            #print j, self.dots[j].fitness,runningsum
            if runningsum > val :
                return self.dots[j]
            #else:
                #print self.dots[j].fitness,self.dots[j].pos
                #print runningsum, val
                #self.none += 1
                #return None
            

    def naturalSelection(self):
        self.step = 0
        self.calcFitness()
        self.newDots = [Dot(brain=False) for x in range(self.size)]
        self.none = 0
        for il in range(self.size):
            #print "Index ",il," looking for parent:"
            #select newborns based on parent fitness
            parent = self.selectParent()
            #get babies from them (low chance of mutation)
            self.newDots[il]=parent.getBaby()
        #print "None: ",self.none
        self.dots = copy.deepcopy(self.newDots)
        self.mutateBabies()
        self.generation += 1

    def calcFitnessSum(self):
        #print "Calcfitness called"
        self.fitnessSum = 0.
        for i in range(self.size):
            self.fitnessSum += self.dots[i].fitness
        #print "Total fitness: " , self.fitnessSum
        return
    

    def mutateBabies(self):
        for x in range(self.size):
            self.dots[x].brain.mutate()

def dot_update(d):
    d.move()
    return d

a = Population(200)

fig=plt.figure(figsize=(20,10))

fig.suptitle(
    """
    Iteration: %d, Dots: %d, Step: %d, Reached: %d, Dead: %d, Runout: %d.
    """ % (a.generation,a.size,0,0,0,0)
)


ax = fig.add_subplot(1,1,1)
ax.set_xlim(0,height)
ax.set_ylim(0,width)
#ax.plot([a.pos[0]],[a.pos[1]],marker='o',ls='',markersize=2)
for i in range(a.size):
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
    global end,animstop
    #a.update()
    with Pool(processes=6) as pool:
        newDots = pool.map(dot_update,a.dots,1)
    a.dots = newDots.copy()
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
    for k in range(a.size):
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
        end = a.step
    else:
        #animstop = True
        end = 0
        FinalDie.append(dead)
        FinalReach.append(reached)
        FinalRunOut.append(runout)
        FinalEnd.append(a.step)
        a.naturalSelection()
        FinalFitness.append(a.fitnessSum)
    fig.suptitle(
    """
    Iteration: %d, Dots: %d, Step: %d, Reached: %d, Dead: %d, Runout: %d.
    """ % (a.generation,a.size,end,reached,dead,runout)
    )

px = anim.FuncAnimation(fig,update,repeat=False)
plt.show(block=True)
# while not animstop:
#     plt.pause(1)
# plt.close()
plt.figure()
plt.plot(FinalDie)
plt.plot(FinalReach)
plt.plot(FinalRunOut)
plt.plot(FinalEnd)
plt.plot(FinalFitness/(50*np.max(np.array(FinalFitness))))
plt.show()
