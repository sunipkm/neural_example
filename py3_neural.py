#--------- Imports ----------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos,pi,sqrt
import matplotlib.animation as anim
from shapely.geometry import Polygon,LineString,Point
import copy
from multiprocessing import Pool
#----------------------------------------------------------------------------------------------------------

#--------- Global Variables -------------------------------------------------------------------------------
height = 100
width = 100
dt = 1
vlim = 10
#----------------------------------------------------------------------------------------------------------

#--------- Goal Settings --------------------------------------------------------------------------------
goal = np.array([50,96])
goalboundx=[48,52,52,48,48]
goalboundy=[94,94,98,98,94]
goalcoords = ((48.,94.),(52.,94.),(52.,98.),(48.,98.),(48.,94.))
goalpoly = Polygon(goalcoords)
#----------------------------------------------------------------------------------------------------------

#--------- Blocking walls ---------------------------------------------------------------------------------
wallco1 = ((0,48),(30,48),(30,52),(0,52),(0,48))
wallco2 = ((100,48),(70,48),(70,52),(100,52),(100,48))
wallpoly = [Polygon(wallco1),Polygon(wallco2)]
#----------------------------------------------------------------------------------------------------------

#--------- Boundary Lines ------------------------------------------------------------------------------
bound1 = LineString([(0,0),(100,0)])
bound2 = LineString([(0,0),(0,100)])
bound3 = LineString([(100,0),(100,100)])
bound4 = LineString([(0,100),(100,100)])
bounds = [bound1,bound2,bound3,bound4]
#----------------------------------------------------------------------------------------------------------
#col = 'b' #default color

#--------- Statistics -------------------------------------------------------------------------------------
FinalReach = []
FinalDie = []
FinalRunOut = []
FinalEnd = []
FinalFitness = []
#----------------------------------------------------------------------------------------------------------

#--------- Gene pool of particles -------------------------------------------------------------------------
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
#----------------------------------------------------------------------------------------------------------

#--------- Particles --------------------------------------------------------------------------------------
class Dot:
    def __init__(self,brain=True,brainsize=100):
        if brain:
            self.brain = Brain(brainsize)
        else:
            self.brain = None
        self.dead = False
        self.reachedGoal = False
        self.runOut = False
        self.pos = np.zeros((2),dtype=np.float)
        self.vel = np.zeros((2),dtype=np.float)
        self.acc = np.zeros((2),dtype=np.float)
        self.pos += (height/2.#+np.random.random()*4
        ,10#+np.random.random()*4
        )
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
        path = LineString([(self.pos[0],self.pos[1]),(pos[0],pos[1])])
        # if goalpoly.contains(Point(pos[0],pos[1])):
        #     self.reachedGoal = True
        #     self.pos = pos
        if path.intersects(goalpoly):
            self.reachedGoal = True
            #intersect = path.intersection(goalpoly)
            self.pos = goal
        for wall in wallpoly:
            if path.intersects(wall):
                self.dead = True
                intersect = path.intersection(wall)
                pos = np.array(intersect.coords[0])
                self.pos = pos
        for bound in bounds:
            if path.intersects(bound):
                self.dead = True
                intersect = path.intersection(bound)
                self.pos=np.array([intersect.x,intersect.y])
        if not (0<pos[0]<=height and 0<pos[1]<=width):
            self.dead = True
        if not self.dead :
            if not self.reachedGoal:
                self.pos = pos
    
    def calcFitness(self):
        if not self.reachedGoal:
            dx = self.pos[0]-goal[0]
            dy = self.pos[1]-goal[1]
            dist = dx*dx+dy*dy
            if dist > 16.:
                self.fitness = 1.0/16.0
            else:
                self.fitness = 1./dist
        #print self.fitness, self.pos[0], self.pos[1],dx,dy,dist
        else:
            self.fitness = 1./16. + 2500./(self.brain.step*self.brain.step)

    def getBaby(self):
        baby = Dot()
        baby.brain = copy.deepcopy(self.brain)
        baby.brain.step = 0 #else baby will be too grown up
        return baby
#----------------------------------------------------------------------------------------------------------

#--------- Population Class -------------------------------------------------------------------------------
class Population:
    step = 0
    maxstep = 100
    def __init__(self,size,maxstep=100):
        self.size = size
        self.maxstep = maxstep
        self.bestIndex = size
        self.dots = [Dot(brainsize=maxstep) for i in range(self.size)]
        self.generation = 0
        #self.px = [d.pos[0] for d in self.dots]
        #self.py = [d.pos[1] for d in self.dots]
    def update(self):
        #self.px = []
        #self.py = []
        self.step += 1
        for z in range(self.size):
            if(self.step<self.maxstep):
                self.dots[z].move()
            else:
                self.dots[z].runOut = True
            #self.px.append(d.pos[0])
            #self.py.append(d.pos[1])
    def calcFitness(self):
        # #maxstep = self.maxstep
        # fitness = 0
        # bestIndex = -1
        for x in range(self.size):
            # if self.dots[x].reachedGoal:
            #     if self.dots[x].brain.step < maxstep:
            #         maxstep = self.dots[x].brain.step

            self.dots[x].calcFitness()
            # if self.dots[x].fitness>fitness:
            #     fitness = self.dots[x].fitness
            #     bestIndex = x
        # self.maxstep = maxstep
        # if bestIndex >= 0 :
        #     self.bestIndex = bestIndex
        # else:
        #     self.bestIndex = self.size
        
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
            # if x != self.bestIndex:
            self.dots[x].brain.mutate()
#----------------------------------------------------------------------------------------------------------

#--------- Parallelization Help ---------------------------------------------------------------------
def dot_update(d):
    d.move()
    return d
#----------------------------------------------------------------------------------------------------------

#--------- Begin MAIN -------------------------------------------------------------------------------------
a = Population(200,maxstep=400)

#--------- Animation Figure -------------------------------------------------------------------------------
fig=plt.figure(figsize=(10,10))

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
#----------------------------------------------------------------------------------------------------------

#--------- Frame Iterator ---------------------------------------------------------------------------------
def update(i):
    global end,animstop
    a.update()
    # with Pool(processes=6) as pool:
    #     newDots = pool.map(dot_update,a.dots,1)
    # a.dots = newDots.copy()
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
#----------------------------------------------------------------------------------------------------------

#--------- Statistics Plotter -----------------------------------------------------------------------------
FinalFitness = np.array(FinalFitness)
plt.figure(figsize=(10,10))
plt.plot(FinalDie,label='Deaths')
plt.plot(FinalReach,label='Wins')
plt.plot(FinalRunOut,label='Out of breath')
plt.plot(FinalEnd,label='Simulation length')
plt.plot(50*FinalFitness/(np.max(np.array(FinalFitness))),label='Normalized fitness sum')
plt.grid()
plt.legend()
plt.show()
#----------------------------------------------------------------------------------------------------------