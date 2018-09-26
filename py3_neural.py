#--------- Imports ----------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos,pi,sqrt,exp
import matplotlib.animation as anim
from shapely.geometry import Polygon,LineString,Point
import copy
from multiprocessing import Pool
import matplotlib.gridspec as grspec
#----------------------------------------------------------------------------------------------------------

#--------- Global Variables -------------------------------------------------------------------------------
height = 100
width = 100
dt = 1
vlim = 10
t = 0 ; v = 1
wall_right = True
#----------------------------------------------------------------------------------------------------------

#--------- Goal Settings --------------------------------------------------------------------------------
goal = np.array([50,96])
goalboundx=[48,52,52,48,48]
goalboundy=[94,94,98,98,94]
goalcoords = ((48.,94.),(52.,94.),(52.,98.),(48.,98.),(48.,94.))
goalpoly = Polygon(goalcoords)
#----------------------------------------------------------------------------------------------------------

#--------- Blocking walls ---------------------------------------------------------------------------------
# wallco1 = ((0,48-10),(30,48-10),(30,52-10),(0,52-10),(0,4810))
# wallco2 = ((100,48+20),(70,48+20),(70,52+20),(100,52+20),(100,48+20))
# wallpoly = [Polygon(wallco1),Polygon(wallco2)]
# wallcol1 = []
# for wx in wallco1:
#     wallcol1.append(list(wx))
# wallcol2 = []
# for wx in wallco2:
#     wallcol2.append(list(wx))
# wallpoly = [(wallcol1),(wallcol2)]

wallco1 = [[15,41],15,2,1]
wallco2 = [[85,71],15,2,-1]
wallpoly = [wallco1,wallco2]
#----------------------------------------------------------------------------------------------------------

#--------- Moving Walls -----------------------------------------------------------------------------------
def move_wall(w_coord,w_num=10):
    """
    move_wall:
        1. w_coord: List of the format [[middle of box x, middle of box y],width/2,height/2,direction]
        2. w_num: Sets direction (and velocity) of individual walls  
    """
    global width
    rx = w_coord[0][0]+w_coord[1]+w_coord[3]*w_num
    lx = w_coord[0][0]-w_coord[1]+w_coord[3]*w_num
    if rx > width or lx < 0:
        w_coord[3]*=-1
        rx = w_coord[0][0]+w_coord[1]+w_coord[3]*w_num
        lx = w_coord[0][1]-w_coord[1]+w_coord[3]*w_num
    w_coord[0][0] += w_coord[3]*w_num
    x=w_coord[0][0];y=w_coord[0][1]

    w=[[x-w_coord[1],y-w_coord[2]],[x+w_coord[1],y-w_coord[2]],[x+w_coord[1],y+w_coord[2]],[x-w_coord[1],y+w_coord[2],[x-w_coord[1],y-w_coord[2]]]]
    wall = Polygon(w)
    return w_coord,wall
#----------------------------------------------------------------------------------------------------------

#--------- Boundary Lines ---------------------------------------------------------------------------------
bound1 = LineString([(0,0),(100,0)])
bound2 = LineString([(0,0),(0,100)])
bound3 = LineString([(100,0),(100,100)])
bound4 = LineString([(0,100),(100,100)])
bounds = [bound1,bound2,bound3,bound4]
#----------------------------------------------------------------------------------------------------------
#col = 'b' #default color

#--------- Statistics Variables --------------------------------------------------------------------------
FinalReach = [0]
FinalDie = [0]
FinalRunOut = [0]
FinalEnd = [0]
FinalFitness =[0.001]
#----------------------------------------------------------------------------------------------------------

#--------- Gene pool of particles -------------------------------------------------------------------------
class Brain:
    """
    class Brain: Brain of the dots that basically indicate the acceleration of the dot at the n'th step

    Methods:
        1. __init__:
            a. size: Length of the 2D acceleration array

            Returns None, initializes the Brain object.

        2. mutate:
            a. mutRate: Rate of mutation (0<=mutRate<=1)
            b. mode: Default-> 'random'
                'random': % of brain array provided by mutRate is randomly mutated.
                'slow': % of brain array provided by mutRate is changed according to
                        slewrate (deviation in % from the existing value) and signrate
                        (chance of sign flip on one of the directions).
    """
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

    def mutate(self,mutRate = 0.01,mode='random',slewrate=0.5,signrate=0.5): #slewrate: deviation in % on the selected gene, signrate: sign flip rate of the derived direction
        # if mode == 'slow':
        #     print("Starting slow")
        mutationRate = mutRate #1% chance of mutation
        if mutationRate == 0:
            return
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
                        dev = np.random.random()*2*slewrate-slewrate
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
                        dev = np.random.random()*2*slewrate-slewrate
                    ay = sqrt(1-ax*ax)
                    if np.random.random() <= signrate:
                        ay *= -1.0
                    self.directions[i][1]=ax
                    self.directions[i][0]=ay
        # if mode == 'slow' :
        #     print("End slow")
        
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
        self.bestDot = False
    def move(self,wallpoly):
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
            dist = (dx*dx+dy*dy)
            #print(dist)
            if dist < 16.:
                self.fitness = 1./16.
            else:
                self.fitness = 1./dist
        else:
            self.fitness = 1./16. + 100000./(self.brain.step*self.brain.step)
        #print(self.fitness)

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
    def update(self,wallpoly):
        #self.px = []
        #self.py = []
        self.step += 1
        for z in range(self.size):
            if(self.step<self.maxstep):
                self.dots[z].move(wallpoly)
            else:
                self.dots[z].runOut = True
            #self.px.append(d.pos[0])
            #self.py.append(d.pos[1])
    def calcFitness(self,calcSum=True,setBestDot=False):
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
        if calcSum:
            self.calcFitnessSum()

        if setBestDot:
            self.setBestDot()

    def setBestDot(self):
        max = 0
        maxIndex = 0
        for i in range(self.size):
            if self.dots[i].fitness > max:
                max = self.dots[i].fitness
                maxIndex = i

        if self.dots[maxIndex].reachedGoal:
            self.dots[maxIndex].bestDot = True
            self.maxstep = self.dots[maxIndex].brain.step

        return

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
        global t, wallpoly, wallco1, wallco2
        t = 0; wallpoly=[wallco1,wallco2]
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
        del self.newDots
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
            if self.dots[x].bestDot:
                self.dots[x].brain.mutate(mutRate=0.0)
            else:
                self.dots[x].brain.mutate()
            self.dots[x].bestDot = False
#----------------------------------------------------------------------------------------------------------

#--------- Parallelization Help ---------------------------------------------------------------------
def dot_update(d):
    d.move()
    return d
#----------------------------------------------------------------------------------------------------------

#--------- Begin MAIN -------------------------------------------------------------------------------------
a = Population(200,maxstep=400)

#--------- Animation Figure -------------------------------------------------------------------------------
fig=plt.figure(figsize=(20,10))

fig.suptitle(
    """
    Iteration: %d, Dots: %d, Step: %d, Reached: %d, Dead: %d, Runout: %d.
    """ % (a.generation,a.size,0,0,0,0)
)

outer = grspec.GridSpec(2,2,wspace=0.3,hspace=0.3)

ax = plt.subplot(outer[:,0],adjustable='box',aspect=1.0)
ax.set_title('Particle Trajectories')
ax1 = plt.subplot(outer[0,1])
ax1.set_title('Statistics')
ax1.set_xlabel('Iterations')
ax2 = plt.subplot(outer[1,1])
ax2.set_title('Probabilities')
ax2.set_xlabel('Particle ID')
ax2.set_ylabel('Normalized Fitness')

ax1_xlim = 10

# ax = fig.add_subplot(121)
# ax1 = fig.add_subplot(122)
ax.set_xlim(0,height)
ax.set_ylim(0,width)
#ax.plot([a.pos[0]],[a.pos[1]],marker='o',ls='',markersize=2)
for i in range(a.size):
    ax.plot([a.dots[i].pos[0]],[a.dots[i].pos[1]],marker='o',ls='',markersize=2,color='b')
#ax.scatter([goal[0]],[goal [1]],marker='s',color='g',s=36*fig.dpi/72.)

ax.plot(goalboundx,goalboundy,color='g')
for wall in wallpoly:
    w2,wpy=(move_wall(wall,w_num=0))
    wall = w2
    wx,wy=wpy.exterior.xy
    ax.plot(wx,wy,color='r')


major_ticks = np.arange(0,101,10)
minor_ticks = np.arange(0,101,1)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks,minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks,minor=True)
ax.grid(which='both')
end = 0
animstop = False
#----------------------------------------------------------------------------------------------------------

#--------- Frame Iterator ---------------------------------------------------------------------------------
def update(i):
    global end,animstop,ax1_xlim,t,v
    # with Pool(processes=6) as pool:
    #     newDots = pool.map(dot_update,a.dots,1)
    # a.dots = newDots.copy()

    ax.clear()
    ax.set_title('Particle Trajectories')
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
    wallfeed = []
    for wall in wallpoly:
        w2,wpy=(move_wall(wall))
        wall = w2
        wallfeed.append(wpy)
        wx,wy=wpy.exterior.xy
        ax.plot(wx,wy,color='r')

    a.update(wallfeed)
    a.calcFitness(calcSum=False)
    
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

    ax1.clear()
    ax1.set_title('Statistics')
    ax1.set_xlabel('Iterations')
    if v > 0 :
        if ax1_xlim < len(FinalDie):
            ax1_xlim += 10
        ax1.set_xlim(1,ax1_xlim)
        ax1.plot([x for x in range(v)],FinalDie,label='Deaths')
        ax1.plot([x for x in range(v)],FinalReach,label='Wins')
        ax1.plot([x for x in range(v)],FinalRunOut,label='Out of breath')
        ax1.plot([x for x in range(v)],FinalEnd,label='Simulation length')
        #print(len(FinalFitness))
        fnftns = np.array(FinalFitness)
        fnftns = fnftns/np.max(fnftns)

        ax1.plot([x for x in range(v)],20*fnftns,label='Normalized fitness sum')
        ax1.legend()
        #print(FinalFitness)
    ax1.grid()
    ftns = np.array([a.dots[x].fitness for x in range(len(a.dots))])
    ax2.clear()
    ax2.set_title('Probabilities')
    ax2.set_xlabel('Particle ID')
    ax2.set_ylabel('Normalized Fitness')
    ax2.plot(ftns/np.max(ftns),ls='',marker='o',markersize=2)
    ax2.grid()
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
        v += 1

    if not a.generation > 0:    
        fig.suptitle(
        """
        Iteration: %d, Dots: %d, Step: %d, Reached: %d, Dead: %d, Runout: %d.
        """ % (a.generation,a.size,end,reached,dead,runout)
        )
    else:
        fig.suptitle(
        """
        Previous Iteration: %d, Dots: %d, Step: %d, Reached: %d, Dead: %d, Runout: %d.
        Iteration: %d, Dots: %d, Step: %d, Reached: %d, Dead: %d, Runout: %d.
        """ % (a.generation-1,a.size,FinalEnd[v-1],FinalReach[v-1],FinalDie[v-1],FinalRunOut[v-1],a.generation,a.size,end,reached,dead,runout)
        )
    t += dt

px = anim.FuncAnimation(fig,update,repeat=False,interval=100)
#plt.show(block=True)
plt.show()
# while not animstop:
#     plt.pause(1)
# plt.close()
#----------------------------------------------------------------------------------------------------------

#--------- Statistics Plotter -----------------------------------------------------------------------------
# FinalFitness = np.array(FinalFitness)
# plt.figure(figsize=(10,10))
# plt.plot(FinalDie,label='Deaths')
# plt.plot(FinalReach,label='Wins')
# plt.plot(FinalRunOut,label='Out of breath')
# plt.plot(FinalEnd,label='Simulation length')
# plt.plot(50*FinalFitness/(np.max(np.array(FinalFitness))),label='Normalized fitness sum')
# plt.grid()
# plt.legend()
# plt.show()
#----------------------------------------------------------------------------------------------------------