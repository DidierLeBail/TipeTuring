import numpy as np
import numpy.random as rd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

class milieuReactionnel():
	
	cInitialeReactifsArray = [] #c initiales [A,B,D,E,F,H,S,T]
	diffusions = [] #diffusions des reactifs (nul pour H,S,T)
	dx = 0 #pas suivant x
	dy = 0 #pas suivant y
	Nx = 0 #dimension en x du milieu
	Ny = 0 #dimension en y du milieu
	Nz = 0 #nombre de reactifs
	milieuArray = [] #contient positions suivant x et y et concentrations suivant z
	kVitesses = [] #contient des constantes de vitesse [k1a,k1b,k2,k3a,k3b,k4,k5]

	def __init__(this,_cInitialeReactifsArray,_diffusions,_dx,_dy,_nx,_ny,_kVitesses,_alpha):
		this.cInitialeReactifsArray = list(_cInitialeReactifsArray)
		this.kVitesses = _kVitesses
		this.diffusions = list(_diffusions)
		this.alpha = _alpha
		this.dx = _dx
		this.dy = _dy
		this.Nx = int(_nx/_dx)
		this.Ny = int(_ny/_dy)
		this.Nz = len(_cInitialeReactifsArray)
		#Remplissage du milieu
		this.milieuArray=np.zeros((this.Nx,this.Ny,this.Nz))
		for x in range(this.Nx):
			for i,c in enumerate(_cInitialeReactifsArray):
				this.milieuArray[x,this.Ny-1,i]=c
				this.milieuArray[x,0,i]=c
		for y in range(this.Ny):
			for i,c in enumerate(_cInitialeReactifsArray):
				this.milieuArray[this.Nx-1,y,i]=c
				this.milieuArray[0,y,i]=c
	
	def __str__(this):
		output = ""
		i=0;j=0
		for x in this.milieuArray:
			for y in x:
				output+="("+str(i)+","+str(j)+")"
				j+=1
				output+=str(y)+"\n"
			i+=1
			j=0
		output+="\nDiffusions:"+str(this.diffusions)+"\n"
		output+="Constantes de vitesse:"+str(this.kVitesses)
		return output

			
	def progConcentrationtdt(this,dt,lap):
		
		k1a,k1b,k2,k3a,k3b,k4,k5 = tuple(this.kVitesses)
		da,db,dd,de,df = tuple(this.diffusions)

		for x in range(this.Nx):
			for y in range(this.Ny):
				r1=(k1a*this.milieuArray[x,y][0]*this.milieuArray[x,y][1])/(k1b+this.milieuArray[x,y][1]) ; r3=k3a*this.milieuArray[x,y][4]*this.milieuArray[x,y][2]*this.milieuArray[x,y][5] + (k3b*this.milieuArray[x,y][4]*this.milieuArray[x,y][1]*this.milieuArray[x,y][2])/(this.alpha+this.milieuArray[x,y][2]**2)
				r2=k2*this.milieuArray[x,y][2]*this.milieuArray[x,y][3] ; r4=k4*this.milieuArray[x,y][6]*this.milieuArray[x,y][1]*this.milieuArray[x,y][2]-k5*this.milieuArray[x,y][5]
				this.milieuArray[x,y][0]+=dt*(-r1+da*lap[x,y][0])
				this.milieuArray[x,y][1]+=dt*(-r1+0.5*r2+2*r3-r4+db*lap[x,y][1])
				this.milieuArray[x,y][2]+=dt*(r1+r2-4*r3-r4+dd*lap[x,y][2])
				this.milieuArray[x,y][3]+=dt*(-r2+de*lap[x,y][3])
				this.milieuArray[x,y][4]+=dt*(r2-r3+df*lap[x,y][4])
				this.milieuArray[x,y][5]+=dt*(r1-4*r3)
				this.milieuArray[x,y][6]+=-dt*r4
				this.milieuArray[x,y][7]+=dt*r4
				if this.milieuArray[x,y][0]<0:
					this.milieuArray[x,y][0]=0
				if this.milieuArray[x,y][1]<0:
					this.milieuArray[x,y][1]=0
				if this.milieuArray[x,y][2]<0:
					this.milieuArray[x,y][2]=0
				if this.milieuArray[x,y][3]<0:
					this.milieuArray[x,y][3]=0
				if this.milieuArray[x,y][4]<0:
					this.milieuArray[x,y][4]=0
				if this.milieuArray[x,y][5]<0:
					this.milieuArray[x,y][5]=0
				if this.milieuArray[x,y][6]<0:
					this.milieuArray[x,y][6]=0
				if this.milieuArray[x,y][7]<0:
					this.milieuArray[x,y][7]=0
			
		for x in range(1,this.Nx-1):
			for y in range(1,this.Ny-1):
				for i in range(this.Nz):
					lap[x,y,i]=(this.milieuArray[x+1,y][i]-2*this.milieuArray[x,y][i]+this.milieuArray[x-1,y][i])/((this.dx)**2)
					+(this.milieuArray[x,y+1][i]-2*this.milieuArray[x,y][i]+this.milieuArray[x,y-1][i])/((this.dy)**2)
		for y in range(1,this.Ny-1):
			for i in range(this.Nz):
				lap[this.Nx-1,y,i]=((this.cInitialeReactifsArray[i]-2*this.milieuArray[this.Nx-1,y][i]+this.milieuArray[this.Nx-2,y][i])/((this.dx)**2))+(this.milieuArray[this.Nx-1,y+1][i]-2*this.milieuArray[this.Nx-1,y][i]+this.milieuArray[this.Nx-1,y-1][i])/((this.dy)**2)
				lap[0,y,i]=((this.cInitialeReactifsArray[i]+this.milieuArray[1,y,i]-2*this.milieuArray[0,y][i])/((this.dx)**2))+(this.milieuArray[0,y+1][i]-2*this.milieuArray[0,y][i]+this.milieuArray[0,y-1][i])/((this.dy)**2)
		for x in range(1,this.Nx-1):
			for i in range(this.Nz):
				lap[x,this.Ny-1,i]=((this.cInitialeReactifsArray[i]-2*this.milieuArray[x,this.Ny-1][i]+this.milieuArray[x,this.Ny-2][i])/((this.dy)**2))+(this.milieuArray[x+1,this.Ny-1][i]-2*this.milieuArray[x,this.Ny-1][i]+this.milieuArray[x-1,this.Ny-1][i])/((this.dx)**2)
				lap[x,0,i]=((this.cInitialeReactifsArray[i]+this.milieuArray[x,1][i]-2*this.milieuArray[x,0,i])/((this.dy)**2))+(this.milieuArray[x+1,0][i]-2*this.milieuArray[x,0][i]+this.milieuArray[x-1,0][i])/((this.dx)**2)
		for i in range(this.Nz):
			lap[0,this.Ny-1,i]=((this.cInitialeReactifsArray[i]+this.milieuArray[1,this.Ny-1,i]-2*this.milieuArray[0,this.Ny-1][i])/((this.dx)**2))+((this.cInitialeReactifsArray[i]-2*this.milieuArray[0,this.Ny-1][i]+this.milieuArray[0,this.Ny-2][i])/((this.dy)**2))
			lap[this.Nx-1,0,i]=((this.cInitialeReactifsArray[i]-2*this.milieuArray[this.Nx-1,0][i]+this.milieuArray[this.Nx-2,0][i])/((this.dx)**2))+((this.cInitialeReactifsArray[i]+this.milieuArray[this.Nx-1,1][i]-2*this.milieuArray[this.Nx-1,0,i])/((this.dy)**2))
			lap[this.Nx-1,this.Ny-1,i]=((this.cInitialeReactifsArray[i]-2*this.milieuArray[this.Nx-1,this.Ny-1][i]+this.milieuArray[this.Nx-2,this.Ny-1][i])/((this.dx)**2))+((this.cInitialeReactifsArray[i]-2*this.milieuArray[this.Nx-1,this.Ny-1][i]+this.milieuArray[this.Nx-1,this.Ny-2][i])/((this.dy)**2))
			lap[0,0,i]=((this.cInitialeReactifsArray[i]+this.milieuArray[1,0,i]-2*this.milieuArray[0,0][i])/((this.dx)**2))+((this.cInitialeReactifsArray[i]+this.milieuArray[0,1][i]-2*this.milieuArray[0,0,i])/((this.dy)**2))
		return lap

	def initLaplacien(this):
		lap=np.zeros((this.Nx,this.Ny,this.Nz))
		for i in range(this.Nz):
			for y in range(1,this.Ny-1):
				lap[this.Nx-1,y][i]=this.cInitialeReactifsArray[i]/((this.dx)**2)
				lap[0,y][i]=lap[this.Nx-1,y][i]
			for x in range(1,this.Nx-1):
				lap[x,this.Ny-1][i]=this.cInitialeReactifsArray[i]/((this.dy)**2)
				lap[x,0][i]=lap[x,this.Ny-1][i]
			lap[this.Nx-1,this.Ny-1][i]=(this.cInitialeReactifsArray[i]/((this.dx)**2))+(this.cInitialeReactifsArray[i]/((this.dy)**2))
			lap[this.Nx-1,0][i]=lap[this.Nx-1,this.Ny-1,i]
			lap[0,this.Ny-1][i]=lap[this.Nx-1,this.Ny-1,i]
			lap[0,0][i]=lap[this.Nx-1,this.Ny-1,i]
		return lap

	def grapheEnFonctiondeT(this,dt,duree,x0,y0):
		Nt=int(duree/dt) ; instants=np.linspace(0,duree,Nt)
		X0=[this.milieuArray[x0,y0][2]]
		lap=this.initLaplacien()
		for i in range(1,Nt):
			lap=this.progConcentrationtdt(dt,lap)
			X0.append(this.milieuArray[x0,y0][2])
		print(this)
		fig=plt.figure()
		plt.grid(True)
		plt.plot(instants,X0)
		#plt.savefig()
		plt.show()

## Begin script


milieu = milieuReactionnel([21*10**(-3),2.2*10**(-3),2.2*10**(-3),22*10**(-3),22*10**(-3),20*10**(-3),4.5*10**(-3),0.0],[0.0,0.0,7*10**(-10),0.0,7.5*10**(-6)],0.01,0.01,0.05,0.05,[6.2*10**(-4),5*10**(-5),900.0,9.2*10**(-5),9.2*10**(-5),1.0*10**(-3),1.0],5*10**(-13))
milieu.grapheEnFonctiondeT(0.01,10.0,1,1)