from pylab import *
from scipy.integrate import quad
from scipy.optimize import fsolve,root
from scipy.interpolate import interp1d
import pickle

eps = np.finfo(float).eps

# Since it is not easy to follow units outside of Brian,
#  pass only unitless variables here i.e. convert all to SI and pass in

class IntegralPop():
    def __init__(self,N,h0,tau0,uth,uth_base,noise,w0,tausyn,fconn,win,rate0):
        self.N = N
        self.w0 = w0
        self.win = win
        self.h0 = h0
        self.rate0 = rate0
        self.tau0 = tau0
        self.uth = uth
        self.uth_base = uth_base
        self.noise = noise
        self.tausyn = tausyn
        self.fconn = fconn                  # fraction connected

        # hazard and kernel integration steps
        self.integrate_dt = 1e-4            # seconds

        # upper limit of integration for survivor integral
        #  depends on how fast survivor is dropping.
        # check how fast the survivor integral is dropping
        #  and use it to set the upper limit
        #  for the survivor integral (see comments in g_sigma()).
        self.tupper = 0.1
        survivor_neglect = 1e-4
        while self.tupper<1e100:
            if self.survivor([self.tupper],h0)[0] < survivor_neglect:
                break # use this value of tupper
            self.tupper *= 2.0
        survivor_integrationsteps = 1000
        self.fullt = linspace(0.,self.tupper,survivor_integrationsteps)
        self.fullt_dt = self.fullt[1]-self.fullt[0]
        print "Survivor probability drops to <",survivor_neglect,\
            "by tupper =",self.tupper,"seconds."

        self.kernelInf = 1.0                # infinity for kernel integrations
        xkernelrange = arange(0,self.kernelInf,self.integrate_dt)
        print 'Calculating syn-tilde kernel (for input rate)'
        kernelsyntildeList = [self.kernelsyntilde(t) for t in xkernelrange]
        self.kernelsyntildeInterp = interp1d(xkernelrange,\
                                    kernelsyntildeList,kind='linear',\
                                    bounds_error=False,fill_value=0.)
                                            # accepts vector arguments
        self.kernelsyntildeIntegral = trapz(kernelsyntildeList,dx=self.integrate_dt)
        print 'Calculating memb*syn-tilde kernel (for adaptation): unused!'
        kernelsmtildeList = [self.kernelsmtilde(t) for t in xkernelrange]
        self.kernelsmtildeInterp = interp1d(xkernelrange,\
                                    kernelsmtildeList,kind='linear',\
                                    bounds_error=False,fill_value=0.)
                                            # accepts vector arguments
        self.kernelsmtildeIntegral = trapz(kernelsmtildeList,dx=self.integrate_dt)

        print 'Calculating ISI (P) distibution'
        self.ISIdistrib = self.ISI(self.fullt,self.h0)

    def kernelsyn(self,t):
        return (t>0)*np.exp(-t/self.tausyn)
                                            # not normalized by 1/tausyn
                                            # (t>0) acts as Heaviside function

    def kernelsyntilde(self,t):
        return np.exp(self.win/self.noise*self.kernelsyn(t))-1

    def kernelmemb(self,t):
        return (t>0)*np.exp(-t/self.tau0)
                                            # not normalized by 1/tau0
                                            # (t>0) acts as Heaviside function

    def kernelsm(self,t):                   # memb and syn kernels convolved
        tarray = arange(-self.kernelInf,t,self.integrate_dt)
        return trapz(self.kernelsyn(tarray)*self.kernelmemb(t-tarray),\
                        dx=self.integrate_dt)
                        
    def kernelsmtilde(self,t):
        return np.exp(self.win/self.noise*self.kernelsm(t))-1
        
    def hazard(self,s,h):
        """
            uth_base is a constant threshold,
            uth gets bumped up after each reset.
        """
        ##### might want to make a lookup table wrt s of this for speed?
        uth_at_s = self.uth*exp(-s/self.tau0)
        ##### might want to make a lookup table wrt h-uth of this for speed?
        return 1.0/self.tau0*exp((h-(uth_at_s+self.uth_base))/self.noise)
        #return (s>2e-3) * 200              # 200 Hz * H(s-refract)
                                            # constant hazard with refractory period    

    def survivor(self,s,h):
        ''' survivor function (unitless) is a survival probability (S=1 at t=t_hat)
            (it is not a probability density, hence not integral-normalized)
            s is time since last spike (no brian units here)
            For constant h0 currently. '''
        #return exp(-quad(self.hazard,0.0,s,args=(h,))[0])
        #                                    # quad returns (integral, errorbound)
        return array([exp(-trapz(
                            self.hazard(arange(0.,t,self.integrate_dt),h),\
                            dx=self.integrate_dt)) \
                        for t in s])

    def ISI(self,s,h):
        return self.hazard(s,h)*self.survivor(s,h)
        
    def ISIprime(self,s,h):
        return -self.hazard(s,h)**2.0 * self.survivor(s,h)

    def g_sigma(self,h):
        # SI units second for integration variable
        # quad returns (integral, errorbound), hence take [0]
        #survivor_integral = quad(self.survivor,0.0,10.0,args=(h,))[0]
        # Ideally, I should integrate to infinity `inf`,
        # but for low input, survivor ~= 1, and the integral diverges.
        # If you integrate to 1e4, then for reasonable input that causes spiking,
        #  the constraint optimization doesn't converge.
        # Better instead to see how fast the survivor integral is dropping
        #  and use it to set the upper limit, thus fullt is used (see __init__).
        survivor_integral = sum(self.survivor(self.fullt,h))*self.fullt_dt
        if survivor_integral > eps:
            return 1.0/survivor_integral
        else: return 1.0/eps

    def constraint(self,args):
        A0 = args[0]
        # A0 assumed in Hz,
        # Need to use synaptic weight*tausyn/1second to obtain avg of exp syn
        self.h = self.h0 + self.w0*self.tausyn*self.fconn*self.N*A0 + \
                    self.rate0*self.kernelsyntildeIntegral*self.noise
                                        # current input h + recurrent self-activity h
                                        # + rate input h
                                        # Note win/noise is already inside kernelsyntilde
                                        #  hence *noise to cancel /noise in g_sigma->hazard

        return ( A0 - self.g_sigma( self.h ), )

    def get_background_rate(self):
        answer = fsolve(self.constraint,[0,],full_output=True)
        self.A0 = answer[0][0]
        print answer[-1]                # see that it converged
        self.constraint((self.A0,))      # sets final self.h for future
        return self.A0

    def h_from_rate(self,dratearray,dt):
        return dt*array([ np.sum( \
                    self.kernelsyntildeInterp(arange(min(int(self.kernelInf/dt),tidx),0,-1)*dt)\
                    *(self.rate0+dratearray[max(tidx-int(self.kernelInf/dt),0):tidx]) \
                    *self.noise) for tidx in range(len(dratearray)) ] )
                    # Note input weight: win/noise is already inside kernelsyntilde
                    #  hence *noise here to cancel /noise in g_sigma->hazard

    def evolve(self,tarray,dratearray,mpoints):
        '''
        tarray is an array of times with constant spacing dt
        ratearray is the incoming rate at above time points
        (after convolution with kernelsyntilde)
        mpoints are number of dt-s to hold population activity memory
        '''
        dt = tarray[1]-tarray[0]
        tlen = len(tarray)
        self.harray = self.h_from_rate(dratearray,dt) + \
                        self.h0 + \
                        self.w0*self.tausyn*self.fconn*self.N*self.A0
                                            # h due to input rate (rate0+dratearray)
                                            # basic current input h
                                            # h due to recurrent self-activity
        mvec = zeros(mpoints)               # rolling A*dt: see Wulfram's book2 sec 14.1.5
        mvec[0] = 1.0                       # all neurons have just fired
        Avec = zeros(tlen)                  # population activity
        for i,t in enumerate(tarray):
            hhere = self.harray[i]
            #for k in range(mpoints-1,0,-1): # have to go in reverse order!
            #    mvec[k] = mvec[k-1]*np.exp(-self.hazard(k*dt,hhere)*dt)
            mvec[1:] = mvec[:-1]*np.exp(-self.hazard(arange(1,mpoints,1)*dt,hhere)*dt)
            mvec[0] = 1 - np.sum(mvec[1:])  # note: don't subtract mvec[0]
            Avec[i] = mvec[0]/dt
        return Avec

    def L_rho(self,x):
        '''
        returns linear filter vs rate input based on linearization of exp in rho,
        after using the moments expansion to take <exp(k*rate)> into exp(k~*<rate>).
        '''
        return self.kernelsyntildeInterp(x) # interp1d has already been
                                             # specified to return 0 beyond either end

    def lin_response_rate(self,ratearray,dt):
        tarray = arange(0.0,self.kernelInf,dt)
        ktildearray = self.kernelsyntildeInterp(tarray)
        return self.A0*(1+dt*convolve(ratearray,ktildearray)[:len(ratearray)])


class IntegralPops(IntegralPop):
    def __init__(self,tau0,uth,uth_base,noise,Ns,h0s,w0s,tausyns,fconns):
        IntegralPop.__init__(self,Ns[0],h0s[0],tau0,uth,uth_base,noise,\
                                w0s[0,0],tausyns[0],fconns[0,0])
        self.Ns = Ns
        self.h0s = h0s
        self.w0s = w0s
        self.tausyns = tausyns
        self.fconns = fconns

    def g_sigma(self,h):
        # SI units second for integration variable
        # over-ridden to handle 1D array input for h
        survivor_integral = array([ quad(self.survivor,0.0,10.0,args=(hi,))[0] \
                                            for hi in h ])
        survivor_integral[survivor_integral<eps] = eps
        return 1/survivor_integral

    def constraint(self,args):
        A0s = args
        # A0 assumed in Hz,
        # Need to use synaptic weight*tausyn/second to obtain avg of exp syn
        self.hs = self.h0s + dot(self.w0s*self.fconns,self.tausyns*self.Ns*A0s)
        return A0s - self.g_sigma( self.hs )

    def get_background_rate(self,init_nus):
        answer = root(self.constraint,init_nus,method='krylov')
                                            # only resolution of 0.1Hz needed
        self.A0s = answer.x
        print answer.message               # see if it converged
        self.A0s = self.g_sigma( self.hs )
        self.constraint(self.A0s)           # compute self.hs for future
        return self.A0s

if __name__ == '__main__':
    # neuronal constants
    R = 1.0e8               # Ohm
    tausyn = 100.0e-3       # s
    tau0 = 20.0e-3          # s
    noise = 5.0e-3          # V
    uth = 20.0e-3           # V
    
    # network constants
    N = 10000
    connprob = 0.1
    I0 = 10.0e-3/R          # V/Ohm = A
    #totalw_pernrn = 15.0e-3 # V, recurrent weight
    totalw_pernrn = 0.0e-3  # V, recurrent weight
                            # if I0 = 10mV/R, and noise = 5 mV,
                            #  then totalw_pernrn = 15mV/R is ~ limit
                            #  before activity blow up at 20mV/R
    w0 = totalw_pernrn/connprob/N
    win = 1.0e-3            # V, input weight for rate below

    # stimulus constants
    rate0 = 10              # Hz
    ratemod = 5             # Hz
    stimfreq = 5            # Hz
    fullt = 1.0             # s

    intPop = IntegralPop(N,I0*R,tau0,\
                uth,0.0,noise,\
                w0,tausyn,connprob,win,rate0)

    dt = intPop.integrate_dt
    mpoints = int(intPop.tupper/dt)
    tarray = arange(0,fullt,dt)
    dratearray = ratemod*sin(2*pi*stimfreq*tarray)
    
    intPop.get_background_rate()
    print "The background rate is",intPop.A0
    
    print 'Evolving rate input'
    Avec = intPop.evolve(tarray,dratearray,mpoints)
    
    print 'Convolving linear rate input'
    Aveclin = intPop.lin_response_rate(dratearray,dt)

    figure()
    plot(intPop.harray,color='blue')
    ylabel('h (V)',color='blue')
    twinx()
    plot(Avec,color='red')
    plot(Aveclin,color='magenta')
    ylabel('rate (Hz)',color='red')
    
    figure()
    # same points at which the interplolation function was defined
    #  so essentially, no interpolation.
    tarray = arange(0.0,intPop.kernelInf,intPop.integrate_dt)
    linizeExpMomentKernel = intPop.kernelsyntildeInterp(tarray)
    plot(tarray,intPop.A0*linizeExpMomentKernel)
    
    fh = open('linizeExpMomentKernel.pickle','wb')
    pickle.dump((intPop.A0, tarray, linizeExpMomentKernel),fh)
    fh.close()
    
    show()
