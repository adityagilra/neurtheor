# Methods from Richardson 2007 Appendix A; Trousdale et al 2012
#  (see also Gerstner book 2014; Ocker, Litwin-Kumar, Doiron 2014)
# Obtain linear response function of cell
# Differential equation method
# Concrete Example
# Put in concrete values for all constants

from pylab import *
from utils import *

class DiffLIF():
    """ Using Richardson 2007 Appendix A
    LIF membrane potential distribution described by FP equation
    J0, P0:
        Numerical integration of unperturbed steady state FP equations
    J1free, J1driven:
        Numerical integration of linear response to small perturbation
    """

    def __init__(self,v_leak=-65.0e-3,tau_m=20.0e-3,\
                    h0=20e-3,sigma=2.0e-3,\
                    theta_reset=-45.0e-3,v_reset=-55.0e-3,tau_ref=1e-3,\
                    v_low=-80.0e-3,nsteps=1000):
        # h0=20mV is exactly at the threshold
        self.v_leak = v_leak
        self.tau_m = tau_m
        self.sigma = sigma
        self.theta_reset = theta_reset
        self.v_reset = v_reset
        self.tau_ref = tau_ref
        self.v_low = v_low
        self.nsteps = nsteps

        self.dv = ((theta_reset-v_low))/nsteps
        self.h0 = h0
        self._nreset = int((theta_reset-v_reset)/self.dv)
        
        self.bgndrate = None # call calcJ0P0 to set this

    def f(self,u):
        # LIF
        return -(u-self.v_leak)
        
        # PIF = perfect IF, below
        # Note that v_leak drops out as the leak branch goes out.
        # But I'm still using tau_m, h0 etc. to get Subscript[C, m] .
        #return 0.0

    def calcJ0P0(self):
        j0 = 1.0
        p0 = 0.0
        v0list = zeros(self.nsteps)
        j0list = zeros(self.nsteps)
        p0list = zeros(self.nsteps)
        self.Alist = zeros(self.nsteps)
        self.Blist = zeros(self.nsteps)
        v0 = self.theta_reset       # integrate from theta_reset backwards
        for niter in range(self.nsteps):
            v0 = v0-self.dv
            v0list[niter] = v0
            Gn = 2.0*(-self.f(v0)-self.h0)/self.sigma**2
            An = exp(self.dv*Gn)
            Bn = 2.0/self.sigma**2 * (An-1.0)/Gn
            self.Alist[niter] = An
            self.Blist[niter] = Bn
            p0 = p0*An + Bn*self.tau_m*j0 # before updating j0
            j0 = j0 - KroneckerDelta(niter,self._nreset)
            j0list[niter] = j0
            p0list[niter] = p0
        self.p0integral = sum(p0list)*self.dv
        p0list /= self.p0integral    # normalize the probability distribution
        self.j0list = j0list
        self.p0list = p0list
        self.v0list = v0list
        
        self.bgndrate = 1.0/(self.p0integral+self.tau_ref)

    def calcJ1P1(self,omegalist):
        """
        Numerical Integration of perturbed FP eqn:
         Full solution, looping over omega
        From Richardson 2007 Appendix A
        """
        #omegalist = array([10**omegaexp for omegaexp in arange(-2,3,0.1)])
        gainlist = zeros(len(omegalist),dtype=complex)
        for oiter,omega in enumerate(omegalist):
            j1free = 1.0
            p1free = 0.0
            v0 = self.theta_reset     # integrate from theta_reset backwards
            for niter in range(self.nsteps):
                v0 = v0-self.dv
                p1freeSaved = p1free
                p1free = p1free*self.Alist[niter] + \
                                self.Blist[niter]*self.tau_m*j1free
                                  # before updating j1free
                j1free = j1free + self.dv*1j*omega*p1freeSaved - \
                                KroneckerDelta(niter,self._nreset)
            j1driven = 0.0
            p1driven = 0.0
            v0 = self.theta_reset
            for niter in range(self.nsteps):
                v0 = v0-self.dv
                p1drivenSaved = p1driven
                p1driven = p1driven*self.Alist[niter] + self.tau_m*j1driven + \
                                self.p0list[niter]*self.Blist[niter]   
                                                    # before updating j1driven 
                                                    # epsilon set to 1,
                                                    # epsilon has voltage units
                j1driven = j1driven + self.dv*1j*omega*p1drivenSaved
            gainlist[oiter] = -j1driven/j1free
        self.freqlist = omegalist/(2*pi)
        self.susceptibility = gainlist

    def calcRate(self):        
        ##### adapted from code accompanying Trousdale et al Plos CB 2012 #####
        inv_sigmasq = 1./(self.sigma*self.sigma)
      
        p0 = zeros(self.nsteps)
        j0 = zeros(self.nsteps)
        v0 = zeros(self.nsteps)
        
        # Set the final conditions 
        p0[-1] = 0
        j0[-1] = 1
        v = self.theta_reset    # integrate from theta_reset (v_th) backwards
        above_reset = True     # J0 must reduce by 1 when v goes below reset
        v0[-1] = v
        # Solve backwards
        for i in range(self.nsteps-1,0,-1):
            G = (-self.f(v)-self.h0)*inv_sigmasq
            
            if v<self.v_reset and above_reset:
                j0[i-1] = j0[i] - 1
                above_reset = False
            else:
                j0[i-1] = j0[i]
            
            p0[i-1] = p0[i]*exp(self.dv*G) + \
                        self.tau_m*j0[i]*inv_sigmasq*(exp(self.dv*G)-1)/G
            
            v -= self.dv
            v0[i-1] = v            
        
        sum_p0 = sum(p0)
        self.p0list = p0
        self.j0list = j0
        self.v0list = v0
            
        self.bgndrate = 1. / (self.dv*sum_p0 + self.tau_ref)


    def calcPowerSpectrum(self,omegalist):
        ##### adapted from code accompanying Trousdale et al Plos CB 2012 #####
        self.power = zeros(len(omegalist))
        
        N = int((self.theta_reset-self.v_low)/self.dv) + 1
        inv_sigmasq = 1./(self.sigma**2.)
        inv_tau_m = 1./self.tau_m

        pf_r = zeros(N)
        pf_i = zeros(N)
        
        jf_r = zeros(N)
        jf_i = zeros(N)
        
        p0_r = zeros(N)
        p0_i = zeros(N)
        
        j0_r = zeros(N)
        j0_i = zeros(N)
        
        # For each frequency value,
        #  solve perturbed FP for power at that frequency.
        for i,w in enumerate(omegalist):
            
            # Set the final conditions
            pf_r[N-1] = 0
            pf_i[N-1] = 0
            jf_r[N-1] = 1
            jf_i[N-1] = 0
            
            p0_r[N-1] = 0
            p0_i[N-1] = 0
            j0_r[N-1] = 0
            j0_i[N-1] = 0
            
            v = self.theta_reset
        
            # Solve the perturbed FP equation
            for j in range(N-1,0,-1):
                #psi = DT*exp((v-self.theta_reset)/DT) # for exp IF
                #G = (v-E0-psi)*inv_sigmasq
                G = (v-self.h0-self.v_leak)*inv_sigmasq
                if G==0.: G=eps*inv_sigmasq
                
                jf_r[j-1] = jf_r[j] - self.dv*w*2*pi*pf_i[j]
                jf_i[j-1] = jf_i[j] + self.dv*w*2*pi*pf_r[j]
                
                pf_r[j-1] = pf_r[j]*exp(self.dv*G) + \
                            self.tau_m*jf_r[j]*inv_sigmasq*(exp(self.dv*G)-1)/G
                pf_i[j-1] = pf_i[j]*exp(self.dv*G) + \
                            self.tau_m*jf_i[j]*inv_sigmasq*(exp(self.dv*G)-1)/G
                
                if (abs(v-self.v_reset)<self.dv/100.):
                    j0_r[j-1] = j0_r[j] - self.dv*w*2*pi*p0_i[j] - \
                                            cos(-w*2*pi*tau_ref)
                    j0_i[j-1] = j0_i[j] + self.dv*w*2*pi*p0_r[j] - \
                                            sin(-w*2*pi*tau_ref)
                else:
                    j0_r[j-1] = j0_r[j] - self.dv*w*2*pi*p0_i[j]
                    j0_i[j-1] = j0_i[j] + self.dv*w*2*pi*p0_r[j]
                
                p0_r[j-1] = p0_r[j]*exp(self.dv*G) + \
                            self.tau_m*j0_r[j]*inv_sigmasq*(exp(self.dv*G)-1)/G
                p0_i[j-1] = p0_i[j]*exp(self.dv*G) + \
                            self.tau_m*j0_i[j]*inv_sigmasq*(exp(self.dv*G)-1)/G
                
                v = v-self.dv
        
            pw_r = (-(pow(j0_r[0],2) + pow(j0_i[0],2)) - j0_r[0]*jf_r[0] - \
                                    j0_i[0]*jf_i[0])/(pow(j0_r[0] + jf_r[0],2) + \
                                    pow(j0_i[0] + jf_i[0],2))
            
            self.power[i] = self.bgndrate*(1 + 2*pw_r)

    def calcSusceptibility(self,omegalist):
        ##### adapted from code accompanying Trousdale et al Plos CB 2012 #####
        length_w = len(omegalist)
        self.susc_r = zeros(length_w)
        self.susc_i = zeros(length_w)
        
        N = int((self.theta_reset-self.v_low)/self.dv) + 1
        inv_sigmasq = 1./(self.sigma**2.)
        
        pr_r = zeros(N)
        pr_i = zeros(N)
        
        jr_r = zeros(N)
        jr_i = zeros(N)
        
        p0_r = zeros(N)
        p0_i = zeros(N)
        
        j0_r = zeros(N)
        j0_i = zeros(N)
        
        pe_r = zeros(N)
        pe_i = zeros(N)
        
        je_r = zeros(N)
        je_i = zeros(N)
        
        # For each frequency value,
        #  solve perturbed FP eqn for gain at that frequency.
        for i,w in enumerate(omegalist):
            
            # Set the final conditions
            pr_r[N-1] = 0
            pr_i[N-1] = 0
            jr_r[N-1] = 1
            jr_i[N-1] = 0
            
            p0_r[N-1] = 0
            p0_i[N-1] = 0
            j0_r[N-1] = 1
            j0_i[N-1] = 0
            
            pe_r[N-1] = 0
            pe_i[N-1] = 0
            je_r[N-1] = 0
            je_i[N-1] = 0
            
            v = self.theta_reset
        
            # Solve the perturbed FP eqn
            for j in range(N-1,0,-1):
                #psi = DT*exp((v-v_t)/DT)       # for expIF
                #G = (v-E0-psi)*inv_sigmasq
                G = (v-self.h0-self.v_leak)*inv_sigmasq
                if abs(G)<eps*inv_sigmasq:
                    G = copysign(eps*inv_sigmasq,G) # sign(G)*eps*inv_sigmasq
                
                # Iterate j_r/p_r
                
                if abs(v-self.v_reset)<self.dv/100.:
                    jr_r[j-1] = jr_r[j] - self.dv*w*2*pi*pr_i[j] - \
                                cos(-w*2*pi*self.tau_ref)
                    jr_i[j-1] = jr_i[j] + self.dv*w*2*pi*pr_r[j] - \
                                sin(-w*2*pi*self.tau_ref)
                else:
                    jr_r[j-1] = jr_r[j] - self.dv*w*2*pi*pr_i[j]
                    jr_i[j-1] = jr_i[j] + self.dv*w*2*pi*pr_r[j]

                pr_r[j-1] = pr_r[j]*exp(self.dv*G) + \
                            self.tau_m*inv_sigmasq*jr_r[j]*(exp(self.dv*G)-1)/G
                pr_i[j-1] = pr_i[j]*exp(self.dv*G) + \
                            self.tau_m*inv_sigmasq*jr_i[j]*(exp(self.dv*G)-1)/G
                
                # Iterate j_0/p_0
                 
                if abs(v-self.v_reset)<self.dv/100.: 
                    j0_r[j-1] = j0_r[j] - 1
                else:
                    j0_r[j-1] = j0_r[j]
                
                p0_r[j-1] = p0_r[j]*exp(self.dv*G) + \
                            self.tau_m*inv_sigmasq*j0_r[j]*(exp(self.dv*G)-1)/G
                p0_i[j-1] = p0_i[j]*exp(self.dv*G) + \
                            self.tau_m*inv_sigmasq*j0_i[j]*(exp(self.dv*G)-1)/G
                
                
                # Iterate j_E/p_E
                
                je_r[j-1] = je_r[j] - self.dv*w*2*pi*pe_i[j]
                je_i[j-1] = je_i[j] + self.dv*w*2*pi*pe_r[j]
                
                pe_r[j-1] = pe_r[j]*exp(self.dv*G) + (self.tau_m*je_r[j]-\
                                self.bgndrate*p0_r[j])*inv_sigmasq*(exp(self.dv*G)-1)/G
                pe_i[j-1] = pe_i[j]*exp(self.dv*G) + (self.tau_m*je_i[j]-\
                                self.bgndrate*p0_i[j])*inv_sigmasq*(exp(self.dv*G)-1)/G
                
                v = v-self.dv
            
            self.susc_r[i] = -(je_r[0]*jr_r[0] + je_i[0]*jr_i[0])/(pow(jr_r[0],2) + pow(jr_i[0],2))
            self.susc_i[i] = (je_r[0]*jr_i[0] - je_i[0]*jr_r[0])/(pow(jr_r[0],2) + pow(jr_i[0],2))   

        self.susceptibilityW = self.susc_r + 1J*self.susc_i
        
        self.tsusclist,self.susceptibilityT = \
                    real(inv_f_trans_on_vector(omegalist,self.susceptibilityW))

def matlab_josic_trousdale_compare(w):
    """
    NOT COMPARABLE NOW SINCE I'm USING EIF params below, while my DiffLIF only handles LIF!
    Can subclass it as DiffEIF! To do later.
    """
    E0 = -54e-3                 # Effective rest potential (V)
    sigma = sqrt(12e-3)         # Noise variance (V)
    tau_ref = 2e-3              # Absolute refractory period (s)                # to use this for J0 & P0
    v_reset = -54e-3            # Post-spike reset potential (V)
    v_t = -52.5e-3              # Soft threshold for the EIF model (V)          # I don't use this
    v_th = 20e-3                # Spiking threshold (V)                         # I don't use this
    DT = 1.4e-3                 # Spike shape parameter for the EIF model (V)   # I don't use this
    tau_m = 20e-3               # Membrane time constant (s)
    vlb = -100e-3               # Lower bound on membrane potential
                                #  to use in solving for the statistics
    dv = 10e-6                  # Membrane potential step to use in solving
                                #  for the statistics

    v_leak = -65e-3             # resting potential (V)
    R = 20.0e6                  # Input resistance (Ohms)                       # unused, see h0
    current0 = (E0-v_leak)/R    # mean input current (amp)                      # unused, see h0
    h0 = (E0-v_leak)            # mean input (volt)                             # h0 = current0*R
    theta_reset = -45e-3        # I use this as I use LIF, not EIF
    nsteps = int((v_th-vlb)/dv) # number of steps for integation

    diffLIF = DiffLIF(v_leak,tau_m,\
                    h0,sigma,\
                    v_th,v_reset,tau_ref,\
                    vlb,nsteps)
    diffLIF.calcJ0P0()
    diffLIF.calcSusceptibility(w)
    susceptibility2 = inv_f_trans_on_vector(w,diffLIF.susceptibilityW)

    figure()
    plot(diffLIF.v0list,diffLIF.p0list,'.-b')
    ylabel('P0')
    twinx()
    plot(diffLIF.v0list,diffLIF.j0list,'.-r')
    ylabel('J0')
    xlabel('v0 (V)')
    print "Mean firing rate (slightly off due to discretization) =",\
                                1.0/diffLIF.p0integral
    title("P0 (blue), J0 (red)")

    figure()
    plot(w,absolute(diffLIF.susceptibilityW),'.-c')
    ylabel('|gain| (?)')
    twinx()
    plot(w,angle(diffLIF.susceptibilityW),'.-m')
    ylabel('angle (rad)')
    xlabel('frequency (Hz)')
    title("gain ampl(b/c), phase(r/m)")

if __name__ == "__main__":
    Tmax = 200e-3       # Maximum time lag (seconds)
                        #  over which to calculate cross-correlations
    dt = 1e-3           # Bin size for which to calculate cross-correlations (s)
    # Generate a vector of frequencies at which to solve for the spectral
    # statistics in order to generate cross-correlations with maximum lag Tmax
    # and bin size dt.
    dw = 1./2./Tmax
    wmax = 1./2./dt
    w = arange(-wmax,wmax,dw)
    w[abs(w) < 1e-3] = 1e-3
    #w = array([10**omegaexp for omegaexp in arange(-2,3,0.1)])

    diffLIF = DiffLIF()
    diffLIF.calcJ0P0()
    print 'rate by calcJ0P0',diffLIF.bgndrate
    diffLIF.calcJ1P1(w)
    diffLIF.calcRate()
    print 'rate by calcRate',diffLIF.bgndrate
    diffLIF.calcSusceptibility(w)
    susceptibility2 = inv_f_trans_on_vector(w,diffLIF.susceptibilityW)
    
    figure()
    plot(diffLIF.v0list,diffLIF.p0list,'.-b')
    ylabel('P0')
    twinx()
    plot(diffLIF.v0list,diffLIF.j0list,'.-r')
    ylabel('J0')
    xlabel('v0 (V)')
    title("P0 (blue), J0 (red)")

    figure()
    #semilogx(diffLIF.freqlist,absolute(diffLIF.susceptibility),'.-b')
    plot(diffLIF.freqlist,absolute(diffLIF.susceptibilityW),'.-c')
    ylabel('|gain| (?)')
    twinx()
    #semilogx(diffLIF.freqlist,angle(diffLIF.susceptibility),'.-r')
    plot(diffLIF.freqlist,angle(diffLIF.susceptibilityW),'.-m')
    ylabel('angle (rad)')
    xlabel('frequency (Hz)')
    title("gain ampl(b/c), phase(r/m)")
    
    #matlab_josic_trousdale_compare(w)

show()
