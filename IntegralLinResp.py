from pylab import *
from IntegralPops import IntegralPop

class IntegralLinResp(IntegralPop):
    """
        Using SI units implicitly throughout. 
        Convert from Brian when passing.
    """
    def __init__(self,runtime,bindt,\
                    tlowinf,tupinf,
                    *args,**kwargs):
        IntegralPop.__init__(self,*args,**kwargs)
        # inf approx for time
        self.tupinf = tupinf    # seconds
        self.tlowinf = tlowinf  # seconds
        self.bindt = bindt
        self.runtime = runtime
        self.Nbins = int(runtime/bindt)
        self.deltaA = zeros(self.Nbins)
        
        self.noise2 = self.noise**2

        # linear filter evaluation points
        #  function below interpolate between these points
        self.xlow = 1e-100
        self.xhigh = 0.1
        self.xdt = 0.0005
        self.trange = arange(self.xlow,self.xhigh,self.xdt)
        self.numx = len(self.trange)

    def L_integrand(self,tprime,x,h0):
        return self.hazard(tprime,h0)/sqrt(self.noise2) \
                    *self.survivor(tprime+x,h0)

    def L_SRM(self,x,h0):
        '''
        L_SRM is independent of current time
        See this by substituting t' = s-t_hat in eqn (14.58)
        as also seen from eqn (14.54) in Wulfram's book2
        '''
        #return (x>=0) * quad(L_integrand,0,tupinf,args=(x,h0,A0))[0]
        return (x>=0) * trapz(self.L_integrand(self.fullt,x,h0),dx=self.fullt_dt)
            # Heaviside(x) * integral

    def compute_linfilter(self):
        self.L_SRMarray = array([self.L_SRM(tpt,self.h0) \
                                for tpt in self.trange])

    def L_SRMinterp(self,x):
        '''
        returns linearly interpolated values of linear filter for SRM
        from the already calculated array L_SRMarray for given h0.
        '''
        idx = int((x-self.xlow)/self.xdt)
        if x<0: return 0.0
        elif idx>=(self.numx-1): return 0.0 # assume response decays to zero
        else:
            (Llow,Lhigh) = self.L_SRMarray[idx:idx+2]
            return (Llow + (x/self.xdt-idx)*(Lhigh-Llow))
        
    def ISIinterp(self,x):
        '''
        returns linearly interpolated values of P0
        from the already calculated array ISIdistrib for given h0.
        '''
        idx = int(x/self.fullt_dt)
        endidx = len(self.fullt)
        if x<0: return 0.0
        elif idx>=(endidx-1): return 0.0 # assume ISIdistrib decays to zero
        else:
            (P0low,P0high) = self.ISIdistrib[idx:idx+2]
            return (P0low + (x/self.fullt_dt-idx)*(P0high-P0low))

    def P0_deltaA(self,t_hat,bini,h0):
        t = bini*bindt
        return self.survivor([t-t_hat],h0)[0]*deltaA[bini]

    ## deltah_interp() and compute_deltaA() below are not needed by L_SRM()
    ##  but they are needed to compute the linear response
    ##  see lin_response_recurrent_comparisons_v2.py
    def deltah_interp(self,t,hdt):
        '''
        returns linearly interpolated values of h0
        from the user-given deltaharray.
        '''
        tidx = int(t/hdt)
        extrafraction = t/hdt-tidx        
        if tidx<0: deltah = 0.0
        else:
            (deltahlow,deltahhigh) = self.deltaharray[tidx:tidx+2]
            deltah = (deltahlow + extrafraction*(deltahhigh-deltahlow))
        return deltah

    def compute_deltaA(self,deltaharray,hdt,deltat=0.01):
        self.deltaharray = deltaharray
        self.tarray = arange(0.0,self.runtime,deltat)
        deltaAarray = zeros(len(self.tarray))
        extrapadneeded = len(self.tarray)-len(self.ISIdistrib)
        ISIarray = array([self.ISIinterp(ti) for ti in self.tarray])
        LSRMarray = array([self.L_SRMinterp(ti) for ti in self.tarray])
        deltaharray_interp = array([self.deltah_interp(ti,hdt) for ti in self.tarray])
        h0prime = diff(deltaharray_interp)  # don't divide by deltat here,
                                            #  as it is compensated
                                            #  by not multiplying with deltat below
        for i,ti in enumerate(self.tarray[1:],start=1):
            deltaAarray[i] = sum(deltaAarray[:i]*ISIarray[:i][::-1])*deltat + \
                              self.A0*sum(LSRMarray[:i]*h0prime[:i][::-1])

        return deltaAarray
