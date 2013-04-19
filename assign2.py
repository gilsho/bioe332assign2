import brian_no_units
from brian import *
import numpy
from matplotlib.pyplot import *
import argparse

parser = argparse.ArgumentParser()
#parser.add_argument('-k', type = int, default = 0)
parser.add_argument('-s', type = int, default = 0)
parser.add_argument('-c', type = float, default = 0)
pa = parser.parse_args()
#numpy.random.seed(pa.k)

# since we can receive only one argument from qsub, we parse this
# argument to determine the coherence and trial number

# N_TRIALS_PER_LEVEL = 20
# COHERENCE_LEVELS = [3,6.05,12.2,24.6,49.59,100]

# c = COHERENCE_LEVELS[pa.k/N_TRIALS_PER_LEVEL]
# trial = pa.k % N_TRIALS_PER_LEVEL

c = pa.c
trial = pa.s

numpy.random.seed(int(c*100+trial))

# population sizes
Ne = 1600               # number of excitatory (pyramidal) neurons
Ni = 400                # number of inhibitory interneurons

# sub-populations
f = 0.15
Ne1 = int(f*Ne)
Ne2 = int(f*Ne)
Ne0 = Ne-(Ne1+Ne2)

# clock parameters 
dt_sim = 0.02*ms
sim_duration = 3 * second    # in seconds

simulation_clock = Clock(dt = dt_sim)
rate_clock = Clock(dt=50*ms)
decision_clock = Clock(dt = 10*ms)

# stimulus parameters
#coherence = pa.c
coherence = c
mu = 40 * Hz
sigmaMu = 4 * Hz
stim_start = 0*second
stim_stop =  1*second
rate_threshold = 15


# each type of cell is characterized by 6 intrinsic parameters:

# pyramidal cells
Cm_e = 0.5 * nF           # total capacitance Cm
gl_e = 25 * nS            # total leak conductance gL
El_e = -70 * mvolt        # leak reversal potential El
Vt_e = -50 * mvolt        # threshold potential Vt
Vr_e = -55 * mvolt        # reset potential Vr
tr_e = 2 * msecond        # refractory time tau


# interneurons
Cm_i = 0.2 * nF           # total capacitance Cm
gl_i = 20 * nS            # total leak conductance gL
El_i = -70 * mvolt        
Vt_i = -50 * mvolt        # threshold potential Vt
Vr_i = -55 * mvolt        # reset potential Vr
tr_i = 1 * msecond        # refractory time tau


# external input is modeled as uncorrelated Poisson spike trains
# to each neuron at a rate of v_ext = 1800 Hz per cell
fext = 2400 * Hz

# equation constants
a = 0.062 * 1/mvolt
b = 1/3.57

# external input mediated exclusively by AMPA receptors
g_ext_ampa_e = 2.1 * nS          # max conductance on pyramidal (excitatory) cells
g_rec_ampa_e = 0.05 * nS
g_nmda_e = 0.165 * nS
g_gaba_e = 1.3 *nS

g_ext_ampa_i = 1.62 * nS         # max conductance on interneurons (inhibitory)
g_rec_ampa_i = 0.04 * nS
g_nmda_i = 0.13 * nS
g_gaba_i = 1.0 * nS

# connection weight within selective sub-populations
wp = 1.7

# connection weight between selective sub-populations
wm = (1-f*(wp - 1)/(1-f))

# recurrent inhibitory excitation mediated by AMPA receptors.
# in order to connect the recurrent current to the same state variable
# as the external current we scale the conductance ratio by w_ext_i
w_ext_i = g_ext_ampa_i/g_rec_ampa_i
w_ext_e = g_ext_ampa_e/g_rec_ampa_e


# synaptic responses 
tau_syn_ampa = 2 * msecond     # time constants for AMPA
tau_syn_gaba = 5 * msecond    # time constants for GABA
tau_syn_nmda = 100 * msecond   # time constant for NMDA decay time
tau_syn_x = 2 * msecond        # time constant for NMDAR rise time
alfa = 500 * hertz             # controls saturation properties of NMDAR

# synaptic reversal potential
E_ampa = 0 * mvolt          # excitatory syn reversal potential
E_gaba = -70 * mvolt        # inhibitory syn reversal potential
E_nmda = 0 * mvolt          # excitatory syn reversal potential

# specify the interneuron model
eqs_i = '''
dV/dt = (-gl_i*(V - El_i) - g_rec_ampa_i*s_ampa*(V - E_ampa) \
- g_gaba_i*s_gaba*(V - E_gaba) \
- g_nmda_i*s_tot*(V - E_nmda)/(1 + b*exp(-a*V)))/Cm_i : volt
ds_ampa/dt = -s_ampa/tau_syn_ampa : 1
ds_gaba/dt = -s_gaba/tau_syn_gaba : 1
s_tot : 1
'''

# specify the excitatory neuron model
eqs_e = '''
dV/dt = (-gl_e*(V - El_e) - g_rec_ampa_e*s_ampa*(V - E_ampa) \
- g_gaba_e*s_gaba*(V - E_gaba) \
- g_nmda_e*s_tot*(V - E_nmda)/(1 + b*exp(-a*V)))/Cm_e : volt
ds_ampa/dt = -s_ampa/tau_syn_ampa : 1
ds_gaba/dt = -s_gaba/tau_syn_gaba : 1
ds_nmda/dt = -s_nmda/tau_syn_nmda + alfa*x*(1 - s_nmda) : 1
dx/dt = -x/tau_syn_x : 1
s_tot : 1
'''


# make the inhibitory neurons
Pi = NeuronGroup(N=Ni, model=eqs_i, threshold=Vt_i, reset=Vr_i,
    refractory=tr_i, clock=simulation_clock, order=2)

# make the excitatory neurons
Pe = NeuronGroup(N=Ne, model=eqs_e, threshold=Vt_e, reset=Vr_e,
    refractory=tr_e, clock=simulation_clock, order=2)  

# divide excitatory neurons into sub-groups
Pe1 = Pe.subgroup(Ne1)
Pe2 = Pe.subgroup(Ne2)
Pe0 = Pe.subgroup(Ne0)

# add the external Poisson drive to the populations
PGe1 = PoissonGroup(Ne1,fext,clock=simulation_clock)
PGe2 = PoissonGroup(Ne2,fext,clock=simulation_clock)
PGe0 = PoissonGroup(Ne0,fext,clock=simulation_clock)
PGi = PoissonGroup(Ni, fext, clock=simulation_clock) 

Cpe1 = IdentityConnection(PGe1, Pe1, 's_ampa', weight=w_ext_e)
Cpe2 = IdentityConnection(PGe2, Pe2, 's_ampa', weight=w_ext_e)
Cpe0 = IdentityConnection(PGe0, Pe0, 's_ampa', weight=w_ext_e)
Cpi = IdentityConnection(PGi, Pi, 's_ampa', weight=w_ext_i)


# set up recurrent inhibition from inhibitory to excitatory cells
Cie = Connection(Pi, Pe, 's_gaba', weight=1.0, delay=0.5*ms)
Cei = Connection(Pe, Pi, 's_ampa', weight=1.0, delay=0.5*ms)
Cii = Connection(Pi, Pi, 's_gaba', weight=1.0, delay=0.5*ms)

# set up synaptic latency for NMDA gating
selfnmda = IdentityConnection(Pe,Pe,'x',weight=1.0,delay=0.5*ms)

# create excitatory recurrent connections within sub groups
C11 = Connection(Pe1,Pe1, 's_ampa', weight=wp, delay=0.5*ms)
C22 = Connection(Pe2,Pe2, 's_ampa', weight=wp, delay=0.5*ms)
C00 = Connection(Pe0,Pe0, 's_ampa', weight=wm, delay=0.5*ms)

# create excitatory recurrent connections between sub groups
C12 = Connection(Pe1,Pe2, 's_ampa', weight=wm, delay=0.5*ms)
C21 = Connection(Pe2,Pe1, 's_ampa', weight=wm, delay=0.5*ms)

C01 = Connection(Pe0,Pe1, 's_ampa', weight=wm, delay=0.5*ms)
C10 = Connection(Pe1,Pe0, 's_ampa', weight=wm, delay=0.5*ms)

C02 = Connection(Pe0,Pe2, 's_ampa', weight=wm, delay=0.5*ms)
C20 = Connection(Pe2,Pe0, 's_ampa', weight=wm, delay=0.5*ms)

# implement the update of NMDA conductance s_tot on each time step
@network_operation(simulation_clock,when='start')
def update_nmda(simulation_clock):
    s_NMDA1 = Pe1.s_nmda.sum()
    s_NMDA2 = Pe2.s_nmda.sum()
    s_NMDA0 = Pe0.s_nmda.sum()
    Pe1.s_tot = (wp*s_NMDA1 + wm*s_NMDA2 + wm*s_NMDA0)
    Pe2.s_tot = (wm*s_NMDA1 + wp*s_NMDA2 + wm*s_NMDA0)
    Pe0.s_tot = (s_NMDA1 + s_NMDA2 + s_NMDA0)
    Pi.s_tot = (s_NMDA1 + s_NMDA2 + s_NMDA0)

@network_operation(rate_clock, when='start')
def update_rates(rate_clock):
    if (rate_clock.t >= stim_start and rate_clock.t<stim_stop):
        PGe1.rate = fext + mu*(1.0 + 0.01*coherence) + randn()*sigmaMu
        PGe2.rate = fext + mu*(1.0 - 0.01*coherence) + randn()*sigmaMu
    else:
        PGe1.rate = fext
        PGe2.rate = fext

def record_decision(correct,dtime):
    ftrial = open('coh' + str(c) + '.dat','a')
    ftrial.write(str(correct) + ';')
    ftrial.write(str(dtime) + '\n')
    ftrial.close()

def get_rate(r):
    i = min(0,len(r)-10)
    j = len(r)
    return numpy.average(numpy.array(r[i:j]))

@network_operation(decision_clock,when='end')
def check_decision(decision_clock):
    r1 = get_rate(rate_Pe1.rate)
    r2 = get_rate(rate_Pe2.rate)
    if r1 > rate_threshold:
        record_decision(1,decision_clock.t)
        stop()
    elif r2 > rate_threshold:
        record_decision(0,decision_clock.t)
        stop()


def save_rate(rs,n):
    f = open('rate' + str(pa.s) + '_' + str(n) + '.dat','w+')
    for r in rs:
        f.write(str(r) + '\n')
    f.close()


# randomize initial voltage
Pi.V = Vr_i + rand(Ni) * (Vt_i - Vr_i)
Pe.V = Vr_e + rand(Ne) * (Vt_e - Vr_e)
    
rate_Pe1 = PopulationRateMonitor(Pe1, 5*ms)
rate_Pe2 = PopulationRateMonitor(Pe2, 5*ms)
rate_Pi = PopulationRateMonitor(Pi, 5*ms)


M1 = SpikeMonitor(Pe1)
M2 = SpikeMonitor(Pe2)
Mi = SpikeMonitor(Pi)
Me = SpikeMonitor(Pe)

#execute the simulation
run(sim_duration)

# save_rate(rate_Pe1.smooth_rate(width=50*ms,filter='flat'),1)
# save_rate(rate_Pe1.smooth_rate(width=50*ms,filter='flat'),2)


 