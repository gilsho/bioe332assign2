#import brian_no_units
from brian import *
import numpy
from matplotlib.pyplot import *



def smooth_rates(r,bins):
    smooth_r = zeros(len(r))
    for i in range(len(r)):
        sum = 0
        count = 0
        for j in range(min(0,i-bins),i):
            sum += r[j]
            count += 1
        smooth_r[i] = sum/count
    return smooth_r


def make_raster_rate_plot(r1, r2, m1, m2, t,name=''):
    t_vec = numpy.arange(0,t,dt_sim)
    fig = figure()
    # raster plot for selective population 1
    subplot(311)
    raster_plot(m1)
    axvline(x=0,linewidth=8, color='r')
    axvline(x=stim_start/dt_sim, linestyle='--',color='k')
    axvline(x=stim_stop/dt_sim, linestyle='--',color='k')

    # raster plot for selective population 2
    subplot(312)
    raster_plot(m2)
    axvline(x=0,linewidth=8, color='b')
    axvline(x=stim_start/dt_sim, linestyle='--',color='k')
    axvline(x=stim_stop/dt_sim, linestyle='--',color='k')

    # raster plot for selective population 3
    subplot(313)
    r1_smooth = smooth_rates(r1,10)
    r2_smooth = smooth_rates(r2,10)
    plot(t,r1_smooth,color='r',linewidth=2)
    plot(t,r2_smooth,color='b',linewidth=2)
    axvline(x=stim_start/dt_sim, linestyle='--',color='k')
    axvline(x=stim_stop/dt_sim, linestyle='--',color='k')
    xlabel('Time, s')
    ylabel('Firing rate, Hz')
    axis([0,3,0,30])
    
    if name is not '':
        savefig(name)

def make_network_dynamics_plot(r1t1, r2t1,r1t2,r2t2,name=''):
    fig = figure()
    plot(range(0,40),range(0,40),color='k',linestyle='--')
    plot(smooth_rates(r1t1,10),smooth_rates(r2t1,10),color='y',linewidth=4)
    plot(smooth_rates(r1t2,10),smooth_rates(r2t1,10),color='g',linewidth=4)    
    axis([0,40,0,40])
    ylabel('Firing rate rb, Hz')
    xlabel('Firing rate ra, Hz')
    if name is not '':
        savefig(name)

def make_coherence_plot(coh,cor,name=''):
    # plot coherence vs. correctness
    x_vec = numpy.arange(0,100)
    alpha = 9.2
    beta = 1.5
    nmfunc = 100*(1-0.5*numpy.exp(-(x_vec/alpha)**beta))
    fig = figure()
    semilogx(coh,corr,'ko')
    semilogx(x_vec,nmfunc,'r')
    axis([0,100,50,100])
    ylabel('Coherence, %')
    xlabel('%% correct')
    if name is not '':
        savefig(name)

def make_decision_time_plot(coh,mean,stdev,name=''):
    #times must be a numpy array
    plot(coh,mean,'k')
    errorbar(coh,mean,yerr=stdev,fmt='ko')
    if name is not '':
        savefig(name)


def load_rates(filename):
    f = open(filename,'r')
    r = []
    for line in f:
        r.append(float(line))
    return r

def load_coherence_stats(c,ntrials):
    ncorrect = 0
    dtimes = []
    for i in range(1,ntrials+1):
        f = open(str(c) + '/' + 'trial' + str(i) + 'dat')
        ncorrect += int(f.readline())
        dtimes.append(float(f.readline()))
    mean = numpy.mean(dtimes)
    stdev = numpy.std(dtimes)
    perc = 1.0*ncorrect/ntrials
    return (perc, mean, stdev)

def load_stats(coh,ntrials):
    perc = []
    stdev = []
    mean = []
    for c in coh:
        (p,m.s) = load_coherence_stats(c,ntrials)
        perc.append(p)
        mean.append(m)
        stdev.append(s)



r1t1 = rand(600)*20
r2t1 = rand(600)*20
r1t2 = rand(600)*20
r2t2 = rand(600)*20

#r1t1 = load_rates('rate1_1.dat')
#r2t1 = load_rates('rate1_2.dat')
#r1t2 = load_rates('rate0_1.dat')
#r2t2 = load_rates('rate0_2.dat')
make_network_dynamics_plot(r1t1,r2t1,r1t2,r2t2,'test')


coh = [3,6.05,12.2,24.6,49.59,100]
ntrials = 100
#load_stats(coh,ntrials)






