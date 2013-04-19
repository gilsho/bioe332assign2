#import brian_no_units
from brian import *
import numpy
from matplotlib.pyplot import *



# def smooth_rates(r,bins):
#     smooth_r = zeros(len(r))
#     for i in range(len(r)):
#         sum = 0
#         count = 0
#         for j in range(min(0,i-bins),i):
#             sum += r[j]
#             count += 1
#         smooth_r[i] = sum/count
#     return smooth_r


def make_raster_rate_plot(r1, r2, m1, m2, t,
                          stim_start,stim_stop,name=''):
    dt = float(t)/len(r1)
    t_vec = numpy.arange(0,t,dt)
    fig = figure()
    # raster plot for selective population 1
    subplot(311)
    raster_plot(m1)
    axvline(x=0,linewidth=8, color='r')
    axvline(x=stim_start, linestyle='--',color='k')
    axvline(x=stim_stop, linestyle='--',color='k')

    # raster plot for selective population 2
    subplot(312)
    raster_plot(m2)
    axvline(x=0,linewidth=8, color='b')
    axvline(x=stim_start, linestyle='--',color='k')
    axvline(x=stim_stop, linestyle='--',color='k')

    # raster plot for selective population 3
    subplot(313)
    plot(t_vec,r1,color='r',linewidth=2)
    plot(t_vec,r2,color='b',linewidth=2)
    axvline(x=stim_start, linestyle='--',color='k')
    axvline(x=stim_stop, linestyle='--',color='k')
    xlabel('Time, s')
    ylabel('Firing rate, Hz')
    axis([0,3,0,40])
    
    if name is not '':
        savefig(name)

def make_network_dynamics_plot(r1t1, r2t1,r1t2,r2t2,name=''):
    fig = figure()
    plot(range(0,40),range(0,40),color='k',linestyle='--')
    plot(r1t1,r2t1,color='y',linewidth=4)
    plot(r1t2,r2t2,color='g',linewidth=4)    
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
    figure()
    semilogx(coh,cor,'ko')
    semilogx(x_vec,nmfunc,'r')
    axis([0,100,50,100])
    ylabel('Coherence, %')
    xlabel('Correct, %')
    if name is not '':
        savefig(name)

def make_decision_time_plot(coh,mean,stdev,name=''):
    #times must be a numpy array
    figure()
    semilogx(coh,mean,'k')
    errorbar(coh,mean,yerr=stdev,fmt='k')
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
    f = open('coh' + str(c) +'.dat')
    for i in range(1,ntrials+1):
        sline = f.readline().split(';') 
        ncorrect += int(sline[0])
        dtimes.append(float(sline[1]))
    mean = numpy.mean(dtimes)
    stdev = numpy.std(dtimes)
    perc = 1.0*ncorrect/ntrials
    return (perc, mean, stdev)

def load_stats(coh,ntrials):
    perc = []
    stdev = []
    mean = []
    for c in coh:
        (p,m,s) = load_coherence_stats(c,ntrials)
        perc.append(p*100)
        mean.append(m)
        stdev.append(s)
    make_coherence_plot(coh,perc,'coherence_correct')
    make_decision_time_plot(coh,mean,stdev,'coherence_dtime')




coh = [3.0,6.05,12.2,24.6,49.59,100.0]
ntrials = 20
#load_stats(coh,ntrials)






