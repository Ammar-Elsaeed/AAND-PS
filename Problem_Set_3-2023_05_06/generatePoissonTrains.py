import numpy
import scipy.io
'''
This code was originally in Matlab
function generatePoissonTrains
%% author: Paula Kuokkanen
%% 5. Mai 2011

traslated to Python: Nikolay Chenkov
June 2013

homogeneous Poisson process

A homogeneous Poisson process generates events (spikes) with a constant 
probability or rate r(t) = r. Spikes are independent. Thus, at each 
time point t, the probability P of observing a spike within 
a sufficiently small temporal window from t to t + delta_t is 
P= r*delta_t.

There are two simple ways of implementing a homogeneous Poisson process:
1.    Progress through time in small steps of size delta_t, 
      draw a random number x (from a uniform distribution 
      between 0 and 1, Matlab function 'rand' 
      at each time step. If x < P, generate a spike.

2.    Generate interspike intervals from an exponential probability 
      density. When x is uniformly distributed between 0 and 1, 
      its negative logarithm is exponentially distributed. Thus, we 
      can generate spike times t(i) iteratively from the 
      formula t(i+1) = t(i) - log(x)/r.

Using implementation #2, generate 1000 spikes by a homogeneous
Poisson process with a rate of r=100 (Hz)
'''

n = 1000   # desired number of spikes
rate = 100  # rate in spikes per second
randNumbers = numpy.random.rand(n)  # n random numbers, uniform distribution
ISIs = -numpy.log(randNumbers)/rate   # ISIs in seconds
ISIs = ISIs*1000   # ISIs in milliseconds

# compute spike times from ISIs
SpikeTimes = numpy.cumsum(ISIs)

'''
# inhomogeneous Poisson process

 An inhomogeneous Poisson process is characterized by a time-dependent
 rate r(t). There are several ways of generating spikes. 
 Two basic ways are:
 1.    Progress through time in small intervals of width delta_t. 
       For sufficiently small delta_t, the probability P(i) of 
       observing a spike in interval i from t(i) = i*delta_t to 
       t_(i) + delta_t is then given by 
           P(i) = integral_t(i)^[t_(i)+delta_t)] r(tau) d tau . 
       Then, draw a random number x(i) from a uniform distribution 
       for each time interval. If x(i) < P(i)$, generate a spike.

 2.    `Thinning' the spike train of a homogeneous Poisson process: 
       First, a spike sequence is generated with a homogeneous 
       Poisson process at rate r_max = max[r(t)]. The spike 
       sequence is then thinned by generating a random number 
       x(i) for each spike and removing the spike at time 
       t(i) from the train if r(t(i))/r_max < x(i).

 We use now method 2, thinning. Our probability distribution r is
     r(t) = A* sin(2*pi*f*t) + r_0
 with A = 50 Hz, r_0 = 100 Hz and f = 10 Hz. We generate 1000 spikes.

 First, start with the homogeneous, but with more spikes so that you can
 thin out later on
'''

n = 10000   # desired number of spikes
maxRate = 150  # rate in spikes per second
randNumbers2 = numpy.random.rand(n)  # n random numbers, uniform distribution
ISIs2 = -numpy.log(randNumbers2)/maxRate   # ISIs in seconds
ISIs2 = ISIs2*1000   # ISIs in milliseconds

# compute spike times from ISIs
SpikeTimes2 = numpy.cumsum(ISIs2)

# thin the spike train

randNumbers = numpy.random.rand(n)
rates = 50.*numpy.sin(SpikeTimes2*2*numpy.pi/100) + 100
Ps = rates/maxRate
SpikeTimes_inh = SpikeTimes2[randNumbers>Ps]
n = 1000
SpikeTimes_inh = SpikeTimes_inh[0:n]     # take just 1000 spikes

'''
Poisson process with absolute refractory period

Real neurons are in a refractory state immediately after a spike. 
Thus, during a certain 'refractory period' after a spike, it is 
impossible or much harder to evoke the next spike.
When the effect of refractoriness is taken into account, spikes 
can no longer considered to be independent because the probability 
of observing the next spike then also depends on the time that has 
passed since the last spike.

In point process models, an approximation of refractoriness is to 
set the probability of generating a spike to 0 for a certain absolute 
refractory period tr after a spike, and then to r afterwards. 

We simulate a homogeneous Poisson process and add an absolute refractory
period of t_r = 5 ms. We use different 'driving' rates 
r = 10, 50, 100, 200, 500, and 1000Hz.
We simulate 1000 spikes for each rate r.

Adding an absolute refractory period corresponds to simply adding a 
fixed value to each interspike interval of a 'normal' 
homogeneous Poisson process.
'''

rates_ref = [10, 50, 100, 200, 500, 1000]  # "driving" rates in spikes per second
n = 1000   # desired number of spikes
ISIs_ref = numpy.zeros((n,len(rates_ref)))
randNumbers3 = numpy.random.rand(n)  #n random numbers, uniform distribution
t_r = 5    # absolute refractory period in ms

for i,r in enumerate(rates_ref):
    ISIs_ref[:,i] = -numpy.log(randNumbers3)/r*1000 + t_r  #ISIs in milliseconds

# compute spike times from ISIs
SpikeTimes_ref = numpy.zeros((n,len(rates_ref)))
for i,r in enumerate(rates_ref):
    SpikeTimes_ref[:,i] = numpy.cumsum(ISIs_ref[:,i])

scipy.io.savemat('PoissonSpikeTrains.mat',
                    {'SpikeTimes_hom': SpikeTimes,
                    'SpikeTimes_inh': SpikeTimes_inh,
                    'SpikeTimes_ref': SpikeTimes_ref,
                    'rates_ref': rates_ref})

