import PyDSTool as dst
from PyDSTool.Toolbox import phaseplane as pp

#initialize model parameters structure
DSargs = dst.args(name='Decision_making_model')

#model parameters
DSargs.pars = {'tauS'	: 0.06,
							 'gam'	: 0.641,
							 'a'		:	270.0,
							 'b'		: 108.0,
							 'd'		: 0.154,
							 'J11'	: 0.3725,
							 'J12'	: 0.1137,
							 'I0'		: 0.3297, 
							 'Icomm'		: 0
							 }

# auxiliary functions: fI-curce and recurrent current
DSargs.fnspecs = {'fRate'		: (['I'], '(a*I-b)/(1.0-exp(-d*(a*I-b)))'),
									'recCurr'	: (['x','y'], 'J11*x-J12*y+I0') }

# rhs of the differential equations
DSargs.varspecs = {
										's1': '-s1/tauS+(1-s1)*gam*fRate(recCurr(s1,s2)+Icomm)',
										's2':	'-s2/tauS+(1-s2)*gam*fRate(recCurr(s2,s1)+Icomm)'}

# initial conditions
DSargs.ics = {'s1': 0.06,
							's2': 0.06}

# set the range of integration
DSargs.tdomain = [0,30]

# variable domain for the phase plane analysis
DSargs.xdomain = {'s1': [0,1], 's2': [0,1]}

# create model object
dmModel = dst.Vode_ODEsystem(DSargs)

# set lower found of the control parameter
dmModel.set(pars = {'Icomm': -0.05} )

# initial conditions
dmModel.set(ics = {'s1': 0.01, 's2': 0.01})

# set up continuation class
PC = dst.ContClass(dmModel)

# equilibrium point curve (EP-C). The branch is labeled EQ1:
PCargs = dst.args(name='EQ1', type='EP-C')
PCargs.freepars = ['Icomm'] #control parameter
PCargs.MaxNumPoints = 1000
PCargs.MaxStepSize = 1e-4
PCargs.MinStepSize = 1e-5
PCargs.StepSize = 1e-3
PCargs.LocBifPoints = 'all' #detect all bifurcation types
PCargs.SaveEigen = True #to determine stability of branches

PC.newCurve(PCargs)
PC['EQ1'].forward()
PC['EQ1'].display(['Icomm','s1'], stability=True, figure=1)

PC.model.icdict = {'s1': 0.01, 's2':0.8}
PC.model.setPars = {'Icomm': -0.05}
PCargs.MaxNumPoints = 3000
PCargs.LocBifPoints = ['LP']
PCargs.name = 'EQ2'
PC.newCurve(PCargs)
PC['EQ2'].forward()
PC['EQ2'].display(['Icomm','s1'], stability=True, figure=1)

savefig('bifurcation')

