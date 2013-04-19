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
							 #'I0'		: 0.3297, 'I1'	: 0, 'I2'	: 0
							 'I0'		: 0.3297, 'I1'	: 0.035, 'I2'	: 0.0351 
							 #'I0'		: 0.3297, 'I1'	: 0.03, 'I2'	: 0.04
							 #'I0'		: 0.3297, 'I1'	: 0, 'I2'	: 0.07
							 }

# auxiliary functions: fI-curce and recurrent current
DSargs.fnspecs = {'fRate'		: (['I'], '(a*I-b)/(1.0-exp(-d*(a*I-b)))'),
									'recCurr'	: (['x','y'], 'J11*x-J12*y+I0') }

# rhs of the differential equations
DSargs.varspecs = {
										's1': '-s1/tauS+(1-s1)*gam*fRate(recCurr(s1,s2)+I1)',
										's2':	'-s2/tauS+(1-s2)*gam*fRate(recCurr(s2,s1)+I2)'}

# initial conditions
DSargs.ics = {'s1': 0.06,
							's2': 0.06}

# set the range of integration
DSargs.tdomain = [0,30]

# variable domain for the phase plane analysis
DSargs.xdomain = {'s1': [0,1], 's2': [0,1]}

# create model object
dmModel = dst.Vode_ODEsystem(DSargs)

# plot vector field
pp.plot_PP_vf(dmModel, 's1', 's2', scale_exp=-1.5)

# find dixed points of the model
fp_coord = pp.find_fixedpoints(dmModel, n=4, eps=1e-8)

# plot the null-clines
nulls_x, nulls_y = pp.find_nullclines(dmModel, 's1', 's2', n=3, \
										eps=1e-8, max_step=0.01, fps=fp_coord)

plot(nulls_x[:,0], nulls_x[:,1], 'b')
plot(nulls_y[:,0], nulls_y[:,1], 'g')

# compute the jacobian matrix
jac, new_fnspecs = dst.prepJacobian(dmModel.funcspec._initargs['varspecs'],
			['s1','s2'],dmModel.funcspec._initargs['fnspecs'])
scope = dst.copy(dmModel.pars)
scope.update(new_fnspecs)
jac_fn = dst.expr2fun(jac, ensure_args=['t'],**scope)

# add fixed points to the phase portrait
for i in range(0,len(fp_coord)):
	fp = pp.fixedpoint_2D(dmModel, dst.Point(fp_coord[i]),
						jac = jac_fn, eps=1e-8)
	pp.plot_PP_fps(fp)

# compute and plot projectories
traj = dmModel.compute('trajectory1')
pts = traj.sample()
plot(pts['s1'], pts['s2'], 'r-o')

xlabel('s_1')
ylabel('s_2')
title('Phase plane I1=0.035 nA, I2=0.035 nA')
# savefig('pp2')
# show()



######################################################################

figure()
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

