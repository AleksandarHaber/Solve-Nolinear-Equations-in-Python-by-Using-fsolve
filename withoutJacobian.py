import numpy as np
from scipy.optimize import fsolve


# for a given variable w, this function returns F(w)
# if w is the solution of the nonlinear system, then 
# F(w)=0
# F can be interpreted as the residual
def nonlinearEquation(w):
    F=np.zeros(3)
    F[0]=2*w[0]**2+w[1]**2+w[2]**2-15
    F[1]=w[0]+w[1]+2*w[2]-9
    F[2]=w[0]*w[1]*w[2]-6
    return F


# generate an initial guess
initialGuess=np.random.rand(3)    

# solve the problem    
solutionInfo=fsolve(nonlinearEquation,initialGuess,full_output=1)

# investigate the solution dictionary
solutionInfo

#extract the solution
solution=solutionInfo[0]


# compute the residual
residual=nonlinearEquation(solution)