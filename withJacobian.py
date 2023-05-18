
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

# this function returns 3 by 3 matrix defining 
# the Jacobian matrix of F at the input vector w    
def JacobianMatrix(w):
    JacobianM=np.zeros((3,3))
    
    JacobianM[0,0]=4*w[0]
    JacobianM[0,1]=2*w[1]
    JacobianM[0,2]=2*w[2]
    
    JacobianM[1,0]=1
    JacobianM[1,1]=1
    JacobianM[1,2]=2
    
    JacobianM[2,0]=w[1]*w[2]
    JacobianM[2,1]=w[0]*w[2]
    JacobianM[2,2]=w[0]*w[1]
    
    return JacobianM

# generate an initial guess
initialGuess=np.random.rand(3)    

# solution     
solutionTuple=fsolve(nonlinearEquation,initialGuess,fprime=JacobianMatrix,full_output=1)

# investigate the solution dictionary

#extract the solution
solution=solutionTuple[0]


# compute the residual
residual=nonlinearEquation(solution)