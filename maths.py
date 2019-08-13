import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

def integrate_forward(r,u):             #set position and initial conditions

    f = np.zeros_like(u)                # matrix for initial conditions stated in the code

    f1,f2,f3,f4,f5 = u

  # this is where your equations go
    f[0] =                                          #set of coupled differential equations
    f[1] =
    f[2] =
    f[3] =
    f[4] =


    return f

def Deriv(a,b,h):
    Dv = (a - b)/h
    return Dv


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def matrix(r,l):

    M =  np.array([])           #Setup an matrix with two different variables

    return M


def eigvecCoeffs(R,d = 10):
#
#This functions finds the eigenvector coefficients and takes the ratios, then, the ratios and coefficients are
# fed into a DataFrame
#
    global Rmax, a, ix
    steps = 100000

    rr = np.linspace(a,Rmax,steps)
    for i in range(len(rr)):
        if np.round(rr[i]/R,4) == 1:
            ix = i                                          #Find the index where rr = R

    indx = []
    i=0
    for i in range(len(matrix(r,l))):
        indx.append(f"Coefficient {i}")                     #Create an array with the name of indeces
        #indx
    eigval, eigvecs = np.linalg.eig(1j*matrixRel(R,d))        # Obtain the eigenvectors and Eigenvalues
        for i in range(len(eigvecs)):
            indx.append(f"Coefficient {i}")                     #Create an array with the name of indeces
    DiffSolns = np.array([sol.y[i][ix]for i in range(len(eigves))])   #Solutions of the coupled differential eqns
    C = np.matmul(np.conj(eigvecs.T),DiffSolns)                        # \lambda^(\dagger)* solns
    Coeffs  = pd.DataFrame(index = indx)                            #A dataframe to compare solution
    for i in range(len(C)):                                   #Loop to find the difference between coefficients
        D = []
        for j in range(len(C)):
            #if i == j:
            D.append(np.round(np.abs(C[i]/C[j]),4))         #Feed the loop with the ratios of the coeffs
                #continue
        Coeffs[f"Coefficient {i}"] = D
    Coeffs["Coefficients"] = C
    return Coeffs
