
import numpy as np
from numpy.linalg import norm
from math import pi, log10
import matplotlib.pyplot as plt
import time
from B2R2 import reconstruction_method


def generate_BL_signal(num_of_coeff, Length, rho,n):
    coeff = (np.random.rand(num_of_coeff)-0.5)*2 #Uniform coefficients in [-1,1]
    x_n = np.zeros(Length)
    for k in range(num_of_coeff):
        offset = (k-int(num_of_coeff/2))*15
        x_n += coeff[k] * np.sinc(rho*(n-offset))
    max_norm_bl = max(abs(x_n))
    x_n /= max_norm_bl
    X_ft = np.fft.fft(np.fft.ifftshift(x_n))    #Fourier transform
    E_x_n = (1 / of) * np.linalg.norm(x_n) ** 2
    Liphscitz_c = max(abs(np.diff(x_n)))
    return x_n, X_ft, E_x_n, Liphscitz_c



def M_lambda(delta, Lambda, L):
    thr = 0.1 * Lambda
    m_min = np.where(abs(delta) > thr)[0][0]
    m_max = np.where(abs(delta) > thr)[0][-1]
    M = int(2 * max(m_max - 0.5 * L, 0.5 * L - m_min) + 1)
    return M

def plot_signal(x_1, x_2, name_1, name_2, Lambda, Length, scale):
    plt.plot(n[int(L / 2 - scale):int(L / 2 + scale)], x_1[int(L / 2 - scale):int(L / 2 + scale)], '-r')
    plt.plot(n[int(L / 2 - scale):int(L / 2 + scale)], x_2[int(L / 2 - scale):int(L / 2 + scale)], color='b',
             linestyle='dashed', linewidth=1)
    plt.xlim(-scale, scale)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.75)
    plt.axhline(y=Lambda, color='navy', linestyle='-', linewidth=0.25)
    plt.axhline(y=-Lambda, color='navy', linestyle='-', linewidth=0.25)
    plt.text(x=-scale - 10, y=Lambda, s=chr(955), fontsize=16, va='center_baseline', ha='left', backgroundcolor='w')
    plt.text(x=-scale - 12, y=-Lambda, s='-' + chr(955), fontsize=16, va='center_baseline', ha='left',
             backgroundcolor='w')
    plt.legend([name_1, name_2], loc="upper right", prop={'size': 11})
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.close()

def plot_FT(X_FT, w, rho, beta):
    fig = plt.figure()
    plt.plot(w, X_FT.real, color="k")
    plt.axhspan(xmin=0, xmax=beta * rho / 2, facecolor='r', alpha=0.125)
    plt.axhspan(xmin=beta * rho / 2, xmax=(2 - beta * rho) / 2, facecolor='g',
                alpha=0.125)
    plt.axhspan(xmin=(2 - beta * rho) / 2, xmax=1, facecolor='r', alpha=0.125)
    plt.xlabel("theta (frequency)")
    plt.title("Signal in frequency domain")
    plt.show()
    plt.close()



####################################
############ Parameters ############
####################################
signal_ft_name = "Sos"          # Sum of Sincs
L = 2 ** 10                     # number of samples
w = np.linspace(0, 2 * pi, L)   # frequency
n = np.arange(-L / 2, L / 2)    # time
beta = 1.05
of = 5                          # oversampling factor
rho = 1 / of
Lambda = 0.25                   # Modulo parameter
coeff_num = 10                  # number of sincs
scale = 100

####################################
######## Generating the Model ######
####################################
x_n, X_ft, Energy, Liphscitz_c = generate_BL_signal(coeff_num, L, rho, n)
# Modulo samples
x_lambda = ((np.real(x_n) + Lambda) % (2 * Lambda)) - Lambda
X_ft_lambda = np.fft.fft(np.fft.ifftshift(x_lambda))    # FFT modulo
delta1 = x_n - x_lambda      # True residual signal
M = M_lambda(delta1, Lambda, L) # Number of folded samples. Assume it is known.
print("Number of folded samples = " + str(M))
plot_signal(x_lambda, x_n, "Modulo", "Original", Lambda, L, scale)
time.sleep(2)


####################################
########### Reconstruction #########
####################################
toc = time.time()
r_m_pgd = reconstruction_method(x_lambda.copy(), X_ft_lambda, M, of, w)
x_rec = r_m_pgd.BBRR(Lambda, L)
tic = time.time()
print("Inference Time = {}".format(tic - toc))

# Error
error = x_n - x_rec
mse = (np.linalg.norm(error)) ** 2 / Energy
print("MSE = {} , MSE(dB) = {}".format(mse, 10 * log10(mse)))

plot_signal(x_rec, x_n, "Recovery", "Original", Lambda, L, scale)

