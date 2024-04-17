import numpy as np 
import matplotlib.pyplot as plt 
from scipy import integrate
from scipy.special import jv
from numpy import exp, cos, sin, sqrt, pi
from mediapack import Air
plot_color = ["r", "b", "m", "k", "g", "y", "k", "r--", "b--", "m--", "k--", "g--", "y--", "k--"]

class Window():
    def __init__(self, L_x=1., L_y=1., **kwargs):
        self.L_x = L_x 
        self.L_y= L_y
        self.S = L_x*L_y
        self.r = L_x/L_y
        self.rho_0 = Air.rho
        self.c_0 = Air.c
        self.update_frequency(2*pi*1000)

    def update_frequency(self, omega):
        self.omega = omega
        self.k_0 = self.omega / self.c_0

    def sigma_Rhazi(self, k, phi):
        def R_theta(t):
            theta_l = np.arctan(1/self.r)
            if t < theta_l:
                return 2/cos(t)
            else:
                return 2/(self.r*sin(t))
        def f_r(psi):
            alpha_1 = (self.L_x/2)*(self.k_0+k*cos(psi-phi))
            alpha_2 = (self.L_x/2)*(self.k_0-k*cos(psi-phi))
            alpha_3 = (self.L_x/2)*(self.k_0+k*cos(psi+phi))
            alpha_4 = (self.L_x/2)*(self.k_0-k*cos(psi+phi))
            R = R_theta(psi)
            def f(alpha):
                out = 4* (1/alpha-cos(R*alpha)/alpha)
                out += -2*(cos(psi)+self.r*sin(psi))*(-R*cos(R*alpha)/alpha+sin(R*alpha)/alpha**2)
                out += (self.r*cos(psi)*sin(psi))*(-R**2*cos(R*alpha)/alpha+2*R*sin(alpha*R)/alpha**2+2*cos(alpha*R)/alpha**3-2/alpha**3)
                return out
            return (f(alpha_1)+f(alpha_2)+f(alpha_3)+f(alpha_4))/4
        ReZ = (self.rho_0*self.omega*self.L_y/(4*pi))*self.r*integrate.quad(f_r, 0, pi/2)[0] # Eq 17 (Rhazi)
        return ReZ /(self.rho_0*self.c_0) # Eq 2 (Rhazi)

    def sigma_Villot(self, k, psi):
        # Equation (7) from Villot et al.
        def ff(k_r, phi):
            out  = (1-cos((k_r*cos(phi)-k*cos(psi))*self.L_x))/((k_r*cos(phi)-k*cos(psi))*self.L_x)**2
            out *= (1-cos((k_r*sin(phi)-k*sin(psi))*self.L_y))/((k_r*sin(phi)-k*sin(psi))*self.L_y)**2
            out *= self.k_0*k_r/sqrt(self.k_0**2-k_r**2)
            return out
        return (self.rho_0*self.c_0*self.S/(pi**2))*np.array(integrate.dblquad(ff, 0, 2*pi, 0, self.k_0))[0]


    def sigma_average_Rhazi(self, k):
        def func(phi):
            return self.sigma_Rhazi(k, phi)/(2*pi)
        return integrate.quad(func, 0, 2*pi)[0]

    def sigma_average_Villot(self, k):
        def func(phi):
            return self.sigma_Villot(k, phi)/(2*pi)
        return integrate.quad(func, 0, 2*pi)[0]

    def sigma_average_Yu(self, k):
        l_x = max([self.L_x, self.L_y])
        l_y = min([self.L_x, self.L_y])

        sigma = 0
        def f1(R):
            out = l_x*l_y*pi/2-l_y*R-l_x*R+R**2/2
            out *= ((2*1j*self.k_0)/(pi*self.S))*jv(0, k*R)*exp(-1j*self.k_0*R)
            return np.real(out)
        sigma += integrate.quad(f1, 0, l_y)[0]
        def f2(R):
            out = l_x*l_y*np.arcsin(l_y/R)-l_y**2/2+l_x*sqrt(R**2-l_y**2)-l_x*R
            out *= ((2*1j*self.k_0)/(pi*self.S))*jv(0, k*R)*exp(-1j*self.k_0*R)
            return np.real(out)
        sigma += integrate.quad(f2, l_y, l_x)[0]
        def f3(R):
            out = l_x*l_y*(np.arcsin(l_y/R)-np.arccos(l_x/R))+l_y*sqrt(R**2-l_x**2)+l_x*sqrt(R**2-l_y**2)-(l_x**2+l_y**2+R**2)/2
            out *= ((2*1j*self.k_0)/(pi*self.S))*jv(0, k*R)*exp(-1j*self.k_0*R)
            return np.real(out)
        sigma += integrate.quad(f3, l_x, sqrt(l_x**2+l_y**2))[0]
        return sigma

    def sigma_average_Leppington(self, f, fc, CBC=1, COB=1):
        U = 2*(self.L_x+self.L_y)
        
        k = 2*pi*f/self.c_0
        t = (f<fc).sum()
        f_LF = f[:t]
        f_HF = f[t:]
        mu_LF = sqrt(fc/f_LF)
        mu_HF = sqrt(fc/f_HF)
        k_LF = 2*pi*f_LF/self.c_0
        k_HF = 2*pi*f_HF/self.c_0
        
        
        sigma_LF = U/(2*pi*mu_LF*k_LF*self.S*sqrt(mu_LF**2-1))
        sigma_LF *= np.log((mu_LF+1)/(mu_LF-1))+(2*mu_LF)/(mu_LF**2-1)
        sigma_LF *= CBC*COB-((CBC*COB-1)/mu_LF**8)
        sigma_HF = 1/sqrt(1-mu_HF**2)
        
        sigma = np.append(sigma_LF, sigma_HF)
        
        L_1 = min([self.L_x, self.L_y])
        L_2 = max([self.L_x, self.L_y])
        sigma_c = (0.5-0.15*L_1/L_2)* sqrt(k)*sqrt(L_1)*f**0
        
        sigma = np.minimum(sigma, sigma_c)
        return sigma
