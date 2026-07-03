import numpy as np
import multiprocessing as mp
num_cores = mp.cpu_count()-1   
import time
from scipy.integrate import quad 
import pandas as pd
from scipy.interpolate import interp1d
from scipy.constants import h, c
from numba import jit
import copy
from scipy.sparse.linalg import eigs
from multiprocess import Pool
from functools import lru_cache
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.integrate import simpson
from multiprocess import Pool
from scipy.integrate import quad_vec


import time




h = 6.62607015e-34
c = 299792458
q = 1.60217663e-19
k_B = 1.380649e-23


solardata_path = '/PATH/TO/DATA/'
solardata_path = '/Users/jon/Documents/Research/Sheldon_Research/Matlab/astmg173v2.xlsx'
solardata = pd.read_excel(solardata_path, engine='openpyxl')

wavelength = solardata.iloc[:, 0].to_numpy()
power_GT = solardata.iloc[:, 2].to_numpy()
power_DC = solardata.iloc[:, 3].to_numpy()

wl_range = np.arange(280, 4000.001, 0.001)

interp_power_GT = interp1d(wavelength, power_GT, bounds_error=False, fill_value=0)
power_GT_interp = interp_power_GT(wl_range)

interp_power_DC = interp1d(wavelength, power_DC, bounds_error=False, fill_value=0)
power_DC_interp = interp_power_DC(wl_range)

photons_GT = (power_GT * 1e-9 * wavelength) / (h * c)
interp_photons_GT = interp1d(wavelength, photons_GT, bounds_error=False, fill_value=0)
photons_GT_interp = interp_photons_GT(wl_range)

photons_DC = (power_DC * 1e-9 * wavelength) / (h * c)
interp_photons_DC = interp1d(wavelength, photons_DC, bounds_error=False, fill_value=0)
photons_DC_interp = interp_photons_DC(wl_range)

abs_lst = [1e4,1e5]
abs_lst = np.logspace(0,7,2)
P_abs = []
def vec_Markov_parallel(abs_v,n_ref_L,dev_angle,QY_v,sun_v):
    eta_tilt = []
    return_vals = []
    #sunlight=0 ## Global
    sunlight = 1 ## Direct
    dev = dev_angle
    rho_L = abs_v
    #rho_P = 1e6
    rho_P = 1216200
    dz = 1e-6
    n_L = 97
    n_P = 3
    n_tot = n_L+n_P+1
    n_m = n_L+n_P
    M = n_m
    T_c = 300
    E_g_array = q*np.linspace(1.42,1.42,1)
    P_alt_array = np.array([0] + list(np.logspace(-6,0,1000)))
    V_app_fit = np.linspace(0,2.5,5001)
    QY_L = QY_v
    QY_P = 1

    n_ref_E = 1
    n_ref_L = n_ref_L
    n_ref_P = 4
    n_ref_P = 3.666

    t_c_P2L = np.arcsin(n_ref_L/n_ref_P) ### NR medium refractive index must be less than or equal to PV medium
    

    t_E2L = lambda t: np.arcsin((n_ref_E/n_ref_L)*np.sin(t))
    t_E2P = lambda t: np.arcsin((n_ref_E/n_ref_P)*np.sin(t))
    t_L2P = lambda t: np.arcsin((n_ref_L/n_ref_P)*np.sin(t))
    t_P2L = lambda t: np.arcsin((n_ref_P/n_ref_L)*np.sin(t))

    P_ref_1 = np.zeros((n_tot,n_tot))
    P_dir = np.zeros((n_tot,n_tot))
    P_ref_TIR_1 = np.zeros((n_tot,n_tot))
    P_ref_TIR_2 = np.zeros((n_tot,n_tot))
    
    t_Sun = np.deg2rad(0.267)
    t_Sun = sun_v
    #t_AR = np.deg2rad(0.267) ### Angular range of emission/absorption   Emission matches that of direct sunlight
    t_AR = np.deg2rad(90) ### Angular range of emission/absorption
    rho_L = abs_v

    @jit(cache=True)
    def fx(theta,alpha):
        if abs_v==0:
            return 1
        return 1 - ((np.cos(alpha)**2) * (np.cos(theta)**2)) - (0.5 * (np.sin(alpha)**2) * (np.sin(theta)**2)) ### Tilted

    @jit()
    def L(theta,alpha):
        return np.cos(theta)*np.cos(alpha)

    @jit()
    def Lt(theta,alpha):
        return np.arccos(np.cos(theta)*np.cos(alpha))

    def un(*args):
        return 1

    def a_L(theta,alpha): 
        return rho_L*fx(theta,alpha)
        
    def a_P(theta):
        return rho_P

    ############################################## Emission Functions ###############################################################################
    
    def dp_e_L(theta,alpha): 
        nu = 1/(2*np.pi*quad(lambda t: np.sin(t)*fx(t,alpha),0,np.pi)[0])
        return nu*np.sin(theta)*fx(theta,alpha)

    @jit()
    def dp_e_P(theta,alpha):
        return (1/(2*np.pi))*(1/2)*np.sin(theta)
    ###############################################################################################################################



    ############################################## Absorption Functions ###############################################################################

    def p_abs_E2L(theta,alpha): 
        return 1-np.exp(-a_L(t_E2L(theta),alpha)*dz/abs(np.cos(t_E2L(theta))))

    def p_abs_E2P(theta,alpha): 
        return 1-np.exp(-a_P(theta)*dz/abs(np.cos(t_E2P(theta))))

    def p_abs_X2E(*args):
        return 1

    def p_abs_L2L(theta,alpha):
        return 1-np.exp(-(dz*a_L(theta,alpha))/abs(L(theta,alpha)))

    def p_abs_P2P(theta,alpha):
        return 1-np.exp(-(a_P(theta)*dz)/abs(np.cos(theta)))

    def p_abs_L2P(theta,alpha):
        return 1-np.exp(-(a_P(theta)*dz)/abs(np.cos(t_L2P(Lt(theta,alpha)))))

    def p_abs_P2L(theta,alpha):
        return 1-np.exp(-(a_L(t_P2L(theta),alpha)*dz)/abs(np.cos(t_P2L(theta))))

    def p_abs_L2L_self(theta,alpha):
        return 2*(1-np.exp(-(0.5*a_L(theta,alpha)*dz)/abs(L(theta,alpha))))

    def p_abs_P2P_self(theta,alpha):
        return 2*(1-np.exp(-(0.5*a_P(theta)*dz)/abs(np.cos(theta))))


    ###############################################################################################################################

    def p_e_L(t,a):
        return dp_e_L(t,a)

    def p_e_P(t,a):
        return (1/(2*np.pi))*(1/2)*np.sin(t)

    def p_a_L2L(t,a):
        return 1-np.exp(-(dz*a_L(t,a))/L(t,a))
        
    def p_a_L2P(t,a):
        return 1-np.exp(-(a_P(t)*dz)/np.cos(t_L2P(Lt(t,a))))
        
    def p_a_P2L(t,a): 
        return 1-np.exp(-(dz*a_L(t_P2L(t),a))/np.cos(t_P2L(t)))
    
    def p_a_P2P(t,a): 
        return 1-np.exp(-(a_P(t)*dz)/np.cos(t))

    def f_TIR_1_L(t,a):
        return (n_L*dz*a_L(t,a))/L(t,a)+(a_P(t)*n_P*dz)/np.cos(t_L2P(Lt(t,a)))
        
    def f_TIR_1_P(t,a): 
        return (n_L*dz*a_L(t_P2L(t),a))/np.cos(t_P2L(t))+(a_P(t)*n_P*dz)/np.cos(t)

    def p_fx(theta,alpha,j,i,fe,fa,ft):
        return fe(theta,alpha) * fa(theta,alpha) * ft(theta,alpha,j,i)

    def p_e_L(t,a):
        return dp_e_L(t,a)
        
    def p_e_P(t,a):
        return (1/(2*np.pi))*(1/2)*np.sin(t)

    def p_a_L2L(t,a):
        return 1-np.exp(-(dz*a_L(t,a))/L(t,a))
        
    def p_a_L2P(t,a):
        return 1-np.exp(-(a_P(t)*dz)/np.cos(t_L2P(Lt(t,a))))
        
    def p_a_P2L(t,a): 
        return 1-np.exp(-(dz*a_L(t_P2L(t),a))/np.cos(t_P2L(t)))
    
    def p_a_P2P(t,a): 
        return 1-np.exp(-(a_P(t)*dz)/np.cos(t))

    def f_TIR_1_L(t,a):
        return (n_L*dz*a_L(t,a))/L(t,a)+(a_P(t)*n_P*dz)/np.cos(t_L2P(Lt(t,a)))
        
    def f_TIR_1_P(t,a): 
        return (n_L*dz*a_L(t_P2L(t),a))/np.cos(t_P2L(t))+(a_P(t)*n_P*dz)/np.cos(t)

    def p_t_L2L_ref_TIR_1_UU2(t,l,p,a): 
        return np.exp(-((l*dz*a_L(t,a))/L(t,a)))*np.exp(-((a_P(t)*p*dz)/np.cos(t_L2P(Lt(t,a)))))*np.exp(-1*f_TIR_1_L(t,a))/(1-np.exp(-2*f_TIR_1_L(t,a)))
        
    def p_t_L2L_ref_TIR_1_UD2(t,l,a):
        return np.exp(-((l*dz*a_L(t,a))/L(t,a)))*(1-np.exp(-2*f_TIR_1_L(t,a)))**-1

    def p_t_L2L_ref_TIR_1_DU2(t,l,p,a):
        return np.exp(-((l*dz*a_L(t,a))/L(t,a)))*np.exp(-((a_P(t)*p*dz)/np.cos(t_L2P(Lt(t,a)))))*(np.exp(2.*f_TIR_1_L(t,a))-1)**-1
        
    def p_t_L2L_ref_TIR_1_DD2(t,l,p,a):
        return np.exp(-((l*dz*a_L(t,a))/L(t,a)))*np.exp(-((a_P(t)*p*dz)/np.cos(t_L2P(Lt(t,a)))))*np.exp(-1*f_TIR_1_L(t,a))/(1-np.exp(-2.*f_TIR_1_L(t,a)))

    def p_t_L2P_ref_TIR_1_UU2(t, l, p,a):
        return (np.exp(-((l * dz * a_L(t,a)) / L(t,a))) * np.exp(-((a_P(t) * p * dz) / np.cos(t_L2P(Lt(t,a))))) * np.exp(-f_TIR_1_L(t,a)) / (1 - np.exp(-2 * f_TIR_1_L(t,a))))
        
    def p_t_L2P_ref_TIR_1_UD2(t, l, p,a):
        return (np.exp(-((l * dz * a_L(t,a)) / L(t,a))) * np.exp(-((a_P(t) * p * dz) / np.cos(t_L2P(Lt(t,a))))) / (1 - np.exp(-2 * f_TIR_1_L(t,a))))
        
    def p_t_L2P_ref_TIR_1_DU2(t, l, p,a):
        return (np.exp(-((l * dz * a_L(t,a)) / L(t,a))) * np.exp(-((a_P(t) * p * dz) / np.cos(t_L2P(Lt(t,a))))) / (np.exp(2 * f_TIR_1_L(t,a)) - 1))

    def p_t_L2P_ref_TIR_1_DD2(t, l, p,a):
        return (np.exp(-((l * dz * a_L(t,a)) / L(t,a))) * np.exp(-((a_P(t) * p * dz) / np.cos(t_L2P(Lt(t,a))))) * np.exp(-f_TIR_1_L(t,a)) / (1 - np.exp(-2 * f_TIR_1_L(t,a))))
  
    def p_t_P2L_ref_TIR_1_UU2(t, l, p,a):
        return (np.exp(-((l * dz * a_L(t_P2L(t),a)) / np.cos(t_P2L(t)))) * np.exp(-((a_P(t) * p * dz) / np.cos(t))) * np.exp(-f_TIR_1_P(t,a)) / (1 - np.exp(-2 * f_TIR_1_P(t,a))))
    
    def p_t_P2L_ref_TIR_1_UD2(t, l, p,a):
        return (np.exp(-((l * dz * a_L(t_P2L(t),a)) / np.cos(t_P2L(t)))) * np.exp(-((a_P(t) * p * dz) / np.cos(t))) / (1 - np.exp(-2 * f_TIR_1_P(t,a))))
    
    def p_t_P2L_ref_TIR_1_DU2(t, l, p,a):
        return (np.exp(-((l * dz * a_L(t_P2L(t),a)) / np.cos(t_P2L(t)))) * np.exp(-((a_P(t) * p * dz) / np.cos(t))) / (np.exp(2 * f_TIR_1_P(t,a)) - 1))
    
    def p_t_P2L_ref_TIR_1_DD2(t, l, p,a):
        return (np.exp(-((l * dz * a_L(t_P2L(t),a)) / np.cos(t_P2L(t)))) * np.exp(-((a_P(t) * p * dz) / np.cos(t))) * np.exp(-f_TIR_1_P(t,a)) / (1 - np.exp(-2 * f_TIR_1_P(t,a))))

    def p_t_P2P_ref_TIR_1_UU2(t, l, p,a):
        return (np.exp(-((l * dz * a_L(t_P2L(t),a)) / np.cos(t_P2L(t)))) * np.exp(-((a_P(t) * p * dz) / np.cos(t))) * np.exp(-f_TIR_1_P(t,a)) / (1 - np.exp(-2 * f_TIR_1_P(t,a))))
    
    def p_t_P2P_ref_TIR_1_UD2(t, l, p,a):
        return (np.exp(-((l * dz * a_L(t_P2L(t),a)) / np.cos(t_P2L(t)))) * np.exp(-((a_P(t) * p * dz) / np.cos(t))) / (1 - np.exp(-2 * f_TIR_1_P(t,a))))
    
    def p_t_P2P_ref_TIR_1_DU2(t, p,a):
        return (np.exp(-((a_P(t) * p * dz) / np.cos(t))) / (np.exp(2 * f_TIR_1_P(t,a)) - 1))
    
    def p_t_P2P_ref_TIR_1_DD2(t, l, p,a):
        return (np.exp(-((l * dz * a_L(t_P2L(t),a)) / np.cos(t_P2L(t)))) * np.exp(-((a_P(t) * p * dz) / np.cos(t))) * np.exp(-f_TIR_1_P(t,a)) / (1 - np.exp(-2 * f_TIR_1_P(t,a))))

    ############################################### TIR 2 Modes ####################################################################

    def f_TIR_2(t,a): 
        return (a_P(t)*n_P*dz)/np.cos(t)

    def p_t_P2P_ref_TIR_2_UU2(t, p,a):
        return (np.exp(-(a_P(t) * p * dz / np.cos(t))) * np.exp(-f_TIR_2(t,a)) / (1 - np.exp(-2 * f_TIR_2(t,a))))
    def p_t_P2P_ref_TIR_2_UD2(t, p,a):
        return (np.exp(-(a_P(t) * p * dz / np.cos(t))) / (1 - np.exp(-2 * f_TIR_2(t,a))))
    def p_t_P2P_ref_TIR_2_DU2(t, p,a):
        return (np.exp(-(a_P(t) * p * dz / np.cos(t))) / (np.exp(2 * f_TIR_2(t,a)) - 1))
    def p_t_P2P_ref_TIR_2_DD2(t, p,a):
        return (np.exp(-(a_P(t) * p * dz / np.cos(t))) * np.exp(-f_TIR_2(t,a)) / (1 - np.exp(-2 * f_TIR_2(t,a))))

    def P_ji_LL_self_dir(alpha,j,i):
        I_dir = 2*np.pi*quad(p_fx,0,np.pi/2,args=(alpha,j,i,dp_e_L,p_abs_L2L_self,un))[0]
        return I_dir

    def P_ji_PP_self_dir(alpha,j,i):
        I_dir = 2*np.pi*quad(p_fx,0,np.pi/2,args=(alpha,j,i,dp_e_P,p_abs_P2P_self,un))[0]
        return I_dir

    def alt_p_t_E2L_dir(theta,alpha,l): 
        return np.exp(-(l*dz*a_L(t_E2L(theta),alpha))/np.cos(t_E2L(theta)))
    
    def alt_p_t_E2P_dir(theta,alpha,p): 
        return np.exp(-((n_L*dz*a_L(t_E2L(theta),alpha))/np.cos(t_E2L(theta))+(a_P(theta)*p*dz)/np.cos(t_E2P(theta))))

    def alt_p_t_L2E_dir(theta,alpha,l): 
        return np.exp(-(l*dz*a_L(theta,alpha))/L(theta,alpha))

    def alt_p_t_L2L_dir(theta,alpha,l): 
        return np.exp(-(l*dz*a_L(theta,alpha))/L(theta,alpha))

    def alt_p_t_L2P_dir(theta,alpha,l,p): 
        return np.exp(-((l*dz*a_L(theta,alpha))/L(theta,alpha)+(a_P(theta)*p*dz)/L(t_L2P(theta),alpha)))

    def alt_p_t_P2E_dir(theta,alpha,p):
        return np.exp(-((a_P(theta)*p*dz)/np.cos(theta)+(n_L*dz*a_L(t_P2L(theta),alpha))/np.cos(t_P2L(theta))))

    def alt_p_t_P2L_dir(theta,alpha,l,p):
        return np.exp(-((a_P(theta)*p*dz)/np.cos(theta)+(l*dz*a_L(t_P2L(theta),alpha))/np.cos(t_P2L(theta))))

    def alt_p_t_P2P_dir(theta,alpha,p):
        return np.exp((-a_P(theta)*p*dz)/abs(np.cos(theta)))
    
    ############################################## Single Reflections ############################################################

    def alt_p_t_E2E_ref_1(theta,alpha): 
        return np.exp(-2*((n_L*dz*a_L(t_E2L(theta),alpha))/np.cos(t_E2L(theta))+(a_P(theta)*n_P*dz)/np.cos(t_E2P(theta))))

    def alt_p_t_E2L_ref_1(theta,alpha,l): 
        return np.exp(-((l*dz*a_L(t_E2L(theta),alpha))/np.cos(t_E2L(theta))+2*(a_P(theta)*n_P*dz)/np.cos(t_E2P(theta))))

    def alt_p_t_E2P_ref_1(theta,alpha,p):
        return np.exp(-((n_L*dz*a_L(t_E2L(theta),alpha))/np.cos(t_E2L(theta))+(a_P(theta)*p*dz)/np.cos(t_E2P(theta))))

    def alt_p_t_L2E_ref_1(theta,alpha,l): 
        return np.exp(-((l*dz*a_L(theta,alpha))/L(theta,alpha)+2*(a_P(theta)*n_P*dz)/np.cos(t_L2P(Lt(theta,alpha)))))

    def alt_p_t_P2E_ref_1(theta,alpha,p): 
        return np.exp(-((a_P(theta)*p*dz)/np.cos(theta)+(n_L*dz*a_L(t_P2L(theta),alpha))/np.cos(t_P2L(theta))))

    def alt_p_t_L2L_ref_1(theta,alpha,l):
        return np.exp(-((l*dz*a_L(theta,alpha))/L(theta,alpha)+2*(a_P(theta)*n_P*dz)/np.cos(t_L2P(Lt(theta,alpha)))))

    def alt_p_t_P2P_ref_1(theta,alpha,p):
        return np.exp(-(a_P(theta)*p*dz)/abs(np.cos(theta)))

    def alt_p_t_L2P_ref_1(theta,alpha,l,p):
        return np.exp(-(a_L(theta,alpha)*(l*dz))/L(theta,alpha))*np.exp(-(a_P(theta)*dz*p)/np.cos(t_L2P(Lt(theta,alpha))))

    def alt_p_t_P2L_ref_1(theta,alpha,l,p):
        return np.exp(-((a_P(theta)*p*dz)/np.cos(theta)+(l*dz*a_L(t_P2L(theta),alpha))/np.cos(t_P2L(theta))))

    def alt_P_ref_TIR_1_L2L(a):
        t = np.linspace(t_E2L(t_AR), np.pi/2, 400)  
        I, J = np.meshgrid(np.arange(2, n_L + 2), np.arange(2, n_L + 2), indexing='ij')
        I3 = I[None,...]
        J3 = J[None,...]                  
        P1_vol = 2*np.pi*p_e_L(t[:,None,None],a)*p_t_L2L_ref_TIR_1_UU2(t[:,None,None], I3 + n_L - J3 - 0.5, n_P,a)*p_a_L2L(t[:,None,None],a)
        P2_vol = 2*np.pi*p_e_L(t[:,None,None],a)*p_t_L2L_ref_TIR_1_UD2(t[:,None,None], I3 + J3 - 3.5,a)*p_a_L2L(t[:,None,None],a)
        P3_vol = 2*np.pi*p_e_L(t[:,None,None],a)*p_t_L2L_ref_TIR_1_DU2(t[:,None,None], 2 * n_L + 2.5 - I3 - J3, 2 * n_P,a)*p_a_L2L(t[:,None,None],a)
        P4_vol = 2*np.pi*p_e_L(t[:,None,None],a)*p_t_L2L_ref_TIR_1_DD2(t[:,None,None], n_L + J3 - I3 - 0.5, n_P,a)*p_a_L2L(t[:,None,None],a)
        P1 = simpson(P1_vol, t, axis=0)       
        P2 = simpson(P2_vol, t, axis=0)
        P3 = simpson(P3_vol, t, axis=0)
        P4 = simpson(P4_vol, t, axis=0)
        return P1 + P2 + P3 + P4
    
    def alt_P_ref_TIR_1_L2P(a):
        t = np.linspace(t_E2L(t_AR), np.pi/2, 400)  
        J, I = np.meshgrid(np.arange(n_L + 2, n_tot + 1), np.arange(2, n_L + 2), indexing='ij')
        I3 = I[None,...]
        J3 = J[None,...]                 
        P1_vol = 2*np.pi*p_e_L(t[:,None,None],a)*     p_t_L2P_ref_TIR_1_UU2(t[:,None,None], I3 - 1.5, n_tot - J3,a)    *p_a_L2P(t[:,None,None],a)
        P2_vol = 2*np.pi*p_e_L(t[:,None,None],a)*     p_t_L2P_ref_TIR_1_UD2(t[:,None,None], I3 - 1.5 + n_L, J3 - n_L - 2,a)    *p_a_L2P(t[:,None,None],a)
        P3_vol = 2*np.pi*p_e_L(t[:,None,None],a)*     p_t_L2P_ref_TIR_1_DU2(t[:,None,None], n_L + 1.5 - I3, n_P + n_tot - J3,a)    *p_a_L2P(t[:,None,None],a)
        P4_vol = 2*np.pi*p_e_L(t[:,None,None],a)*     p_t_L2P_ref_TIR_1_DD2(t[:,None,None], 2 * n_L + 1.5 - I3, n_P + J3 - n_L - 2,a)    *p_a_L2P(t[:,None,None],a)
        P1 = simpson(P1_vol, t, axis=0)       
        P2 = simpson(P2_vol, t, axis=0)
        P3 = simpson(P3_vol, t, axis=0)
        P4 = simpson(P4_vol, t, axis=0)
        return P1 + P2 + P3 + P4

    def alt_P_ref_TIR_1_P2L(a):
        t = np.linspace(t_E2P(t_AR),t_c_P2L,400)
        J,I = np.meshgrid(np.arange(2, n_L+2), np.arange(n_L+2, n_tot+1), indexing='ij')
        I3 = I[None,...]
        J3 = J[None,...]                  
        P1_vol = 2*np.pi*p_e_P(t[:,None,None],a)*     p_t_P2L_ref_TIR_1_UU2(t[:,None,None], 2*n_L+1-J3, n_P+I3-n_L-1.5,a)    *p_a_P2L(t[:,None,None],a)
        P2_vol = 2*np.pi*p_e_P(t[:,None,None],a)*     p_t_P2L_ref_TIR_1_UD2(t[:,None,None], n_L+J3-2,I3-n_L-1.5,a)    *p_a_P2L(t[:,None,None],a)
        P3_vol = 2*np.pi*p_e_P(t[:,None,None],a)*     p_t_P2L_ref_TIR_1_DU2(t[:,None,None], n_L+1-J3,n_P+n_tot-I3+0.5,a)    *p_a_P2L(t[:,None,None],a)
        P4_vol = 2*np.pi*p_e_P(t[:,None,None],a)*     p_t_P2L_ref_TIR_1_DD2(t[:,None,None], J3-2,n_tot-I3+0.5,a)    *p_a_P2L(t[:,None,None],a)
        P1 = simpson(P1_vol, t, axis=0)       
        P2 = simpson(P2_vol, t, axis=0)
        P3 = simpson(P3_vol, t, axis=0)
        P4 = simpson(P4_vol, t, axis=0)
        return P1 + P2 + P3 + P4

    def alt_P_ref_TIR_1_P2P(a):
        t = np.linspace(t_E2P(t_AR), t_c_P2L,400)
        J, I = np.meshgrid(np.arange(n_L + 2, n_tot + 1), np.arange(n_L + 2, n_tot + 1), indexing='ij')
        I3 = I[None,...]
        J3 = J[None,...]                  
        P1_vol = 2*np.pi*p_e_P(t[:,None,None],a)*     p_t_P2P_ref_TIR_1_UU2(t[:,None,None], n_L, I3 - J3 + n_tot - n_L - 1.5,a)    *p_a_P2P(t[:,None,None],a)
        P2_vol = 2*np.pi*p_e_P(t[:,None,None],a)*     p_t_P2P_ref_TIR_1_UD2(t[:,None,None], 2 * n_L, I3 + J3 - 2 * n_L - 3.5,a)    *p_a_P2P(t[:,None,None],a)
        P3_vol = 2*np.pi*p_e_P(t[:,None,None],a)*     p_t_P2P_ref_TIR_1_DU2(t[:,None,None], 2 * n_tot - I3 - J3 + 0.5,a)    *p_a_P2P(t[:,None,None],a)
        P4_vol = 2*np.pi*p_e_P(t[:,None,None],a)*     p_t_P2P_ref_TIR_1_DD2(t[:,None,None], n_L, n_tot - I3 + J3 - n_L - 1.5,a)    *p_a_P2P(t[:,None,None],a)
        P1 = simpson(P1_vol, t, axis=0)       
        P2 = simpson(P2_vol, t, axis=0)
        P3 = simpson(P3_vol, t, axis=0)
        P4 = simpson(P4_vol, t, axis=0)
        

        return P1 + P2 + P3 + P4
    
    def alt_P_ref_TIR_2_P2P(a):
        t = np.linspace(t_c_P2L, np.pi / 2,400)
        J, I = np.meshgrid(np.arange(n_L + 2, n_tot + 1), np.arange(n_L + 2, n_tot + 1), indexing='ij')
        I3 = I[None,...]
        J3 = J[None,...]                  
        P1_vol = 2*np.pi*p_e_P(t[:,None,None],a)*     p_t_P2P_ref_TIR_2_UU2(t[:,None,None], I3 - J3 - n_L + n_tot - 1.5,a)    *p_a_P2P(t[:,None,None],a)
        P2_vol = 2*np.pi*p_e_P(t[:,None,None],a)*     p_t_P2P_ref_TIR_2_UD2(t[:,None,None], I3 + J3 - 2 * n_L - 3.5,a)    *p_a_P2P(t[:,None,None],a)
        P3_vol = 2*np.pi*p_e_P(t[:,None,None],a)*     p_t_P2P_ref_TIR_2_DU2(t[:,None,None], 2 * n_tot - I3 - J3 + 0.5,a)    *p_a_P2P(t[:,None,None],a)
        P4_vol = 2*np.pi*p_e_P(t[:,None,None],a)*     p_t_P2P_ref_TIR_2_DD2(t[:,None,None], n_tot - n_L - I3 + J3 - 1.5,a)    *p_a_P2P(t[:,None,None],a)
        P1 = simpson(P1_vol, t, axis=0)       
        P2 = simpson(P2_vol, t, axis=0)
        P3 = simpson(P3_vol, t, axis=0)
        P4 = simpson(P4_vol, t, axis=0)
        return P1 + P2 + P3 + P4

    def alt_P_dir_calc(P_dir):
        J1 = np.arange(2,n_L+2)
        P_dir[1:n_L+1,0] = alt_p_t_E2L_dir(t_Sun,dev,J1+-2)*p_abs_E2L(t_Sun,dev) ### Solar modification
        P_dir[0,1:n_L+1] = quad_vec(lambda t: 2*np.pi*dp_e_L(t, dev)*p_abs_X2E(t,dev)*alt_p_t_L2E_dir(t,dev,J1-1.5),0,t_E2L(t_AR))[0]
        J2 = np.arange(n_L+2,n_tot+1)
        P_dir[n_L+1:n_tot,0] = 1*alt_p_t_E2P_dir(t_Sun,dev,(J2)-2-n_L)*p_abs_E2P(t_Sun,dev) ### Solar Modification
        P_dir[0,n_L+1:n_tot] = quad_vec(lambda t: 2*np.pi*dp_e_P(t,dev)*p_abs_X2E(t,dev)*alt_p_t_P2E_dir(t,dev,J2-n_L-1.5),0,t_E2P(t_AR))[0]
        J, I = np.meshgrid(np.arange(2,n_L+2),np.arange(2,n_L+2),indexing='ij')
        J3 = J[None,...]
        I3 = I[None,...]
        PL2L_temp = quad_vec(lambda t: 2*np.pi*dp_e_L(t,dev)*p_abs_L2L(t,dev)*alt_p_t_L2L_dir(t,dev,abs((J3)-(I3))-0.5),0,np.pi/2)[0]
        P_dir[1:n_L+1,1:n_L+1] = PL2L_temp.reshape((n_L,n_L))
        J, I = np.meshgrid(np.arange(n_L+2, n_tot+1),np.arange(2, n_L+2), indexing='ij')
        J4 = J[None,...]
        I4 = I[None,...]
        PL2P_temp = quad_vec(lambda t: 2*np.pi*dp_e_L(t,dev)*p_abs_L2P(t,dev)*alt_p_t_L2P_dir(t,dev,n_L+1.5-(I4),(J4)-n_L-2),0,np.pi/2)[0]
        P_dir[n_L+1:,1:n_L+1] = PL2P_temp.reshape((n_P,n_L))
        J, I = np.meshgrid(np.arange(2, n_L+2), np.arange(n_L+2, n_tot+1), indexing='ij')
        J5 = J[None,...]
        I5 = I[None,...]
        PP2L_temp = quad_vec(lambda t: 2*np.pi*dp_e_P(t,dev)*p_abs_P2L(t,dev)*alt_p_t_P2L_dir(t,dev,n_L+1-(J5),(I5)-n_L-1.5),0,t_c_P2L)[0]
        P_dir[1:n_L+1,n_L+1:] = PP2L_temp.reshape((n_L,n_P))
        J,I = np.meshgrid(np.arange(n_L+2,n_tot+1),np.arange(n_L+2,n_tot+1),indexing='ij')
        J6 = J[None,...]
        I6 = I[None,...]
        PP2P_temp = quad_vec(lambda t: 2*np.pi*dp_e_P(t,dev)*p_abs_P2P(t,dev)*alt_p_t_P2P_dir(t,dev,abs((J6)-(I6))-0.5),0,np.pi/2)[0]
        P_dir[n_L+1:,n_L+1:] = PP2P_temp.reshape((n_P,n_P))
        for i in range(1,n_L+1):
            P_dir[i][i] = P_ji_LL_self_dir(dev,0,i)
        for i in range(n_L+1,n_tot):
            P_dir[i][i] = P_ji_PP_self_dir(dev,0,i)
        return P_dir

    def alt_P_ref_1_calc(P_ref_1):
        P_ref_1[0][0] = alt_p_t_E2E_ref_1(t_Sun,dev)*p_abs_X2E() ### Solar modification
        J1 = np.arange(2,n_L+2)
        P_ref_1[1:n_L+1,0] = 1*alt_p_t_E2L_ref_1(t_Sun,dev,2*n_L+1-J1)*p_abs_E2L(t_Sun,dev) # Solar modification
        P_ref_1[0,1:n_L+1] = quad_vec(lambda t: 2*np.pi*dp_e_L(t, dev)*p_abs_X2E(t,dev)*alt_p_t_L2E_ref_1(t,dev,2*n_L+1.5-J1),0,t_E2L(t_AR))[0]
        J2 = np.arange(n_L+2,n_tot+1)
        P_ref_1[n_L+1:n_tot,0] = 1*alt_p_t_E2P_ref_1(t_Sun,dev,n_P+n_tot-J2)*p_abs_E2P(t_Sun,dev) # Solar modification
        P_ref_1[0,n_L+1:n_tot] = quad_vec(lambda t: 2*np.pi*dp_e_P(t,dev)*p_abs_X2E(t,dev)*alt_p_t_P2E_ref_1(t,dev,n_tot+n_P-J2+0.5),0,t_E2P(t_AR))[0]
        J, I = np.meshgrid(np.arange(2,n_L+2),np.arange(2,n_L+2),indexing='ij')
        J3 = J[None,...]
        I3 = I[None,...]
        PL2L_temp = quad_vec(lambda t: 2*np.pi*dp_e_L(t,dev)*p_abs_L2L(t,dev)*alt_p_t_L2L_ref_1(t,dev,2*n_L+2.5-I3-J3),0,np.pi/2)[0]
        P_ref_1[1:n_L+1,1:n_L+1] = PL2L_temp.reshape((n_L,n_L))
        J, I = np.meshgrid(np.arange(n_L+2, n_tot+1),np.arange(2, n_L+2), indexing='ij')
        J4 = J[None,...]
        I4 = I[None,...]
        PL2P_temp = quad_vec(lambda t: 2*np.pi*dp_e_L(t,dev)*p_abs_L2P(t,dev)*alt_p_t_L2P_ref_1(t,dev,n_L+1.5-I4,n_tot+n_P-J4),0,np.pi/2)[0]
        P_ref_1[n_L+1:,1:n_L+1] = PL2P_temp.reshape((n_P,n_L))
        J, I = np.meshgrid(np.arange(2, n_L+2), np.arange(n_L+2, n_tot+1), indexing='ij')
        J5 = J[None,...]
        I5 = I[None,...]
        PP2L_temp = quad_vec(lambda t: 2*np.pi*dp_e_P(t,dev)*p_abs_P2L(t,dev)*alt_p_t_P2L_ref_1(t,dev,n_L+1-J5,n_tot+n_P-I5+0.5),0,t_c_P2L)[0]
        P_ref_1[1:n_L+1,n_L+1:] = PP2L_temp.reshape((n_L,n_P))
        J,I = np.meshgrid(np.arange(n_L+2,n_tot+1),np.arange(n_L+2,n_tot+1),indexing='ij')
        J6 = J[None,...]
        I6 = I[None,...]
        PP2P_temp = quad_vec(lambda t: 2*np.pi*dp_e_P(t,dev)*p_abs_P2P(t,dev)*alt_p_t_P2P_ref_1(t,dev,2*n_tot+0.5-I6-J6),0,np.pi/2)[0]
        P_ref_1[n_L+1:,n_L+1:] = PP2P_temp.reshape((n_P,n_P))
        return P_ref_1

    P_ref_1_e = np.zeros((n_tot,n_tot))
    P_dir_e = np.zeros((n_tot,n_tot))
    P_dir = alt_P_dir_calc(P_dir_e)
    P_ref_1 = alt_P_ref_1_calc(P_ref_1_e)
    TIR_1_L2L = alt_P_ref_TIR_1_L2L(dev)
    TIR_1_L2P = alt_P_ref_TIR_1_L2P(dev)
    TIR_1_P2L = alt_P_ref_TIR_1_P2L(dev)
    TIR_1_P2P = alt_P_ref_TIR_1_P2P(dev)
    P_ref_TIR_1[1:n_L+1,1:n_L+1] = TIR_1_L2L.reshape((n_L,n_L))
    P_ref_TIR_1[n_L+1:,1:n_L+1] = TIR_1_L2P.reshape((n_P, n_L))
    P_ref_TIR_1[1:n_L+1, n_L+1:] = TIR_1_P2L.reshape((n_L, n_P))
    P_ref_TIR_1[n_L+1:, n_L+1:] = TIR_1_P2P.reshape((n_P, n_P))
    TIR_2_P2P = alt_P_ref_TIR_2_P2P(dev)
    P_ref_TIR_2[n_L+1:, n_L+1:] = TIR_2_P2P.reshape((n_P, n_P))

    P = P_dir + P_ref_1 + P_ref_TIR_1 + P_ref_TIR_2
    
    Pcopy = copy.deepcopy(P)
    P_mod = Pcopy
    P_mod[:,1:n_L+1] = QY_L*Pcopy[:,1:n_L+1]
    P_mod[0,1:n_L+1] = 1-QY_L+Pcopy[0,1:n_L+1]
    vfill = np.zeros((n_tot,len(P_alt_array)))
    P_mod_array = copy.deepcopy(P_mod)
    P_mod_tiled = np.tile(P_mod_array[...,None],(1,1,len(P_alt_array)))
    factor = QY_P*(1-P_alt_array)
    P_mod_tiled[:,n_L+1:,:] *= factor
    P_mod_tiled[0, n_L+1:, :] += 1 - factor
    def vfill_tiles(P_mod_tiles):
        vfill = np.zeros((n_tot,len(P_alt_array)))
        K = P_mod_tiles.shape[2]
        results = [eigs(P_mod_tiles[:, :, i], k=1, which='LM', sigma=1.0) for i in range(K)]
        for i in range(K):
            EV, V = results[i]
            V = V.real.flatten()
            V = V / np.sum(V)  # Normalize to sum to 1
            vfill[:, i] = V
        return vfill
    vfill = vfill_tiles(P_mod_tiled)
    tst = QY_P*(sum(Pcopy[0:n_L+1,n_L+1:n_tot])+sum(P_ref_TIR_1[n_L+1:n_tot,n_L+1:n_tot]))
    


    tst_t = tst[:,None]
    em_prob_scaled = tst_t*(1-P_alt_array)
    em_prob_scaled[0,:]



    eta_array = []
    for loop in range(len(E_g_array)):
        E_g = E_g_array[loop]
        E_L = 1e9 * (h * c) / E_g 
        x_g = E_g / (k_B * T_c)
        
        wl_range_power = np.arange(280, 4000.001, 0.001)

        interp_photons_DC = interp1d(wavelength, photons_DC, bounds_error=False, fill_value=0)
        interp_photons_GT = interp1d(wavelength, photons_GT, bounds_error=False, fill_value=0)

        @lru_cache(maxsize=2)  # Caches both 'GT' and 'DC' spectra
        def cached_photon_flux(spectra_type, E_g):
            wl_max = 1e9 * (h * c) / E_g
            wl_range = np.arange(280, wl_max, 0.001)
            if spectra_type == 'GT':
                return np.trapz(interp_photons_GT(wl_range), wl_range), np.trapz(power_GT_interp, wl_range_power)
            else:
                return np.trapz(interp_photons_DC(wl_range), wl_range), np.trapz(power_DC_interp, wl_range_power)
        tot_inc_flux, tot_inc_power = cached_photon_flux('GT' if sunlight==0 else 'DC', E_g)
        
        def vec_fit(E_g):
            threshold = (E_g-k_B*T_c)/q
            V_app_fit_lower = V_app_fit[V_app_fit<threshold]
            V_app_fit_upper = V_app_fit[V_app_fit>=threshold]
            x_c_lower = (q * V_app_fit_lower[:]) / (k_B * T_c)
            x_c_upper = (q*V_app_fit_upper[:])/(k_B*T_c)
            fun_lower = lambda x: (x**2)/(np.exp(x-x_c_lower)-1)
            fun_upper = lambda x: (x**2)*np.exp(x_c_upper-x)
            n_emit_low = quad_vec(fun_lower,x_g,np.inf,epsabs=1e-12,epsrel=1e-12)[0]
            n_emit_up = quad_vec(fun_upper,x_g,np.inf,epsabs=1e-12,epsrel=1e-12)[0]
            n_emit_fit2 = np.concatenate((n_emit_low,n_emit_up))*((2 * (k_B * T_c)**3) / ((h**3) * (c**2))) * np.pi
            return n_emit_fit2
        n_emit_fit = vec_fit(E_g)
        D_ss_e = np.zeros((n_tot, len(P_alt_array)))
        def D_ss_calc(D_ss):
            for i in range(len(P_alt_array)):
                
                V = vfill[:,i]
                D_ss[:, i] = V*(tot_inc_flux/V[0])
            return D_ss
        D_ss = D_ss_calc(D_ss_e)
        n_emit = np.sum(D_ss[n_L+1:, :] * em_prob_scaled,axis=0)
        J_tot = q * np.sum(D_ss[n_L+1:, :] * P_alt_array[np.newaxis, :] * QY_P, axis=0)
        J_tot = q * np.sum(D_ss[n_L+1:n_tot, :] * P_alt_array * QY_P, axis=0)
        n_emit_grid, n_emit_fit_grid = np.meshgrid(n_emit, n_emit_fit)
        diff = np.abs(n_emit_grid - n_emit_fit_grid)
        ind_min_diff = np.argmin(diff, axis=0)
        V_app = np.zeros(len(P_alt_array))
        V_app = V_app_fit[ind_min_diff]
        power = V_app * J_tot
        max_power_idx = np.argmax(power)
        eta = 100 * power[max_power_idx] / tot_inc_power
        J_SC = J_tot[-1]
        V_OC = V_app[0]
        V_MPP = V_app[max_power_idx]
        eta_array.append(eta)
        eta_tilt.append(eta)
        
        return_vals.append(abs_v)
        return_vals.append(n_ref_L)
        return_vals.append(np.round(E_g_array[0]/q,2))
        return_vals.append(np.round(n_L*dz*1e6,1))
        return_vals.append(np.round(n_P*dz*1e6,1))
        return_vals.append(np.rad2deg(dev_angle))
        return_vals.append(QY_v)
        return_vals.append(np.rad2deg(sun_v))
        return_vals.append(round(eta,1))
        return_vals.append(round(V_OC,2))
        return_vals.append(round(V_MPP,2))
        return_vals.append(round(J_SC/10,1))
    return return_vals


from multiprocessing import Pool 
from multiprocessing import Process
time_s = time.time()
n_lst = [1,2,3,4]
abs_lst = [0]
tilt_lst = np.deg2rad(np.linspace(0,10,30))
sun_lst = np.deg2rad(np.logspace(-1,np.log10(40),30))
q_lst = np.logspace((np.log10(0.8)),0,30)
qy_lst = 1-np.logspace(-3,np.log10(0.2),30)
qy_lst = 1-np.logspace(-6,-2,30)
qy_lsts = [1]
for i in qy_lst:
    qy_lsts.append(i)
qy_lsts
q_lst = np.array(qy_lsts)

current_var = np.array([0]) ### Ideal
current_param = len(current_var)
table_arr = np.zeros((current_param,9))

abs_vs          =        np.zeros(current_param)
n_ref_L_vs      =    np.zeros(current_param)
Eg_vs           =  np.zeros(current_param)
N_L_vs          =  np.zeros(current_param)
N_P_vs          =  np.zeros(current_param)
dev_angle_vs    =  np.zeros(current_param)
QY_vs           =         np.zeros(current_param)
sun_vs          =        np.zeros(current_param)
eta_vs          =        np.zeros(current_param)
V_OC_vs         =       np.zeros(current_param)
V_MPP_vs        =      np.zeros(current_param)
J_SC_vs         =       np.zeros(current_param)

if __name__ == '__main__':
    with Pool(4) as pp:
        args = [(0,3.666,ideal,1,0) for ideal in [0]] ### Loops over tilt angle

        res = pp.starmap(vec_Markov_parallel,args)    
        ress = np.array([i for i in res])
        for i in range(len(ress)):
            abs_vs[i] = ress[i][0]
            n_ref_L_vs[i] = ress[i][1]
            Eg_vs[i] = ress[i][2]
            N_L_vs[i] = ress[i][3]
            N_P_vs[i] = ress[i][4]
            dev_angle_vs[i] = ress[i][5]
            QY_vs[i] = ress[i][6]
            sun_vs[i] = ress[i][7]
            eta_vs[i] = ress[i][8]
            V_OC_vs[i] = ress[i][9]
            V_MPP_vs[i] = ress[i][10]
            J_SC_vs[i] = ress[i][11] 
        
df = pd.DataFrame({
    'A_L (m-1)':abs_vs,
    'n_LSC':n_ref_L_vs,
    'Band Gap (eV)':Eg_vs,
    'LSC Thickness (um)':N_L_vs,
    'PV Thickness (um)':N_P_vs,
    'NR Tilt (deg)':dev_angle_vs,
    'QY':QY_vs,
    'Solar Angle (deg)':sun_vs,
    'Efficiency (%)':eta_vs,
    'V_OC (V)':V_OC_vs,
    'V_MPP (V)':V_MPP_vs,
    'J_SC (mA/cm2)':J_SC_vs
})
output = 'PATH/TO/OUTPUT/'
output = '/Users/jon/Documents/test.csv'
df.to_csv(output,index=False)

time_f = time.time()
print(time_f-time_s)
    
