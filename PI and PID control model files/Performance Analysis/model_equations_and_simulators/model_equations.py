## Model equations for TCS model with integral error

import numpy as np 
import scipy.integrate as scint


def dydt(t, x, params, setpoint = 0, input = 'dark'):
    
    if input.lower() == 'dark': 
        red, green = (0, 0)

    if input.lower() == 'green': 
        red, green = (0, 1)

    if input.lower() == 'red': 
        red, green = (1, 0)

    # Unpack variables 
    S, Sp, R, Rp, Ac, M, C_tic, P, Pm, C, I = x 

    # Parameters
        
    #p = params.to_dict()
    p = params.valuesdict()


    k_green = p['k_green'] * green  # 1/min
    k_red = p['k_red'] * red  # 1/min
    b_green = p['b_green'] # 1/min
    b_red = p['b_red'] # 1/min
    k_sp_b = p['k_sp_b'] # 1/min
    k_sp_u = p['k_sp_u'] # 1/min
    k_rp_b = p['k_rp_b'] # 1/min
    k_rp_u = p['k_rp_u'] # 1/min 

    beta = p['beta'] # nM/min 
    l0 = p['l0'] # leak coefficient
    Kc = p['Kc'] # nM
    d_m = p['d_m'] # 1/min 
    k_tli_b = p['k_tli_b'] # 1/min 
    k_tli_u = p['k_tli_u'] # 1/min 
    k_tl = p['k_tl'] # 1/min 
    d_p = p['d_p'] # 1/min 
    k_fold = p['k_fold'] # 1/min
    b_fold = p['b_fold'] 
    n_delta = 6
    n_tcs = p["n_tcs"]

    n_gamma = p['n_gamma']
    R_max = p['R_max']

    k_gr = p['k_gr']
    C_max = p['C_max']

    

    f = C / C_max
    y = f * (1 - f)

    y_resource = np.power(y, n_gamma)
    y_rate = np.power(y, n_gamma)
    delta = f**n_delta/(1 +  f**n_delta)

    R_total = R_max * y_resource
    R_free = R_total - C_tic


    d_dil = k_gr * (1 - f) 
    d_m = d_m * (1 - f) + d_dil 
    d_p = d_p * delta + d_dil

    beta = beta * y_resource 
    k_fold = k_fold * (y_rate + b_fold)

    r_tl_intn = k_tli_b * M * R_free - k_tli_u * C_tic # translation initiation rate
    r_tl_elng = k_tl * C_tic # trnalsation elongation rate


    

    if setpoint is None:
        e = 0
    else: 
        e = setpoint - Pm

    
    # ODEs

    ODEs = []
    dSdt = - (k_green + b_green) * S + (k_red + b_red) * Sp + k_sp_b * Sp * R  - k_sp_u * Rp * S
    ODEs.append(dSdt) # Light sensing module 

    dSpdt =  (k_green + b_green) * S - (k_red + b_red) * Sp - k_sp_b * Sp * R  + k_sp_u * Rp * S
    ODEs.append(dSpdt) # Transcription output module

    dRdt =  - k_sp_b * Sp * R + k_sp_u * Rp * S
    ODEs.append(dRdt) # Activating complex monomer formation
    
    dRpdt = k_sp_b * Sp * R - k_sp_u * Rp * S - k_rp_b * Rp * Rp + k_rp_u * Ac 
    ODEs.append(dRpdt) # Activating complex monomer formation

    dAcdt = k_rp_b * Rp * Rp - k_rp_u * Ac
    ODEs.append(dAcdt) # Activating complex dimer formation

    dMdt = beta * (Ac**n_tcs / (Kc**n_tcs + Ac**n_tcs) + l0) - d_m * M - r_tl_intn + r_tl_elng
    ODEs.append(dMdt) # Transcription 

    dCticdt = r_tl_intn - r_tl_elng
    ODEs.append(dCticdt) # Translation initiation

    dPdt = r_tl_elng - k_fold * P - d_p * P 
    ODEs.append(dPdt) # Translation 

    dPmdt = k_fold * P - d_p * Pm 
    ODEs.append(dPmdt) # Protein folding 

    dCdt = k_gr * C * (1 - f)  
    ODEs.append(dCdt) # Logistic growth 

    dIdt = e  # Control input (error term)
    ODEs.append(dIdt) # Control input for feedback


    return ODEs

def run_TCS_model(x0, time, params, setpoint, input, output = 'solve_ivp'): 

    sol = scint.solve_ivp(dydt, [time[0], time[-1]], x0, args = (params, setpoint, input), t_eval = time, 
                          method = 'Radau')

    # # # # Extract results in odeint format
    if output == 'solve_ivp':
        return sol
    else:
        return sol.y.T