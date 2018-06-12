from amari import *

w_ex, w_in     = 4,  2
sig_ex, sig_in = 2,  3.5
g_in           = 0.5

kernel = w_ex*np.exp(-X**2/(2*sig_ex**2)) - w_in*np.exp(-X**2/(2*sig_in**2)) - g_in


