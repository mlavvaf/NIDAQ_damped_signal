# Run NIDAQ box for perturbation coil tests


from RW_relay_resistor_test import NIDAQ_RW
from RW_relay_resistor_test import relay
import matplotlib.pyplot as plt
import numpy as np

  
# channel
chan = relay()               #number of the switches starts from 0 (doesn't match with the label in the box)
# chan.all_off()             #all switches off (down)
chan.all_on()                #all switches on (up)
# chan.turn_on(3)            #all switches off except ---
chan.turn_off(0)             #all switches on except ---

# chan.set_switches()
# chan.set_switches(7)

do_run          = True
 
# settings
freq_list = [60]


# print freq list
print('Freq order (Hz)')
for i in freq_list:
    print(f'{i:.2f} Hz')

save_dir = 'UW_test'

settings = {'relay_number':         1,       # Perturbation coil offset included to zero the field?
            'techron_gain':         40,
            'exp':                  'df, rm, ml',
            'title':                'UW_No, amplifier, linear, damped sine wave',
            'fluxgate_model':       'Bartington Mag13MSL100',
            'DAQ_model':            'NI USB-6281',
            't1':                   3,         #s
            't2':                   57,
            't3':                   0
           }


# make object
nidaq = NIDAQ_RW(readback_freq = 2e4,
                 samples_per_channel = 1e4, 
                 **settings
                 )
# calcuate duration
def get_duration(freq):
              
    duration = settings['t1'] + settings['t2'] + settings['t3']
    rebin = 1
    
    return (duration, rebin)

# get_duration(freq_list, **settings['tot_dur'])

# estimate run duration
total_dur = np.sum([get_duration(f)[0] for f in freq_list])
print(f'total duration: {total_dur} sec or {total_dur/60} min or {total_dur/3600} hours')


if do_run:
    plt.figure()
   
    for freq in freq_list:
        
        print(f'Time left: {total_dur/60} min')
        
        # do the signal measurement
        dur, rebin = get_duration(freq)
        nidaq.run(duration = dur,
                  freq = [freq],
                  amp  = [1],    # V
                  draw_s = 4/freq,
                  save_signal = False,
                  )
        nidaq.to_csv(save_dir=save_dir, rebin=rebin)  # use default filename scheme
        plt.clf()
    
        # time left
        total_dur -= dur

        nidaq.close()
