# About
This script shows using nidaqmx package to simultaneously and continuously write to, and read from an NI DAQ measurement card analog I/O's. It can also control relays.

# How does it work?
The script registers reading and writing "callback" functions that are hardware triggered whenever a specific amount of data had been generated on the output or logged into the input buffer. Matplotlib is used to preview the input signals. The output data will be saved in a csv file.

# How to use?
The hardware model we used is NI USB-6281. One output channel sends the damped sine wave to the amplifier, and then relay, and the output data will go to the input channels. The number of the relay that is wanted to be used must be typed out.

# Assesment
The code will produce a continuously and simultaneously sine wave, and then the signal will be linearly damped until it goes to the zero. after few seconds, the code opens the relay to get the noise from the environment.

![Alt text](file:///Users/maedeh/Downloads/Figure_1.png)
