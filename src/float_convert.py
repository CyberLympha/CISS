import struct
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def conv_float32_to_bytes(s):
    signal_bytes = [struct.pack('f',x) for x in s.values]
    
    signal_ushort8 = []
    for b in signal_bytes:
        signal_ushort8.append([x[0] for x in list(struct.iter_unpack('B',b))])
        
    b1 = [x[0] for x in signal_ushort8]
    b2 = [x[1] for x in signal_ushort8]
    b3 = [x[2] for x in signal_ushort8]
    b4 = [x[3] for x in signal_ushort8]
    
    return b1, b2, b3, b4

def plot_float32_to_bytes(s, zoom=[[0,40]]):
    b1, b2, b3, b4 = conv_float32_to_bytes(s)
        
    s.plot(figsize=(20,5), title='исходные значения')
        
    plt.figure(figsize=(20,5))
    plt.plot(range(len(b1)), b1, label='b1')
    plt.plot(range(len(b2)), b2, label='b2')
    plt.plot(range(len(b2)), b3, label='b3')
    plt.plot(range(len(b2)), b4, label='b4')
    plt.legend()
    plt.title('Значения байт')
    plt.show()

    for z in zoom:
        from_ = z[0]
        to_ = z[1]

        fig, ax = plt.subplots(nrows=5, figsize=(20,30))

        ax[1].plot(range(len(b1[from_:to_])), b1[from_:to_], label='b1')
        ax[1].set_title(f'Сэмпл {from_} ... {to_}: BYTE_1')
        ax[2].plot(range(len(b2[from_:to_])), b2[from_:to_], label='b2')
        ax[2].set_title(f'Сэмпл {from_} ... {to_}: BYTE_2')
        ax[3].plot(range(len(b3[from_:to_])), b3[from_:to_], label='b3')
        ax[3].set_title(f'Сэмпл {from_} ... {to_}: BYTE_3')
        ax[4].plot(range(len(b4[from_:to_])), b4[from_:to_], label='b4')
        ax[4].set_title(f'Сэмпл {from_} ... {to_}: BYTE_4')
        ax[0].plot(range(len(s[from_:to_])), s[from_:to_], label='raw',lw=4)
        ax[0].set_title(f'Сэмпл {from_} ... {to_}: Исходный')

        
        plt.legend()
        plt.show()
        
def plot_derivate_bytes(s, zoom=[[0,40]]):
    s_bytes = []
    
    for b in conv_float32_to_bytes(s):
        s_bytes.append(np.diff(b,1))
    
    s.plot(figsize=(20,5), title='исходные значения')
        
    plt.figure(figsize=(20,5))
       
    for i,b in enumerate(s_bytes):
        plt.plot(range(len(b)), b, label='b'+str(i))

    plt.legend()
    plt.title('Значения байт (дифференциал)')
    plt.show()

    for z in zoom:
        from_ = z[0]
        to_ = z[1]

        fig, ax = plt.subplots(nrows=5, figsize=(20,30))
        
        ax[0].plot(range(len(s[from_:to_])), s[from_:to_], label='raw',lw=4)
        ax[0].set_title(f'Сэмпл {from_} ... {to_}: Исходный')
        
        for i,b in enumerate(s_bytes):
            ax[i+1].plot(range(len(b[from_:to_])), b[from_:to_], label='b'+str(i))
            ax[i+1].set_title(f'Сэмпл {from_} ... {to_}: BYTE_'+str(i))

        plt.legend()
        plt.show()
