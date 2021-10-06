#!/usr/bin/python

import os

os.environ['TCL_LIBRARY'] = '/home/vital/anaconda3/envs/iraf27/lib/tcl8.6'
os.environ['TK_LIBRARY'] = '/home/vital/anaconda3/envs/iraf27/lib/tk8.6'

import sys
import numpy as np
import pyraf
from collections import OrderedDict


def run_iraf_package(task, task_conf, run_folder=None):

    if run_folder is not None:
        os.chdir(run_folder)

    print('Input run folder: ', run_folder, '\n')
    print('Actual Running folder: ', os.getcwd(), '\n')

    if task == 'zerocombine':
        pyraf.iraf.noao.imred()
        pyraf.iraf.noao.imred.ccdred()
        pyraf.iraf.noao.imred.ccdred.zerocombine(**task_conf)

    elif task == 'ccdproc':
        # With the imred package loaded type this to avoid the error no instrument loaded
        # ccdred.instrument = "ccddb$kpno/camera.dat"
        pyraf.iraf.noao.imred()
        pyraf.iraf.noao.imred.ccdred()
        pyraf.iraf.noao.imred.ccdred.ccdproc(**task_conf)

    elif task == 'flatcombine':
        pyraf.iraf.noao.imred()
        pyraf.iraf.noao.imred.ccdred()
        pyraf.iraf.noao.imred.ccdred.flatcombine(**task_conf)

    elif task == 'response':
        pyraf.iraf.noao.twodspec()
        pyraf.iraf.noao.twodspec.longslit()
        pyraf.iraf.noao.twodspec.longslit.response(**task_conf)

    elif task == 'imcombine':
        pyraf.iraf.imcombine(**task_conf)

    elif task == 'imcopy':
        pyraf.iraf.imutil()
        pyraf.iraf.imutil.imcopy(**task_conf)

    elif task == 'imshift':
        pyraf.iraf.immatch()
        pyraf.iraf.immatch.imshift(**task_conf)

    elif task == 'skyflat':
        pyraf.iraf.noao.imred()
        pyraf.iraf.noao.imred.ccdred()
        pyraf.iraf.noao.imred.ccdred.combine(**task_conf)

    elif task == 'illumination':
        pyraf.iraf.noao.twodspec()
        pyraf.iraf.noao.twodspec.longslit()
        pyraf.iraf.noao.twodspec.longslit.illumination(**task_conf)

    elif task == 'identify':
        input_address       = task_conf['images']
        input_file          = input_address[input_address.rfind("/" ) +1:len(input_address)]
        folder_input        = input_address[0:input_address.rfind('/')]
        task_conf['images'] = input_file
        pyraf.iraf.noao.twodspec()
        pyraf.iraf.noao.twodspec.longslit()
        pyraf.iraf.noao.twodspec.longslit.identify(**task_conf)

    elif task == 'reidentify':
        # Special case for identify set the working folder to the location
        input_address                   = task_conf['images']
        input_file                      = input_address[input_address.rfind("/" ) +1:len(input_address)]
        folder_input                    = input_address[0:input_address.rfind('/')]
        task_conf['images']             = input_file
        task_conf['referenc']           = input_file
        pyraf.iraf.noao.twodspec()
        pyraf.iraf.noao.twodspec.longslit()
        pyraf.iraf.noao.twodspec.longslit.reidentify(**task_conf)

    elif task == 'fitcoords':
        # Special case for identify set the working folder to the location
        input_address                   = task_conf['images']
        input_file                      = input_address[input_address.rfind("/" ) +1:len(input_address)]
        folder_input                    = input_address[0:input_address.rfind('/')]
        task_conf['images']             = input_file
        pyraf.iraf.noao.twodspec()
        pyraf.iraf.noao.twodspec.longslit()
        pyraf.iraf.noao.twodspec.longslit.fitcoords(**task_conf)

    elif task == 'transform':
        # Special case for identify set the working folder to the location
        input_address                   = task_conf['input']
        output_address                  = task_conf['output']
        input_file                      = input_address[input_address.rfind("/" ) +1:len(input_address)]
        output_file                     = output_address[output_address.rfind("/" ) +1:len(output_address)]
        folder_input                    = input_address[0:input_address.rfind('/')]
        task_conf['input']              = input_file
        task_conf['output']             = output_file
        # os.chdir(folder_input)
        pyraf.iraf.noao.twodspec()
        pyraf.iraf.noao.twodspec.longslit()
        pyraf.iraf.noao.twodspec.longslit.transform(**task_conf)

    elif task == 'dispcor':
        pyraf.iraf.noao.onedspec()
        pyraf.iraf.noao.onedspec.dispcor(**task_conf)

    elif task == 'background':
        pyraf.iraf.noao.twodspec()
        pyraf.iraf.noao.twodspec.longslit()
        pyraf.iraf.noao.twodspec.longslit.background(**task_conf)

    elif task == 'apall':
        pyraf.iraf.noao.twodspec()
        pyraf.iraf.noao.twodspec.apextract()
        pyraf.iraf.noao.twodspec.apextract.apall(**task_conf)

    elif task == 'standard':
        pyraf.iraf.noao.onedspec()
        pyraf.iraf.noao.onedspec.standard(**task_conf)

    elif task == 'sensfunc':
        pyraf.iraf.noao.onedspec()
        pyraf.iraf.noao.onedspec.sensfunc(**task_conf)

    elif task == 'calibrate':
        pyraf.iraf.noao.onedspec()
        pyraf.iraf.noao.onedspec.calibrate(**task_conf)

    elif task == 'continuum':
        pyraf.iraf.noao.onedspec()
        pyraf.iraf.noao.onedspec.continuum(**task_conf)

    elif task == 'splot':
        pyraf.iraf.noao.onedspec()
        pyraf.iraf.noao.onedspec.splot(**task_conf)

    elif task == 'sarith':
        pyraf.iraf.noao.onedspec()
        pyraf.iraf.noao.onedspec.sarith(**task_conf)

    elif task == 'imarith':
        pyraf.iraf.imutil()
        pyraf.iraf.imutil.imarith(**task_conf)

    elif task == 'imstat':
        pyraf.iraf.imutil()
        pyraf.iraf.imutil.imstat(**task_conf)

    elif task == 'dopcor':
        pyraf.iraf.noao.onedspec()
        pyraf.iraf.noao.onedspec.dopcor(**task_conf)

    # Store the final command to a text file
    # commands_to_log(task, task_conf, commands_log_address)

    return


def printIrafCommand(task, attributes, printindividually=True, verbose=True):

    keys_attrib = attributes.keys()
    values_attrib = attributes.values()

    command_String = task + ''
    for i in range(len(keys_attrib)):
        command_String = command_String + ' ' + keys_attrib[i] + '=' + str(values_attrib[i])

    if verbose:
        print(command_String)

    if printindividually:
        print('-- Task atributtes:')
        if printindividually:
            for i in range(len(keys_attrib)):
                print(keys_attrib[i] + '=' + str(values_attrib[i]))

    return command_String


def commands_to_log(task, command_str, file_address):

    tasks_logged = np.loadtxt(file_address, dtype='str', comments='--', delimiter=' ', skiprows=1, usecols=[0], ndmin=1)

    with open(file_address, 'a') as file_log:
        if (tasks_logged.size == 0):
            file_log.write('\n--' + task + '\n')
            file_log.write(command_str + '\n')
        elif (tasks_logged[-1] != task):
            file_log.write('\n--' + task + '\n')
            file_log.write(command_str + '\n')
        else:
            file_log.write(command_str + '\n')


# Get task and its configuration from argument
conf_file_address = sys.argv[1]
conf_file = os.path.basename(conf_file_address)
task_name = conf_file[0:conf_file.find('_')]

# Run folder
if len(sys.argv) > 2:
    run_folder = sys.argv[2]
else:
    run_folder = None

# Load the task configuration
dict_keys, dict_values = np.loadtxt(conf_file_address, dtype='str', delimiter=';', usecols=(0, 1), unpack=True)
task_dict = OrderedDict(zip(dict_keys, dict_values))

# Display the IRAF commands we are using
text_command = printIrafCommand(task_name, task_dict, verbose=True)

# Run the task
run_iraf_package(task_name, task_dict, run_folder)




