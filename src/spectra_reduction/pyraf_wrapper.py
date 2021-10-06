import os
import numpy as np

def commands_to_log(task, task_conf, commands_log_address):
    command_String = task
    for key in task_conf:
        command_String = command_String + ' ' + key + '=' + task_conf[key]

    tasks_logged = np.loadtxt(commands_log_address, dtype='str', comments='--', delimiter=' ', skiprows=1, usecols=[0],
                           ndmin=1)

    with open(commands_log_address, 'a') as file_log:
        if (tasks_logged.size == 0):
            file_log.write('\n--' + task + '\n')
            file_log.write(command_String + '\n')
        elif (tasks_logged[-1] != task):
            file_log.write('\n--' + task + '\n')
            file_log.write(command_String + '\n')
        else:
            file_log.write(command_String + '\n')


def set_Iraf_package(task, task_conf, commands_log_address):

    import pyraf

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
        os.chdir(folder_input)
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
        os.chdir(folder_input)
        pyraf.iraf.noao.twodspec()
        pyraf.iraf.noao.twodspec.longslit()
        pyraf.iraf.noao.twodspec.longslit.reidentify(**task_conf)

    elif task == 'fitcoords':
        # Special case for identify set the working folder to the location
        input_address                   = task_conf['images']
        input_file                      = input_address[input_address.rfind("/" ) +1:len(input_address)]
        folder_input                    = input_address[0:input_address.rfind('/')]
        task_conf['images']             = input_file
        os.chdir(folder_input)
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
        os.chdir(folder_input)
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
    commands_to_log(task, task_conf, commands_log_address)

    return


def execute_Iraf_task(launching_command, commands_log, verbose_output=False):

    # Decompose comand to files
    FileAddress = launching_command.split(' ')[2]
    FileName = FileAddress[FileAddress.rfind("/") + 1:len(FileAddress)]
    task = FileName[0:FileName.find('_')]

    print('-- IRAF task:', task)

    # load the task configuration
    dict_keys, dict_values = loadtxt(FileAddress, dtype='str', delimiter=';', usecols=(0, 1), unpack=True)
    task_conf = OrderedDict(zip(dict_keys, dict_values))

    # Reproduce task name
    equivalent_Iraf_comand(task, task_conf, verbose_output)

    # Run the task
    set_Iraf_package(task, task_conf, commands_log)

    return


