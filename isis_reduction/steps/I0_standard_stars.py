import os
import numpy as np
import src.specsiser as sr
from pipeline import SpectraReduction
from astropy.io import fits
from shutil import copyfile


def delete_files_function(file_list):
    for file_path in file_list:
        if os.path.isfile(file_path):
            os.remove(file_path)
    return


# Spectra folder
data_folder = '/home/vital/Astro-data/Observations/LzLCS_objs_ISIS'
conf_file = '../reduction_conf.ini'
obsData = sr.loadConfData(conf_file)
command_log_address = obsData['data_location']['command_log_location']

# Operation details
colors = ['Blue', 'Red']
task_name = 'standard'
data_dict = {'reduc_tag': 'standard'}

# Loop through the nights
for night in obsData['data_location']['night_list']:

    print(f'\nDoing: {night}')

    # Establish night configuration
    pr = SpectraReduction(data_folder, obs_file=None)
    night_conf = obsData[night]
    night_folder = f'{data_folder}/{night}'
    objs, std_stars = night_conf['objects_list'], night_conf['standard_star_list']

    pr.declare_catalogue(night_folder, objs=objs, std_stars=std_stars)
    run_folder = pr.reducFolders['objects']

    for arm_color in colors:

        color_label = f'{arm_color}_arm'

        idcs_objs = (pr.reducDf.frame_tag.isin(std_stars)) & \
                    (pr.reducDf.reduc_tag == 'trace_spec') & \
                    (pr.reducDf.ISIARM == color_label) & \
                    (pr.reducDf.valid_file)
        obj_labels = pr.reducDf.loc[idcs_objs].index.values
        obj_cenwaves = pr.reducDf.loc[idcs_objs].CENWAVE.values.astype(float)

        # Compute the unique configurations for each night given the cenwave and save them
        obj_cenwaves.sort()
        unique_cenwaves = obj_cenwaves[~(np.triu(np.abs(obj_cenwaves[:, None] - obj_cenwaves) <= 4, 1)).any(0)].astype(int)
        night_conf[f'night_{arm_color}_cenwaves_array'] = unique_cenwaves
        sr.safeConfData(conf_file, night_conf, section_name=night)

        # List of standard files
        std_individual_file_list = run_folder + '/' + pr.reducDf.loc[idcs_objs].frame_tag.values + f'_{arm_color}_std.txt'
        std_global_file_list = np.char.add(np.char.add(f'{run_folder}/', unique_cenwaves.astype(str)), f'_{arm_color}_std.txt')

        # Delete the files of they were already there
        delete_files_function(std_individual_file_list)
        delete_files_function(std_global_file_list)

        # Safe the night std set-ups by cenwave
        for idx in obj_labels:
            stdname = pr.reducDf.loc[idx].frame_tag
            cenwave = float(pr.reducDf.loc[idx].CENWAVE)
            ref_cenwave = unique_cenwaves[np.isclose(unique_cenwaves, b=cenwave, atol=4)][0]

            input_file = f'{pr.reducDf.loc[idx].file_name}'
            std_file_local = f'{run_folder}/{stdname}_{arm_color}_std.txt'
            std_file_global = f'{run_folder}/{ref_cenwave}_{arm_color}_std.txt'

            # Quick name conversion to get the standard star file configuration:
            if 'BD33' in stdname:
                ref_std = 'BD33'
            if 'SP0305+261' in stdname:
                ref_std = 'SP0305+261'
            if 'HD93' in stdname or 'SP1045+378' in stdname:
                ref_std = 'HD93'
            if 'SP0501' in stdname:
                ref_std = 'G191'
            if 'SP1036+433' in stdname:
                ref_std = 'feige34'
            print(stdname, '->', ref_std)
            calib_dict = pr.standar_stars_calibrationFile(ref_std)

            task_conf = {}
            task_conf['run folder'] = run_folder
            task_conf['color'] = arm_color
            task_conf['input'] = input_file
            task_conf['output'] = std_file_local
            task_conf['star_name'] = calib_dict['file name']
            task_conf['caldir'] = calib_dict['calibration_folder']
            task_conf['bandwidth'] = calib_dict['bandwidth']
            task_conf['bandsep'] = calib_dict['bandsep']
            task_conf['airmass'] = fits.getval(f'{run_folder}/{input_file}', 'AIRMASS', 0)
            task_conf['exptime'] = fits.getval(f'{run_folder}/{input_file}', 'EXPTIME', 0)

            # Prepare iraf command
            task_conf_address = pr.prepare_iraf_command(task_name, user_conf=task_conf, overwrite=False)

            # Log command
            pr.store_command(task_name, command_log_address)

            # Run the iraf command
            pr.launch_command(task_name, task_conf_address)

            # Log new files to DF
            pr.object_to_dataframe(pr.task_attributes['output'], data_dict)

            #If global file does not exist, create and fill it with the data from the first star
            if os.path.isfile(std_file_global) == False:
                copyfile(std_file_local, std_file_global)

            #otherwise append lines from current star to the global file
            else:
                with open(std_file_local, "r") as infile:
                    std_lines = infile.read()

                with open(std_file_global, "a") as outfile:
                    outfile.write(std_lines)


# print 'Data treated'


