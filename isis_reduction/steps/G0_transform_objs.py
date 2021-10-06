import os
import numpy as np
import src.specsiser as sr
from pipeline import SpectraReduction
from shutil import copyfile
from sys import exit

# Spectra folder
data_folder = '/home/vital/Astro-data/Observations/LzLCS_objs_ISIS'
obsData = sr.loadConfData('../reduction_conf.ini')
command_log_address = obsData['data_location']['command_log_location']

# Operation details
colors = ['Blue', 'Red']
task_name = 'transform'
data_dict = {'reduc_tag': 'obj_wave_calib'}

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

        idcs_objs = (pr.reducDf.frame_tag.isin(objs + std_stars)) &\
                    (pr.reducDf.reduc_tag == 'flat_corr') & \
                    (pr.reducDf.ISIARM == color_label) &\
                    (pr.reducDf.valid_file)
        target_indeces = pr.reducDf.loc[idcs_objs].index.values

        for idx in target_indeces:

            # Input obj
            obj_label = pr.reducDf.loc[idx].frame_tag
            input_file = f'{pr.reducDf.loc[idx].file_location}/{pr.reducDf.loc[idx].file_name}'

            # Arc
            if f'{obj_label}_arc_array' not in night_conf:
                print(f'- NO Entry {obj_label}_arc_array ({pr.reducDf.loc[idx].RUN})')
                exit()
            print(obj_label, arm_color)
            arc_obj_list = night_conf[f'{obj_label}_arc_array']
            idx_arc = (pr.reducDf.frame_tag == 'arc') &\
                      (pr.reducDf.reduc_tag == 'biascorr') & \
                      (pr.reducDf.ISIARM == color_label) & \
                      (pr.reducDf.RUN.isin(arc_obj_list)) & \
                      (pr.reducDf.valid_file)

            arc_code = pr.reducDf.loc[idx_arc].index[0]
            arc_file = f'{run_folder}/database/fc{arc_code}'

            if not os.path.isfile(f'{pr.reducFolders["arcs"]}/database/fc{arc_code}'):
                print(f'- Missing for {arc_file}: arc_obj_list')
                exit()

            # Output file
            output_name = f'{pr.reducDf.loc[idx].file_location}/{obj_label}_{arm_color}_f_w.fits'

            # Security central wavelength
            obj_cenwave = float(pr.reducDf.loc[idx].CENWAVE)
            arc_cenwave = pr.reducDf.loc[idx_arc].CENWAVE.values.astype(float)
            cenwave_check = np.isclose(arc_cenwave, b=obj_cenwave, atol=4)
            if not cenwave_check:
                print(f'ARC MISMATCH for {obj_label}_arc_array ({pr.reducDf.loc[idx].RUN}): has {obj_cenwave}, {arc_code} arcs have {arc_cenwave}')
                exit()

            # Copy arc database to run folder
            if not os.path.exists(f'{run_folder}/database'):
                os.makedirs(f'{run_folder}/database')
            orig_arc_code = f'{pr.reducFolders["arcs"]}/database/fc{arc_code}'
            copy_arc_code = arc_file
            copyfile(orig_arc_code, copy_arc_code)

            # Task configuration
            task_conf = {}
            task_conf['run folder'] = run_folder
            task_conf['color'] = arm_color
            task_conf['input'] = input_file
            task_conf['output'] = output_name
            task_conf['database'] = f'database'
            task_conf['fitnames'] = arc_code

            # Prepare iraf command
            task_conf_address = pr.prepare_iraf_command(task_name, user_conf=task_conf)

            # Log command
            pr.store_command(task_name, command_log_address)

            # Run command
            pr.launch_command(task_name, task_conf_address, verbose=True)

            # Log new files to DF
            pr.object_to_dataframe(pr.task_attributes['output'], data_dict)

#
#
#
 
 
 
 
 
 
 
 
 
 
 
 


