import numpy as np
import src.specsiser as sr
from pipeline import SpectraReduction
from astropy.io import fits
import pandas as pd

# Spectra folder
data_folder = '/home/vital/Astro-data/Observations/LzLCS_objs_ISIS'
conf_file = '../reduction_conf.ini'
obsData = sr.loadConfData(conf_file)
command_log_address = obsData['data_location']['command_log_location']

# Operation details
colors = ['Blue', 'Red']
task_name = 'calibrate'
data_dict = {'reduc_tag': 'calibrate'}

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

        # Select objects and std stars
        idcs_objs = (pr.reducDf.frame_tag.isin(objs + std_stars)) & \
                    (pr.reducDf.reduc_tag == 'trace_spec') & \
                    (pr.reducDf.ISIARM == color_label) & \
                    (pr.reducDf.valid_file)
        obj_idx = pr.reducDf.loc[idcs_objs].index.values
        obj_names = pr.reducDf.loc[idcs_objs].frame_tag.values
        obj_cenwaves = pr.reducDf.loc[idcs_objs].CENWAVE.values.astype(float)
        night_cenWaves = night_conf[f'night_{arm_color}_cenwaves_array'].astype(float)

        # List of calibration stars
        df_targets = pr.reducDf.loc[idcs_objs].copy()
        df_targets['CENWAVE'] = df_targets['CENWAVE'].astype(float)

        # Loop through the objects
        for objIdx, objName, objCenWave in zip(obj_idx, obj_names, obj_cenwaves):

            # Standard stars
            if objName in objs:

                # Trace object
                input_file = f'{pr.reducDf.loc[objIdx].file_name}'

                cenwaveGlobal = night_cenWaves[np.isclose(objCenWave, b=night_cenWaves, atol=4)]

                print(f'{objName} ({objCenWave}) -> \t {cenwaveGlobal}')
                assert len(cenwaveGlobal) == 1, f'WARNING: more than one global sens curve'

                sens_file = f'{run_folder}/{cenwaveGlobal[0]:.0f}_{arm_color}_sens.fits'
                output_file = f'{run_folder}/{input_file.replace(".fits", "")}' + f'_f-global-{cenwaveGlobal[0]:.0f}.fits'

                task_conf = {}
                task_conf['run folder'] = run_folder
                task_conf['color'] = arm_color
                task_conf['input'] = input_file
                task_conf['output'] = output_file
                task_conf['senstivityCurve'] = sens_file
                task_conf['airmass'] = fits.getval(f'{run_folder}/{input_file}', 'AIRMASS', 0)
                task_conf['exptime'] = fits.getval(f'{run_folder}/{input_file}', 'EXPTIME', 0)

                # Prepare iraf command
                task_conf_address = pr.prepare_iraf_command(task_name, user_conf=task_conf)

                # Log command
                pr.store_command(task_name, command_log_address)

                # Run the iraf command
                pr.launch_command(task_name, task_conf_address)

                # Log new files to DF
                pr.object_to_dataframe(pr.task_attributes['output'], data_dict)

# indeces_print = (dz.reducDf.reduc_tag == 'flux_calibrated_objects_flocal') | (dz.reducDf.reduc_tag == 'flux_calibrated_objects_fglobal')
# dz.generate_step_pdf(indeces_print, file_address = dz.reducFolders['reduc_data'] + 'calibrated_objects', plots_type = 'spectra', ext = 0)
#
# print 'Data treated'


