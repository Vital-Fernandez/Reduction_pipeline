import numpy as np
import src.specsiser as sr
from pipeline import SpectraReduction
from astropy.io import fits

# Spectra folder
data_folder = '/home/vital/Astro-data/Observations/LzLCS_objs_ISIS'
obsData = sr.loadConfData('../reduction_conf.ini')
command_log_address = obsData['data_location']['command_log_location']

# Operation details
colors = ['Blue', 'Red']
task_name = 'apall'
data_dict = {'reduc_tag': 'trace_spec'}

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
                    (pr.reducDf.reduc_tag == 'obj_wave_calib') & \
                    (pr.reducDf.ISIARM == color_label) & \
                    (pr.reducDf.valid_file)
        target_indeces = pr.reducDf.loc[idcs_objs].index.values

        for idx in target_indeces:

            # Inputs
            obj_label = pr.reducDf.loc[idx].frame_tag
            input_file = f'{pr.reducDf.loc[idx].file_location}/{pr.reducDf.loc[idx].file_name}'

            # Outputs
            output_name = f'{pr.reducDf.loc[idx].file_location}/{obj_label}_{arm_color}_f_w_e.fits'

            # Recover apall paramters
            key_label = f'{obj_label}_{color_label}_ref_peak_array'
            background_array = night_conf.get(key_label)
            if background_array == None:
                frame_data = fits.getdata(input_file, 0)
                middle_height = frame_data.shape[0]/2
                middle_width = frame_data.shape[1]/2
                w1, w2, w3, w4 = (np.percentile(np.arange(frame_data.shape[1]), (30, 40, 60, 70)) - middle_width).astype(int)
                line_entry = f'{int(middle_height)}'
                b_sample_entry = f'{w1}:{w2},{w3}:{w4}'

            task_conf = {}
            task_conf['run folder'] = run_folder
            task_conf['color'] = arm_color
            task_conf['input'] = input_file
            task_conf['output'] = output_name
            task_conf['extras'] = 'yes'
            task_conf['gain_key'] = 'GAIN'
            task_conf['readnois_key'] = 'READNOIS'
            task_conf['backgro'] = 'median'
            task_conf['trace'] = 'yes'
            task_conf['edit'] = 'yes'
            task_conf['resize'] = 'yes'
            task_conf['nsum'] = 10
            task_conf['referen'] = '""'
            task_conf['line'] = line_entry
            task_conf['ylevel'] = 0.05
            task_conf['b_order'] = 1
            task_conf['order'] = "increasing"
            task_conf['b_sample'] = b_sample_entry

            # Prepare iraf command
            task_conf_address = pr.prepare_iraf_command(task_name, user_conf=task_conf, overwrite=False)

            # Log command
            # pr.store_command(task_name, command_log_address)

            # print(obj_label, input_file)

            # # Run the iraf command
            # pr.launch_command(task_name, task_conf_address)

            # Log new files to DF
            pr.object_to_dataframe(pr.task_attributes['output'], data_dict)

    #Generate pdf file
    output_file = f'{pr.reducFolders["reduc_data"]}/std_stars_extraction'
    idcs_print = ((pr.reducDf.reduc_tag == 'trace_spec') & (pr.reducDf.frame_tag.isin(std_stars)))
    pr.generate_step_pdf(idcs_print, file_address=output_file, ext=0, include_graph=True, sorting_pattern=['ISIARM', 'reduc_tag'],
                         plots_type='extraction', obs_conf=night_conf)
