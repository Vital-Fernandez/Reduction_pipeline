import numpy as np
import src.specsiser as sr
from pipeline import SpectraReduction
from shutil import copy as shu_copy

# Spectra folder
data_folder = '/home/vital/Astro-data/Observations/LzLCS_objs_ISIS'
obsData = sr.loadConfData('../reduction_conf.ini')
command_log_address = obsData['data_location']['command_log_location']

# Operation details
colors = ['Blue', 'Red']
task_name = 'ccdproc'
data_dict = {'reduc_tag': 'biascorr'}

# Loop through the nights
for night in obsData['data_location']['night_list']:

    print(f'\nDoing: {night}')

    # Establish night configuration
    pr = SpectraReduction(data_folder, obs_file=None)
    night_conf = obsData[night]
    night_folder = f'{data_folder}/{night}'
    objs, std_stars = night_conf['objects_list'], night_conf['standard_star_list']
    list_for_bias = std_stars + objs + ['flat', 'arc', 'sky']

    pr.declare_catalogue(night_folder, objs=objs, std_stars=std_stars)
    run_folder = pr.reducFolders["bias"]

    for arm_color in colors:

        color_label = f'{arm_color}_arm'

        #Get the files to bias correct
        indeces_arm = (pr.reducDf.ISIARM == color_label) & (pr.reducDf.file_location.str.contains('raw_fits')) & pr.reducDf.frame_tag.isin(list_for_bias) & pr.reducDf.valid_file
        frames_type = pr.reducDf.loc[indeces_arm, 'frame_tag'].values
        files_folders = pr.reducDf.loc[indeces_arm, 'file_location'].values
        files_names = pr.reducDf.loc[indeces_arm, 'file_name'].values

        #Get the correction files
        idx_zerofile = (pr.reducDf.reduc_tag == 'master_bias') & (pr.reducDf.ISIARM == color_label) & (pr.reducDf.valid_file)
        biasfile = pr.reducDf.file_location[idx_zerofile].values[0] + '/' + pr.reducDf.file_name[idx_zerofile].values[0]
        fixfile = f'{data_folder}/badpix_{arm_color}mask'

        biassec_array = obsData[night][f'{arm_color.lower()}_BIASSEC_array'].astype(int)
        trimsec_array = obsData[night][f'{arm_color.lower()}_TRIMSEC_array'].astype(int)
        biassec_region = f'[{biassec_array[0]}:{biassec_array[1]},{biassec_array[2]}:{biassec_array[3]}]'
        trimsec_region = f'[{trimsec_array[0]}:{trimsec_array[1]},{trimsec_array[2]}:{trimsec_array[3]}]'

        #Make a security copy of the bias file
        backup_file = biasfile[0:biasfile.rfind('.')] + '_backup.fits'
        shu_copy(biasfile, backup_file)

        #Generate the list of input and output files
        input_array = files_folders + '/' + files_names + '[1]'
        input_list_name = f'in_files_to_biasCorrect_{arm_color}.list'

        #Generate the output array
        output_array = np.empty(len(frames_type), dtype=object)
        for i in range(len(frames_type)):
            frame_object = frames_type[i]
            if frame_object in objs:
                output_folder = pr.reducFolders['objects']
            elif frame_object in std_stars:
                output_folder = pr.reducFolders['objects']

            elif frame_object == 'flat':
                output_folder = pr.reducFolders['flat lamp']
            elif frame_object == 'sky':
                output_folder = pr.reducFolders['flat sky']
            elif frame_object == 'arc':
                output_folder = pr.reducFolders['arcs']

            ouput_address = f'{output_folder}/{files_names[i]}'
            if ouput_address.endswith('.fit'):
                ouput_address = ouput_address.replace('.fit', '_b.fits')
            elif ouput_address.endswith('.fits'):
                ouput_address = ouput_address.replace('.fits', '_b.fits')
            output_array[i] = ouput_address
        output_list_file = f'out_files_to_biasCorrect_{arm_color}.list'

        #Set the task attributes
        task_conf = {}

        task_conf['color']         = arm_color
        task_conf['run folder']    = run_folder
        task_conf['input array']   = input_array
        task_conf['in_list_name']  = input_list_name

        task_conf['input']         = f'@{run_folder}/{input_list_name}'
        task_conf['output array']  = output_array
        task_conf['out_list_name'] = output_list_file
        task_conf['output']        = f'@{run_folder}/{output_list_file}'

        task_conf['fixpix']        = 'yes'
        task_conf['fixfile']       = fixfile
        task_conf['oversca']       = 'yes'
        task_conf['biassec']       = biassec_region
        task_conf['trim']          = 'yes'
        task_conf['trimsec']       = trimsec_region
        task_conf['zerocor']       = 'yes'
        task_conf['zero']          = biasfile

        # Prepare iraf command
        task_conf_address = pr.prepare_iraf_command(task_name, user_conf=task_conf)

        # Log command
        pr.store_command(task_name, command_log_address)

        # Run the iraf command
        pr.launch_command(task_name, task_conf_address)

        # Log new files to DF
        pr.object_to_dataframe(pr.task_attributes['output array'], data_dict)

        #Save the ccdproc treated bias frame the bias file
        ccdproc_bias = biasfile.replace('.fits', '_b.fits')
        shu_copy(biasfile, ccdproc_bias)
        pr.object_to_dataframe(ccdproc_bias, {'reduc_tag': 'ccdproc_bias'})

        #Recover the bias file
        shu_copy(backup_file, biasfile)

    #Generate pdf file
    idcs_print = (pr.reducDf.reduc_tag == 'biascorr')
    output_file = f'{pr.reducFolders["reduc_data"]}/bias_corrected'
    pr.generate_step_pdf(idcs_print, file_address=output_file, verbose=True, ext=0)
