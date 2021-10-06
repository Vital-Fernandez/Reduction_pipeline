import os
import src.specsiser as sr
from pipeline import SpectraReduction


# Spectra folder
data_folder = '/home/vital/Astro-data/Observations/LzLCS_objs_ISIS'
obsData = sr.loadConfData('../reduction_conf.ini')
command_log_address = obsData['data_location']['command_log_location']
 
# Operation details
colors = ['Blue', 'Red']
task_name = 'fitcoords'
data_dict = {'reduc_tag': 'fitcoords'}

# Loop through the nights
for night in obsData['data_location']['night_list']:

    # Establish night configuration
    pr = SpectraReduction(data_folder, obs_file=None)
    night_conf = obsData[night]
    night_folder = f'{data_folder}/{night}'
    objs, std_stars = night_conf['objects_list'], night_conf['standard_star_list']

    pr.declare_catalogue(night_folder, objs=objs, std_stars=std_stars)
    run_folder = pr.reducFolders["arcs"]

    # Loop through the arms
    for arm_color in colors:

        color_label = f'{arm_color}_arm'

        idx_arcs = (pr.reducDf.reduc_tag == 'biascorr') & (pr.reducDf.ISIARM == color_label) &\
                   (pr.reducDf.frame_tag == 'arc') & pr.reducDf.valid_file

        File_Folders = pr.reducDf.loc[idx_arcs, 'file_location'].values
        File_Names = pr.reducDf.loc[idx_arcs, 'file_name'].values

        for i in range(len(File_Names)):

            # Fitcoords the arc
            task_conf = {}
            task_conf['run folder'] = run_folder
            task_conf['color'] = arm_color
            task_conf['input'] = f'{os.path.splitext(File_Names[i])[0]}'
            task_conf['fitname'] = '""'

            # Prepare iraf command
            task_conf_address = pr.prepare_iraf_command(task_name, user_conf=task_conf)

            # Log command
            pr.store_command(task_name, command_log_address)

            # Run the iraf command
            pr.launch_command(task_name, task_conf_address, verbose=True)



    # #Get object and global indeces
    # indeces_arc         = (dz.reducDf.reduc_tag == 'arc_trim') & (dz.reducDf.ISIARM == '{color} arm'.format(color = arm_color)) & (dz.reducDf.valid_file)
    # Files_Folder        = dz.reducDf.loc[indeces_arc, 'file_location'].values
    # Files_Name          = dz.reducDf.loc[indeces_arc, 'file_name'].values
    #
    # #Loop through the files
    # for i in range(len(Files_Name)):
    #
    #     input_file                          = Files_Folder[i] + Files_Name[i]
    #     input_file_noExt                    = input_file[0:input_file.rfind('.')]
    #
    #     dz.task_attributes['run folder']    = dz.reducFolders['arcs']
    #     dz.task_attributes['color']         = arm_color
    #     dz.task_attributes['input']         = input_file_noExt
    #     dz.task_attributes['fitname']       = '""'
    #
    #     #Run the task
    #     dz.run_iraf_task('fitcoords', run_externally=False)

