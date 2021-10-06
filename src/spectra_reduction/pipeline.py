import os
import numpy as np
import pandas as pd
import subprocess
from pathlib import Path

from tempfile import mkstemp
from datetime import timedelta
from task_configuration import PyrafTaskConf
from pyraf_wrapper import execute_Iraf_task
from tools import fits_plots
from astropy.io import fits

RUN_SCRIPT_ADDRESS = '/home/vital/PycharmProjects/Reduction_pipeline/isis_reduction/steps/0_RunIraf_task.py'


def lineslogFile_to_DF(lineslog_address):
    """
    This function attemps several approaches to import a lines log from a sheet or text file lines as a pandas
    dataframe
    :param lineslog_address: String with the location of the input lines log file
    :return lineslogDF: Dataframe with line labels as index and default column headers (wavelength, w1 to w6)
    """

    # Text file
    try:
        lineslogDF = pd.read_csv(lineslog_address, delim_whitespace=True, header=0, index_col=0)
    except ValueError:

        # Excel file
        try:
            lineslogDF = pd.read_excel(lineslog_address, sheet_name=0, header=0, index_col=0)
        except ValueError:
            print(f'- ERROR: Could not open lines log at: {lineslog_address}')

    return lineslogDF


class SpectraReduction(fits_plots, PyrafTaskConf):

    def __init__(self, catalogue_folder, obs_file, verbose=True):

        # Declare the support classes
        fits_plots.__init__(self)
        PyrafTaskConf.__init__(self)

        # Declare the catalogue (This should be done from a graphical intervace)
        self.reducFolders = {}
        self.Catalogue_folder = catalogue_folder
        self.observation_properties_file = obs_file
        self.columns_reducDf = None
        self.Objects_to_treat = None

        # Create text file to store the iraf commands
        self.commands_log = 'commands_log.txt'
        file_address = f'{self.Catalogue_folder}/{self.commands_log}'
        if not os.path.isfile(file_address):
            with open(file_address, 'a') as f:
                f.write('Catalogue ({cataloge_code}) reduction commands\n'.format(cataloge_code=self.Catalogue_folder))

        # Frame with the reduced data
        self.reducDf = None
        self.frame_filename = 'reduction_frame.txt'

        # Flags
        self.verbose_output = verbose

        # Fits extensions
        self.Extensions_dict = {}
        self.Extensions_dict['bias'] = '_b'
        self.Extensions_dict['flat'] = '_f'
        self.Extensions_dict['arc'] = '_w'
        self.Extensions_dict['flux'] = '_fx'

        # Organized folders # TODO all calibration files together, just original, pipeline, conf, plots
        self.default_Folders_dict = {}
        self.default_Folders_dict['bias'] = 'bias'
        self.default_Folders_dict['flat lamp'] = 'flats_lamp'
        self.default_Folders_dict['flat sky'] = 'flats_sky'
        self.default_Folders_dict['arcs'] = 'arcs'
        self.default_Folders_dict['objects'] = 'objects'
        self.default_Folders_dict['raw data'] = 'raw_fits'
        self.default_Folders_dict['telescope'] = 'conf_frames'
        self.default_Folders_dict['reduc_data'] = 'reduc_data'

    def target_validity_check(self):

        # Check if the targets are valid and they belong to the objects we want to treat
        if (self.Objects_to_treat is None) or (self.Objects_to_treat is 'All'):
            boolean_array = (self.reducDf.valid_file) & (self.reducDf.frame_tag.isin(self.Objects_to_treat))
        else:
            boolean_array = (self.reducDf.valid_file)

        return boolean_array

    def Folder_Explorer(self, myPattern, Folder, Sort_Output=None, verbose=True):

        # Define the list to store the files (should it be a self)
        FileList = []

        if type(myPattern) is not list:
            myPatternList = [myPattern]

        else:
            myPatternList = myPattern

        for Root, Dirs, Archives in os.walk(Folder):
            for Archive in Archives:
                Meets_one_Pattern = False
                for i in range(len(myPatternList)):
                    if (myPatternList[i] in Archive):

                        # Security check to make sure we are not treating dummy files
                        if "~" not in Archive:
                            Meets_one_Pattern = True

                if Meets_one_Pattern:
                    if Root.endswith("/"):
                        FileList.append(Root + Archive)
                    else:
                        FileList.append(Root + "/" + Archive)

        if Sort_Output == 'alphabetically':
            FileList = sorted(FileList)

        return FileList

    def Analyze_Address(self, FileAddress, verbose=True):

        # Distinguish the three components from the address line
        file_path = Path(FileAddress)
        fileName = file_path.name
        fileFolder = str(file_path.parent)
        codeName = file_path.stem

        if codeName.startswith('obj') or codeName.startswith('std'):
            codeName = codeName[3:codeName.find("_")]

        return codeName, fileName, fileFolder

    def get_spectra_data(self, file_address, ext=0):

        Header_0 = fits.getheader(file_address, ext=ext)
        Flux_array = fits.getdata(file_address, ext=ext)

        if "COEFF0" in Header_0:
            dw = 10.0 ** Header_0['COEFF1']  # dw = 0.862936 INDEF (Wavelength interval per pixel)
            Wmin = 10.0 ** Header_0['COEFF0']
            pixels = Header_0['NAXIS1']  # nw = 3801 number of output pixels
            Wmax = Wmin + dw * pixels
            wavelength = np.linspace(Wmin, Wmax, pixels, endpoint=False)

        elif "LTV1" in Header_0:
            StartingPix = -1 * Header_0['LTV1']  # LTV1 = -261.
            Wmin_CCD = Header_0['CRVAL1']
            dw = Header_0['CD1_1']  # dw = 0.862936 INDEF (Wavelength interval per pixel)
            pixels = Header_0['NAXIS1']  # nw = 3801 number of output pixels
            Wmin = Wmin_CCD + dw * StartingPix
            Wmax = Wmin + dw * pixels
            wavelength = np.linspace(Wmin, Wmax, pixels, endpoint=False)

        else:
            Wmin = Header_0['CRVAL1']
            dw = Header_0['CD1_1']  # dw = 0.862936 INDEF (Wavelength interval per pixel)
            pixels = Header_0['NAXIS1']  # nw = 3801 number of output pixels
            Wmax = Wmin + dw * pixels
            wavelength = np.linspace(Wmin, Wmax, pixels, endpoint=False)

        return wavelength, Flux_array, Header_0

    def declare_folde_structre(self):

        for key in self.default_Folders_dict.keys():
            self.reducFolders[key] = f'{self.reduc_RootFolder}/{self.default_Folders_dict[key]}'
            if os.path.isdir(self.reducFolders[key]) == False:
                os.makedirs(self.reducFolders[key])

        return

    def save_observation_dict(self):

        # Separate the parameters
        parameters_keys, parameters_values = self.observation_dict.keys(), self.observation_dict.values()

        # Format the entries
        for i in range(len(parameters_values)):
            if isinstance(parameters_values[i], list):
                parameters_values[i] = ' '.join(map(str, parameters_values[i]))

        # Saving the file
        observation_properties_file = self.Catalogue_folder + self.observation_properties_file_name
        file_data = np.transpose([parameters_keys, parameters_values])
        np.savetxt(fname=observation_properties_file, X=file_data, fmt='%s', delimiter='; ')

        return

    def load_observation_dict(self, observation_properties_address):

        observation_dict = {}
        observation_properties_file = np.loadtxt(observation_properties_address, dtype=str, delimiter=';', usecols=[0, 1])
        for i in range(len(observation_properties_file)):
            key = observation_properties_file[i][0]
            list_values = observation_properties_file[i][1].split()
            observation_dict[key] = list_values

        return observation_dict

    def load_reduction_dataframe(self, dataframe_folder, dataframe_name='reduction_frame.txt'):

        reduction_dataframe = None

        # Pandas frame where we store all the reduction data
        df_address = f'{dataframe_folder}/{dataframe_name}'

        if os.path.isfile(df_address):
            # reduction_dataframe = pd.read_pickle(dataframe_folder + dataframe_name)

            reduction_dataframe = lineslogFile_to_DF(df_address)

            # Special trick to update the location
            self.columns_reducDf = reduction_dataframe.index.values

            # Check if files have been added to the rejection list
            rejected_file_address = f'{self.Catalogue_folder}/rejected_files.txt'
            self.check_rejected_files(reduction_dataframe, rejected_file_address)

            # Check if files were deleted and remove then from the data frame
            to_delete = []
            addresses = reduction_dataframe.file_location.values
            names = reduction_dataframe.file_name.values
            frame_address = addresses + '/' + names

            for i in range(len(frame_address)):
                if not os.path.isfile(frame_address[i]):
                    to_delete.append(names[i])

            mask_delete = np.logical_not(np.in1d(reduction_dataframe['file_name'], to_delete))
            indeces_to_delete = reduction_dataframe.index[mask_delete]
            reduction_dataframe = reduction_dataframe.loc[indeces_to_delete]

        return reduction_dataframe

    def declare_catalogue(self, catalogue_address, objs, std_stars, data_origin='WHT', objects_type='HII galaxies',
                          verbose=True):

        # If you want to treat only one file please add it here
        self.Objects_to_treat = None

        # Declare catalogue addresses
        if catalogue_address is not None:
            self.reduc_RootFolder = catalogue_address
            self.Catalogue_folder = catalogue_address
        else:
            self.reduc_RootFolder = self.Catalogue_folder

        # Declare the main dictionary addresses
        self.declare_folde_structre()

        # # Load the observation characteristics for the reduction from
        # self.observation_dict = self.load_observation_dict(
        #     self.Catalogue_folder + self.observation_properties_file_name)

        # Load objects to treat:
        if self.Objects_to_treat is None:
            self.Objects_to_treat = list(objs) + list(std_stars)

        # Load reduction dataframe
        self.reducDf = self.load_reduction_dataframe(catalogue_address)

        # Dictionary with the keys which correspond to a given telescope header
        if data_origin == 'WHT':
            self.columns_reducDf = ['OBJECT', 'OBSTYPE', 'file_name', 'RUN', 'file_location', 'reduc_tag', 'frame_tag',
                                    'ISIARM', 'RA', 'DEC', 'UT', 'EXPTIME', 'AIRMASS', 'ISISLITW', 'CENWAVE']

            # Case this is the first time
            if self.reducDf is None:
                self.reducDf = pd.DataFrame(columns=self.columns_reducDf)

        if verbose:
            print('Reduction dataframe {catalogue_address} loaded'.format(catalogue_address=catalogue_address + 'reduction_frame.txt'))

    def check_rejected_files(self, reduction_dataframe, reject_file_address='rejected_files.txt'):

        # Set all to True
        reduction_dataframe['valid_file'] = True

        # Load list of files
        list_rejected = np.loadtxt(reject_file_address, dtype=str, usecols=[0], ndmin=1)

        # Set the column to false
        reduction_dataframe.loc[reduction_dataframe.file_name.isin(list_rejected), 'valid_file'] = False

    def generate_step_pdf(self, indeces_frame, file_address, plots_type='fits', ext=1, include_graph=False,
                          verbose=True, limits=None, sorting_pattern=['frame_tag', 'ISIARM'], obs_conf = None):

        #TODO clear open figures

        if verbose:
            print(f'\nPrinting {indeces_frame.sum()} files:')
            print(self.reducDf[indeces_frame].sort_values(['frame_tag']).file_name.values)
            print(f'{file_address}')

        # ------------Plotting pairs of fits
        if plots_type == 'fits':
            self.fits_catalogue(file_address, indeces_frame, ext=ext, columns_mean=include_graph,
                                sorting_pattern=sorting_pattern)

        # ------------Plotting flats within limits of fits
        if plots_type == 'flats':
            self.flats_plotting(file_address, indeces_frame, ext=ext, columns_mean=include_graph)

        # ------------Plotting flats within limits of fits
        if plots_type == 'flat_calibration':
            self.objects_focus(file_address, indeces_frame, ext=ext)

        # ------------Plotting extracted frames:
        if plots_type == 'extraction':
            self.extracted_frames(file_address, indeces_frame, ext=ext)

        # ------------Plotting the fast combined with notes
        if plots_type == 'cosmic_removal':
            self.cosmic_plot(file_address, indeces_frame, ext=ext, columns_mean=include_graph)

        # ------------Plotting pairs of fits
        if plots_type == 'fits_compare':
            self.fits_compare(file_address, indeces_frame, ext=ext, columns_mean=include_graph)

        # ------------Plotting the spectra
        if plots_type == 'spectra':
            self.spectra(file_address, indeces_frame, ext)

        # ------------Plotting the spectra
        if plots_type == 'masked_pixels':
            self.masked_pixels(file_address, indeces_frame, ext=ext, columns_mean=include_graph)

        # ------------Fast combine
        if plots_type == 'fast_combine':
            self.fast_combined(file_address, indeces_frame, ext=0, observation_dict=self.observation_dict,
                               reducDf=self.reducDf, colors_dict=self.frames_colors)

        # ------------Comparing the combined frames
        if plots_type == 'frame_combine':
            self.frame_combine(file_address, indeces_frame, ext=0, obs_conf=obs_conf)

        # ------------Comparing the combined frames
        if plots_type == 'frame_combine_shifted':
            self.frame_combine_shifted(file_address, indeces_frame, ext=0)

    def save_task_parameter(self, task, parameter, entry):

        import pyraf

        if task == 'response':
            #             from pyraf.iraf.noao.twodspec import response
            #             response()

            p_list = pyraf.iraf.noao.twodspec.longslit.response.getParList()

        # Getting the parameter form the task
        par = p_list[5].get()

        # Clean the format
        if par[0] == ' ':
            entry_value = par[1:].split()
        else:
            entry_value = par[0:].split()

        # Save to the observation properties file
        self.observation_dict[entry] = entry_value
        self.save_observation_dict()

    def prepare_iraf_command(self, task, user_conf, overwrite=True):

        # Establish configuration folder
        configuration_file_address = f'{user_conf["run folder"]}/{task}_{user_conf["color"]}_conf.txt'

        # Establish the task configuration
        self.task_attributes = user_conf
        self.load_task_configuration(task)

        # Save the input file list if necessary
        if 'input array' in self.task_attributes:
            output_file = f'{self.task_attributes["run folder"]}/{self.task_attributes["in_list_name"]}'
            np.savetxt(output_file, self.task_attributes['input array'], fmt='%s')

        if 'output array' in self.task_attributes:
            output_file = f'{self.task_attributes["run folder"]}/{self.task_attributes["out_list_name"]}'
            np.savetxt(output_file, self.task_attributes['output array'], fmt='%s')

        # Save configuration data
        parameter_list, value_list = list(self.task_data_dict.keys()), list(self.task_data_dict.values())
        file_data = np.transpose([parameter_list, value_list])
        np.savetxt(fname=configuration_file_address, X=file_data, fmt='%s', delimiter=';')

        # Destroy file if it already exists
        if ('output' in self.task_attributes) and overwrite:
            if os.path.isfile(self.task_attributes['output']):
                os.remove(self.task_attributes['output'])

        # Destroy files if they are in a list
        if ('output array' in self.task_attributes) and overwrite:
            for output_file in self.task_attributes['output array']:
                clean_name = output_file.replace('[1]', '')
                if os.path.isfile(clean_name):
                    os.remove(clean_name)

        # Destroy files if the output is an IRAF list (, separated) #This is a bit dirty
        if ('output' in self.task_attributes) and overwrite:
            if ',' in self.task_attributes['output']:
                output_files = self.task_attributes['output'].split(',')
                for output_file in output_files:
                    clean_name = output_file.replace('[1]', '')
                    if os.path.isfile(clean_name):
                        os.remove(clean_name)

        return configuration_file_address

    def save_reducDF(self, dataframe, file_address):

        # self.reducDf.to_pickle(self.reduc_RootFolder + self.frame_filename)
        with open(file_address, 'wb') as output_file:
            string_DF = dataframe.to_string()
            output_file.write(string_DF.encode('UTF-8'))

        return

    def create_skycor_fits(self, idx_file, output_folder=None, ext=0):

        frame_name = self.reducDf.loc[idx_file].file_name.values[0]
        frame_folder = self.reducDf.loc[idx_file].file_location.values[0]
        target_code = self.reducDf.loc[idx_file].frame_tag.values[0].replace('[', '').replace(']', '')
        color_arm = self.reducDf.loc[idx_file].ISIARM.values[0]

        # Define folder where Skycorr is run
        if output_folder == None:
            output_folder = '/home/vital/Skycorr/WHT_fittings/'

        # Get fits data (WHT case)
        print('file', frame_folder + frame_name)
        wavelength, Flux_array, Header_0 = self.get_spectra_data(frame_folder + frame_name)
        UT_start = Header_0['UTSTART'].split(':')
        UT_start_s = np.timedelta(hours=int(UT_start[0]), minutes=int(UT_start[1]),
                               seconds=float(UT_start[2])).total_seconds()

        # --Calculate the median sky
        parent_file = frame_name.replace('_eSkyCorr.fits', '.fits')
        apall_file = frame_folder + 'database/ap' + frame_folder.replace('/', '_') + parent_file.replace('.fits', '')
        # background_flux = self.compute_background_median(frame_folder + parent_file, apall_file)
        Flux_Obj = Flux_array[0][0]
        Flux_Sky = Flux_array[1][0]
        Flux_Combined = Flux_array[0][0] + Flux_array[1][0]

        #         Fig, Axis = plt.subplots(1, 1, figsize=(10, 12))
        #         Axis.plot(wavelength, Flux_Sky, label='sky')
        #         Axis.plot(wavelength, Flux_Combined, label='target')
        #         Axis.legend()
        #         plt.show()

        # Generate the header
        prihdr = fits.Header()
        prihdr['MJD-OBS'] = float(Header_0['MJD-OBS'])
        prihdr['TM-START'] = float(UT_start_s)
        prihdr['ESO TEL ALT'] = float(Header_0['LATITUDE'])
        prihdu = fits.PrimaryHDU(header=prihdr)

        # Generate the sky fits
        colA_1 = fits.Column(name='lambda', format='F', array=wavelength)
        colA_2 = fits.Column(name='flux', format='F', array=Flux_Sky)
        colsA = fits.ColDefs([colA_1, colA_2])
        tbhduA = fits.BinTableHDU.from_columns(colsA, header=prihdr)
        thdulistA = fits.HDUList([prihdu, tbhduA])
        skyfits_address = '/home/vital/Skycorr/WHT_fittings/Input_sky/' + target_code + '_background.fits'
        thdulistA.writeto(skyfits_address, clobber=True)

        # Generate the object + sky fits
        colB_1 = fits.Column(name='lambda', format='F', array=wavelength)
        colB_2 = fits.Column(name='flux', format='F', array=Flux_Combined)
        colsB = fits.ColDefs([colB_1, colB_2])
        tbhduB = fits.BinTableHDU.from_columns(colsB, header=prihdr)
        thdulistB = fits.HDUList([prihdu, tbhduB])
        targetfits_address = '/home/vital/Skycorr/WHT_fittings/Input_objects/' + target_code + '_spectrum_and_background.fits'
        thdulistB.writeto(targetfits_address, clobber=True)

        # Generate configuration file
        template = '/home/vital/Skycorr/WHT_fittings/template_parameter_file.par'
        conf_file = '/home/vital/Skycorr/WHT_fittings/{codeTarget}_{armcolor}_skycorr.in'.format(codeTarget=target_code,
                                                                                                 armcolor=color_arm.replace(
                                                                                                     ' ', '_'))
        output_file = '{codeTarget}_{armcolor}_skycorfit'.format(codeTarget=target_code,
                                                                 armcolor=color_arm.replace(' ', '_'))

        dict_modifications = {}
        dict_modifications['INPUT_OBJECT_SPECTRUM='] = targetfits_address
        dict_modifications['INPUT_SKY_SPECTRUM='] = skyfits_address
        dict_modifications['OUTPUT_NAME='] = output_file
        dict_keys = dict_modifications.keys()

        # Temporary file
        fh, abs_path = mkstemp()
        with open(abs_path, 'w') as new_file:
            with open(template) as old_file:
                for line in old_file.readlines():
                    if line.replace('\n', '') in dict_keys:
                        formatted_line = line.replace('\n', '')
                        new_line = formatted_line + dict_modifications[formatted_line] + '\n'
                    else:
                        new_line = line
                    new_file.write(line.replace(line, new_line))

        close(fh)

        # Move new file
        move(abs_path, conf_file)

        # Run skycor
        launch_command = '/home/vital/Skycorr/bin/skycorr {skycorr_script}'.format(skycorr_script=conf_file)
        p = subprocess.Popen(launch_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = p.communicate()[0]
        print
        out

        return

    def object_to_dataframe(self, list_new_file_address, data_dict, new_entry_type=False):

        if type(list_new_file_address) is str:
            list_new_file_address = [list_new_file_address]

        for i, new_file_address in enumerate(list_new_file_address):

            # Security check to not add .fits and .fit to log frame # TODO we should actually add them
            if '.fit' in new_file_address:

                # Identify frame
                CodeName, FileName, FileFolder = self.Analyze_Address(new_file_address)
                CodeName = FileName[0:FileName.find('.')]

                # Get headers
                Header0 = fits.getheader(new_file_address, ext=0)

                # Add variables
                for key in self.columns_reducDf:
                    if key in Header0:
                        entry_value = Header0[key]

                        # Remove spaces
                        if type(entry_value) is str:
                            entry_value = entry_value.replace(' ', '_')
                        self.reducDf.loc[CodeName, key] = entry_value

                # Add file location to the columns
                self.reducDf.loc[CodeName, 'file_name'] = FileName
                self.reducDf.loc[CodeName, 'file_location'] = FileFolder
                self.reducDf.loc[CodeName, 'valid_file'] = True

                # Adjusting corresponding variables
                for key in data_dict:

                    # If key not in dictionary you can create it and set it to false
                    if key not in self.reducDf.columns:
                        self.reducDf[key] = new_entry_type

                    self.reducDf.loc[CodeName, key] = data_dict[key]

                # Add the frame_tag to be the same as its corresponding number
                run_number = self.reducDf.loc[CodeName, 'RUN']
                run_firstindex = (self.reducDf.RUN == run_number)
                frame_tag_value = self.reducDf.loc[run_firstindex, 'frame_tag'].values[0]
                self.reducDf.loc[CodeName, 'frame_tag'] = frame_tag_value

                self.save_reducDF(self.reducDf, f'{self.Catalogue_folder}/reduction_frame.txt')

        return

    def launch_command(self, task_name, task_conf_address, run_script_address=None, verbose=False):

        run_script_address = RUN_SCRIPT_ADDRESS if run_script_address is None else run_script_address

        external_command = f'python {run_script_address} {task_conf_address}'

        if task_name in ['identify', 'reidentify', 'fitcoords', 'transform', 'standard', 'calibrate']:
            external_command += f' {self.task_attributes["run folder"]}'

        run_list_command = ['source /home/vital/anaconda3/bin/activate',
                            'conda activate iraf27',
                            external_command,
                            'conda deactivate']
        run_command = ' && '.join(run_list_command)
        bash_command = f"bash -c '{run_command}'"
        # https://stackoverflow.com/questions/48433478/how-to-activate-an-anaconda-environment-within-a-python-script-on-a-remote-machi

        p1 = subprocess.run(bash_command, shell=True, capture_output=True)
        # p1 = subprocess.Popen(bash_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # Print the output
        if verbose:
            print('- Stdout\n', p1.stdout.decode())

        # Check for subprocess run
        run_check = True if p1.returncode == 0 else False
        print(f'-- {task_name}: subprocess sucess' if run_check else '-- {task_name}: failed subprocess')
        if not run_check:
            print('- Stderr\n', p1.stderr.decode())

        return

    def store_command(self, task, log_address, verbose=False):

        # Generate equivalent IRAF command
        command_str = task + ''
        self.task_data_dict
        for key, item in self.task_data_dict.items():
            if isinstance(item, str):
                if ' ' in item:
                    command_str = command_str + f' {key}="{item}"'

                else:
                    command_str = command_str + ' ' + key + '=' + str(item)
            else:
                command_str = command_str + ' ' + key + '=' + str(item)


        # Load text file
        tasks_logged = np.loadtxt(log_address, dtype='str', comments='--', delimiter=' ',
                                  skiprows=1, usecols=[0], ndmin=1)

        # Append the new txt
        with open(log_address, 'a') as file_log:
            if tasks_logged.size == 0:
                file_log.write('\n--' + task + '\n')
                file_log.write(command_str + '\n')

            elif tasks_logged[-1] != task:
                file_log.write('\n--' + task + '\n')
                file_log.write(command_str + '\n')

            else:
                file_log.write(command_str + '\n')

        # Display iraf command
        if verbose:
            print(command_str)

        return

    def get_closest_time(self, df, idx, bool_cond, to_this):
        others = df.loc[bool_cond, to_this].values
        target = np.datetime64(df.loc[idx, to_this])
        idx_closest = (abs(others - target)).argmin()
        closet_value = others[idx_closest]
        return df.loc[bool_cond & (df[to_this] == closet_value)].file_name.values[0], closet_value

    def get_closest(self, df, idx, bool_cond, to_this):
        others = df.loc[bool_cond, to_this].values
        target = df.loc[idx, to_this]
        idx_closest = (abs(others - target)).argmin()
        closet_value = others[idx_closest]
        return df.loc[bool_cond & (df[to_this] == closet_value)].file_name.values[0], closet_value

    def reset_task_dict(self):

        self.task_attributes = {}

        return

    def beep_alarmn(self):

        # subprocess.call(['speech-dispatcher'])
        subprocess.call(['spd-say', '"Macarena"'])

        return