from os import makedirs, path
from shutil import move
from pipeline import SpectraReduction
import src.specsiser as sr
from astropy.io import fits

# Objects and files
pattern = '.fit'
data_folder = '/home/vital/Astro-data/Observations/LzLCS_objs_ISIS'

# Load tools
obsData = sr.loadConfData('../reduction_conf.ini')

for night in obsData['data_location']['night_list']:

    print(f'\nDoing: {night}')

    # Establish night configuration
    night_conf = obsData[night]
    night_folder = f'{data_folder}/{night}'
    objs, std_stars = night_conf['objects_list'], night_conf['standard_star_list']

    pr = SpectraReduction(data_folder, obs_file=None)
    pr.declare_catalogue(night_folder, objs=objs, std_stars=std_stars)

    # Get file list
    files_list = pr.Folder_Explorer(pattern, night_folder, Sort_Output='alphabetically')

    # Create the raw data folder
    if not path.exists(pr.reducFolders['raw data']):
        makedirs(pr.reducFolders['raw data'])

    # Loop through files
    for i, file_address in enumerate(files_list):

        # Identify object
        CodeName, FileName, FileFolder = pr.Analyze_Address(file_address, verbose=False)

        # Security check to make sure we always work on the default folders
        if (FileFolder == pr.Catalogue_folder) or (FileFolder == pr.reducFolders['raw data']):

            # Read the data from the headers
            try:
                Header0 = fits.getheader(file_address, ext=0)
            except:
                Header0 = {}

            try:
                Header1 = fits.getheader(file_address, ext=1)
            except:
                Header1 = {}

            # Add variables
            for key in pr.columns_reducDf:
                values_to_load = None

                if key in Header0:
                    values_to_load = Header0[key]
                elif key in Header1:
                    values_to_load = Header1[key]

                # Change value to proper format
                if key in ['RUN']:
                    try:
                        values_to_load = int(values_to_load)
                    except:
                        values_to_load = values_to_load

                # Load value
                pr.reducDf.loc[CodeName, key] = values_to_load

            # Move the files
            if FileFolder != pr.reducFolders['raw data']:
                target_destination = f'{pr.reducFolders["raw data"]}/{FileName}'
                move(file_address, target_destination)

            # Load the name
            pr.reducDf.loc[CodeName, 'file_name'] = FileName

    # Adding location column
    pr.reducDf['file_location'] = pr.reducFolders['raw data']

    # Check rejected files
    pr.check_rejected_files(pr.reducDf, reject_file_address=night_conf['rejects_files'])

    # Remove spaces from file name
    pr.reducDf['OBJECT'] = pr.reducDf['OBJECT'].str.replace(' ', '_')
    pr.reducDf['ISIARM'] = pr.reducDf['ISIARM'].str.replace(' ', '_')

    # Check if the files is already there before overwritting
    df_address = f'{pr.reduc_RootFolder}/{pr.frame_filename}'
    pr.save_reducDF(pr.reducDf, df_address)



