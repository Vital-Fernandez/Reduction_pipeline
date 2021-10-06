from pipeline import SpectraReduction
import src.specsiser as sr
import numpy as np
from collections import defaultdict
# Spectra folder
data_folder = '/home/vital/Astro-data/Observations/LzLCS_objs_ISIS'

# Load tools
obsData = sr.loadConfData('../reduction_conf.ini')


for night in obsData['data_location']['night_list']:

    print(f'\nDoing: {night}')

    night_conf = obsData[night]
    night_folder = f'{data_folder}/{night}'
    objs, std_stars = night_conf['objects_list'], night_conf['standard_star_list']

    pr = SpectraReduction(data_folder, obs_file=None)
    pr.declare_catalogue(night_folder, objs=objs, std_stars=std_stars)

    print('\nCATALOGUE OBJECT')
    print(pr.reducDf.OBJECT.unique())
    print('\nCATALOGUE OBSTYPE')
    print(pr.reducDf.OBSTYPE.unique())

    print('\nScientific objects')
    print(objs)
    print('\nStandard stars')
    print(std_stars)
    print()

    # Special conversions
    transformation_dic = {}
    if f'{night}_conversions' in obsData:
        cor_dict = obsData[f'{night}_conversions']
        for key, array_frames in cor_dict.items():
            obj_ref = key[0:key.find('_array')]
            transformation_dic.update(dict.fromkeys(array_frames, obj_ref))

    # Loop through the frame and assign more data
    for i, index in enumerate(pr.reducDf.index):

        ID_found = False

        object_i = pr.reducDf.loc[index].OBJECT.lower().replace(' ', '')
        obstype_i = pr.reducDf.loc[index].OBSTYPE.lower()
        file_name_i = pr.reducDf.loc[index].file_name.lower()
        run_i = pr.reducDf.loc[index].RUN

        # Identify calibration files
        for wht_TYPE in ('acq.', 'bias', 'sky', 'flat', 'arc'):
            if (wht_TYPE in object_i) or (wht_TYPE == object_i):
                ID_found = True
                pr.reducDf.loc[index, 'reduc_tag'] = wht_TYPE
                pr.reducDf.loc[index, 'frame_tag'] = wht_TYPE

        # Identify targets and standard stars
        if not ID_found:

            # Check if object
            for j, obj in enumerate(objs):
                object_code = obj.lower()
                if (object_code in object_i) or (object_code == object_i):
                    ID_found = True
                    pr.reducDf.loc[index, 'reduc_tag'] = 'target'
                    pr.reducDf.loc[index, 'frame_tag'] = obj

            # Check if object
            for k, std in enumerate(std_stars):
                object_code = std.lower()
                if (object_code in object_i) or (object_code == object_i):
                    ID_found = True
                    pr.reducDf.loc[index, 'reduc_tag'] = 'target'
                    pr.reducDf.loc[index, 'frame_tag'] = std

            # Check if the lapture is
            if run_i in transformation_dic:
                pr.reducDf.loc[index, 'frame_tag'] = transformation_dic[run_i]

        # Failure for identification
        if not ID_found:
            pr.reducDf.loc[index, 'reduc_tag'] = 'not_found'
            pr.reducDf.loc[index, 'frame_tag'] = 'not_found'

        message = f'{pr.reducDf.iloc[i].RUN} ({obstype_i}) : {pr.reducDf.iloc[i].OBJECT} \t\t\t->' \
                  f' {pr.reducDf.iloc[i].frame_tag} ({pr.reducDf.iloc[i].reduc_tag}) \t ({pr.reducDf.iloc[i].CENWAVE}) ({pr.reducDf.iloc[i].ISIARM})'

        if not pr.reducDf.iloc[i].valid_file:
            message += f' (NOT VALID)'
        print(message)

    # Check if the files is already there before overwritting
    df_address = f'{pr.reduc_RootFolder}/{pr.frame_filename}'
    pr.save_reducDF(pr.reducDf, df_address)
