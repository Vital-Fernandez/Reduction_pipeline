import numpy as np
from os import remove, close
from shutil import move
from pandas import Timedelta, to_datetime
from astropy import units as u, coordinates as coord, io
from pylatex import Document, Package, Figure, NoEscape, Tabular
from tempfile import mkstemp
from pipeline import SpectraReduction
import src.specsiser as sr
from src.specsiser.data_printing import PdfPrinter

def get_closest(df, idx, bool_cond, to_this):
    others = df.loc[bool_cond, to_this].values
    target = df.loc[idx, to_this].values[0]
    idx_closest = (np.abs(others - target)).argmin()
    closet_value = others[idx_closest]
    return df.loc[bool_cond & (df[to_this] == closet_value)].index.values[0], closet_value

Time_key = 'UT'
Exp_time_key = 'EXPTIME'
Airmass_key = 'AIRMASS'
Position_keys = ['RA', 'DEC']

# Spectra folder
data_folder = '/home/vital/Astro-data/Observations/LzLCS_objs_ISIS'
conf_file = '../reduction_conf.ini'
obsData = sr.loadConfData(conf_file)
command_log_address = obsData['data_location']['command_log_location']

# Operation details
colors = ['Blue', 'Red']
Time_key = 'UT'
Exp_time_key = 'EXPTIME'
Airmass_key = 'AIRMASS'
Position_keys = ['RA', 'DEC']


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

    len_df = len(pr.reducDf.index)

    pr.reducDf['RA'] = pr.reducDf['RA'].str.replace('_', '')
    pr.reducDf['DEC'] = pr.reducDf['DEC'].str.replace('_', '')

    pr.reducDf['UT_time'] = to_datetime(pr.reducDf['UT'])
    pr.reducDf['UT_time'] = pr.reducDf['UT_time'] + (pr.reducDf.UT_time - pr.reducDf.iloc[0].UT_time < Timedelta(0)) * Timedelta(1, 'D')

    idcs_targets = (pr.reducDf.reduc_tag == 'trace_spec') & (pr.reducDf.valid_file)
    df_targets = pr.reducDf.loc[idcs_targets].copy()
    df_targets['CENWAVE'] = df_targets['CENWAVE'].astype(float)
    df_targets['AIRMASS'] = df_targets['AIRMASS'].astype(float)

    pdf = PdfPrinter()
    pdf.create_pdfDoc(pdf_type='table')
    pdf.pdf_insert_table(['Object', 'Time', 'Airmass', 'Arcseconds'])

    for obj in objs:

        for arm_color in colors:

            color_label = f'{arm_color}_arm'

            # Objects combined and extracted spectrum
            idx_obj = (df_targets.frame_tag == obj) &\
                      (df_targets.ISIARM == color_label)

            if idx_obj.sum() > 0:
                input_file = f'{run_folder}/{df_targets.loc[idx_obj].file_name.values[0]}'
                objCenWave = df_targets.loc[idx_obj].CENWAVE.values[0]

                # Calculate number of obj expositions
                idx_obj_exps = (pr.reducDf.frame_tag == obj) & \
                               (pr.reducDf.reduc_tag == 'cr_corr') &\
                               (pr.reducDf.ISIARM == color_label) &\
                               (pr.reducDf.valid_file)
                n_exp = idx_obj_exps.sum()

                # Candiadate stars for calibration
                idxs_stars = (df_targets.frame_tag.isin(std_stars)) &\
                             (df_targets.ISIARM == color_label) & \
                             (df_targets.CENWAVE.apply(np.isclose, b=objCenWave, atol=4))
                star_frames = df_targets.loc[idxs_stars].frame_tag.values

                # Get the target parameters
                Exposition_time = io.fits.getval(input_file, 'EXPTIME', 0)
                Airmass = io.fits.getval(input_file, 'AIRMASS', 0)
                first_obs_time = pr.reducDf.loc[idx_obj_exps].UT_time.values[0]
                ra, dec = df_targets[idx_obj].RA.values[0], df_targets[idx_obj].DEC.values[0]
                RA_obj = coord.Angle(ra, unit=u.hourangle)
                DEC_obj = coord.Angle(dec, unit=u.degree)

                # Add half the time of the total observation
                time_interval = np.timedelta64(int(n_exp * Exposition_time/2), 's')
                df_targets.loc[idx_obj, 'UT_time'] = df_targets.loc[idx_obj, 'UT_time'].values[0] + time_interval

                # Calculating angular separation
                RA_stars = coord.Angle(df_targets[idxs_stars].RA.values, unit=u.hourangle)
                DEC_stars = coord.Angle(df_targets[idxs_stars].RA.values, unit=u.degree)

                Delta_RA = RA_stars.degree - RA_obj.degree
                Delta_Dec = DEC_stars.degree - DEC_obj.degree
                cos_Dec = np.cos(DEC_obj.radian)
                Delta_Theta = np.sqrt((Delta_RA * cos_Dec)**2 + (Delta_Dec**2))

                # Get closests values
                idx_star_by_airmass, airmass_star = get_closest(df_targets, idx_obj, idxs_stars, 'AIRMASS')
                idx_star_by_time, star_time = get_closest(df_targets, idx_obj, idxs_stars, 'UT_time')
                star_by_sep = df_targets.loc[idxs_stars, 'frame_tag'].values[np.argmin(Delta_Theta)]
                star_sep = Delta_Theta[np.argmin(Delta_Theta)]

                print(f'{obj} {color_label} ({objCenWave}): {df_targets.loc[idx_star_by_time].frame_tag} ({df_targets.loc[idx_star_by_time].CENWAVE}) by time, {df_targets.loc[idx_star_by_airmass].frame_tag} by proximity')
                tex_row_i = [f' ({objCenWave}) {obj} {color_label}',
                             f'{df_targets.loc[idx_star_by_time].frame_tag} ({df_targets.loc[idx_star_by_time].CENWAVE})',
                             f'{df_targets.loc[idx_star_by_airmass].frame_tag} ({df_targets.loc[idx_star_by_airmass].CENWAVE})',
                             f'{df_targets.loc[idxs_stars, "frame_tag"].values[np.argmin(Delta_Theta)]} ({df_targets.loc[idxs_stars, "CENWAVE"].values[np.argmin(Delta_Theta)]})']
                for i, item in enumerate(tex_row_i):
                    tex_row_i[i] = item.replace('_', '\_')
                pdf.addTableRow(tex_row_i)

                night_conf[f'{obj}_{arm_color}_std'] = df_targets.loc[idx_star_by_time].frame_tag
                sr.safeConfData(conf_file, night_conf, section_name=night)

        pdf.table.add_hline()

    table_address = f'{pr.reducFolders["reduc_data"]}/stdStar_proximity'
    pdf.generate_pdf(table_address, clean_tex=True)

# #Create the doc
# doc = Document(dz.reducFolders['reduc_data'] + 'table_matching')
# doc.packages.append(Package('geometry', options=['left=1cm', 'right=1cm', 'top=1cm', 'bottom=1cm']))
# doc.packages.append(Package('preview', options=['active', 'tightpage',]))
# doc.packages.append(Package('color', options=['usenames', 'dvipsnames',]))
#
# #Table pre-commands
# doc.append(NoEscape(r'\begin{table*}[h]'))
# doc.append(NoEscape(r'\begin{preview}'))
# doc.append(NoEscape(r'\centering'))
# doc.append(NoEscape(r'\pagenumbering{gobble}'))
#
# #Table header
# # table_format = 'l' + 'c'.join([' c' for s in range(len(dz.reducDf.loc[idxs_stars, 'frame_tag'].values) - 1)])
# table_format = 'lc'
#
# #Create the table
# with doc.create(Tabular(table_format)) as table:
#
#     #table.add_row(['Science targets'] + list(dz.reducDf.loc[idxs_stars, 'frame_tag'].values), escape=False)
#     table.add_row(['Science targets', 'Stars proximity by\n Airmass, Time, angle'], escape=False)
#
#     for i in range(len(Science_targets)):
#
#         target_object = Science_targets[i]
#
#         idx_target_reference    = (dz.reducDf.frame_tag == target_object) & (dz.reducDf.reduc_tag == 'extracted_spectrum') & (dz.reducDf.ISIARM == 'Blue arm') & (dz.target_validity_check())
#         idx_target_observations = (dz.reducDf.frame_tag == target_object) & (dz.reducDf.reduc_tag == target_object) & (dz.reducDf.ISIARM == 'Blue arm')
#         target_address          = dz.reducDf[idx_target_reference].file_location.values[0] + dz.reducDf[idx_target_reference].file_name.values[0]
#
#
#         #Get the target parameters
#         Exposition_time         = getval(target_address, 'EXPTIME', 0)
#         Airmass                 = getval(target_address, Airmass_key, 0)
#         first_obs_time          = dz.reducDf[idx_target_reference].UT_time.values[0]
#         RA_obj, DEC_obj         = coord.Angle(dz.reducDf[idx_target_reference].RA.values[0], unit=u.hourangle),  coord.Angle(dz.reducDf[idx_target_reference].DEC.values[0], unit = u.degree)
#
#         #Add half the time of the total observation
#         time_interval = timedelta64(int(idx_target_observations.sum() * Exposition_time / 2),'s')
#         dz.reducDf.loc[idx_target_reference, 'UT_time'] = dz.reducDf.loc[idx_target_reference, 'UT_time'].values[0] + time_interval
#
#         #Calculating angular separation
#         RA_stars                = coord.Angle(dz.reducDf[idxs_stars].RA.values, unit=u.hourangle)
#         DEC_stars               = coord.Angle(dz.reducDf[idxs_stars].RA.values, unit=u.degree)
#
#         Delta_RA                = RA_stars.degree - RA_obj.degree
#         Delta_Dec               = DEC_stars.degree - DEC_obj.degree
#         cos_Dec                 = cos(DEC_obj.radian)
#         Delta_Theta             = sqrt((Delta_RA * cos_Dec)**2 + (Delta_Dec)**2)
#
#         #Get closests values
#         star_by_airmass, airmass_star   = get_closest(dz.reducDf, idx_target_reference, idxs_stars, Airmass_key)
#         star_by_time, star_time         = get_closest(dz.reducDf, idx_target_reference, idxs_stars, 'UT_time')
#         star_by_sep, star_sep           = dz.reducDf.loc[idxs_stars, 'frame_tag'].values[argmin(Delta_Theta)], Delta_Theta[argmin(Delta_Theta)]
#
#         #Store the star by time
#         dz.observation_dict['{target_code}_calibration_star_blue'.format(target_code = target_object)] = [star_by_time]
#         dz.observation_dict['{target_code}_calibration_star_red'.format(target_code = target_object)] = [star_by_time]
#         replace_observation_data(dz.Catalogue_folder + dz.observation_properties_file_name, dz.observation_dict)
#
#         #Add the row
#         table.add_row([target_object.replace('[','').replace(']',''), '{by_airmass} {by_time} {by_sep}'.format(by_airmass = star_by_airmass, by_time = star_by_time, by_sep=star_by_sep)], escape=False)
#
#     #Adding a double line for different section
#     table.add_hline()
#
# #Close the preview
# doc.append(NoEscape(r'\end{preview}'))
# doc.append(NoEscape(r'\end{table*}'))
#
# #Generate the document
# doc.generate_pdf(clean=True)
#
# print 'Data treated'
    
    
