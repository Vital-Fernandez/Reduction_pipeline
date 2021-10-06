class PyrafTaskConf:

    def __init__(self):

        self.task_data_dict = {}
        self.task_attributes = {}
        # self.objects_configuration = 'HII_galaxies'
        # self.script_location = '/home/vital/git/Thesis_Pipeline/Thesis_Pipeline/Spectra_Reduction/'
        # self.Standard_Stars_Folder = '/home/vital/Dropbox/Astrophysics/Tools/PyRaf/StandardStars_Calibration/'

    def load_task_configuration(self, task):

        # Reset task conf dictionary
        self.task_data_dict = {}

        task_conf_method = self.__getattribute__(task)
        task_conf_method()

        # # Load the task attributes
        # if task == 'zerocombine':
        #     self.zerocombine_task_configuration()


        return

    def standar_stars_calibrationFile(self, StdStar_code):

        calibration_dict = {}
        calibration_dict['calibration_folder'] = '/home/vital/Dropbox/Astrophysics/Tools/PyRaf/StandardStars_Calibration/'

        if StdStar_code in ['FEI34', 'F34', 'feige34', 'Feige']:
            calibration_dict['file name'] = 'feige34_stis004_nobandpass'
            calibration_dict['bandwidth'] = 15
            calibration_dict['bandsep'] = 15

        elif StdStar_code in ['HD93', 'SP1045+378']:
            calibration_dict['file name'] = 'sp1045_378_1a'
            calibration_dict['bandwidth'] = 15
            calibration_dict['bandsep'] = 15

        elif StdStar_code in ['BD29']:
            calibration_dict['file name'] = 'bd29d2091_stis_003_nobandpass'
            calibration_dict['bandwidth'] = 15
            calibration_dict['bandsep'] = 15

        elif StdStar_code in ['BD28', 'BD+28']:
            calibration_dict['file name'] = 'bd28_d4211stis004_nobandpass'
            calibration_dict['bandwidth'] = 15
            calibration_dict['bandsep'] = 15

        elif StdStar_code in ['BD33', 'BD+33']:
            calibration_dict['file name'] = 'bd33_d2642004_nobandpass'
            calibration_dict['bandwidth'] = 15
            calibration_dict['bandsep'] = 15

        elif StdStar_code in ['WOLF1346', 'wolf', 'Wolf_1346', 'WOLF1346A']:
            calibration_dict['file name'] = 'wolf_oke1974_40a'
            calibration_dict['bandwidth'] = 'INDEF'
            calibration_dict['bandsep'] = 'INDEF'

        elif StdStar_code in ['bd17', 'BD+17', 'sp2209+178', 'sp2209178', 'BD17', 'BD+17_4708', 'SP2209+178']:
            calibration_dict['file name'] = 'bd17_d4708stisnic006_nobandpass'
            calibration_dict['bandwidth'] = 15
            calibration_dict['bandsep'] = 15

        elif StdStar_code in ['g191', 'G191']:
            calibration_dict['file name'] = 'g191_b2bstisnic006_nobandpass'
            calibration_dict['bandwidth'] = 15
            calibration_dict['bandsep'] = 15

        elif StdStar_code in ['g158', 'G158']:
            calibration_dict['file name'] = 'g158_oke1990_1a'
            calibration_dict['bandwidth'] = 15
            calibration_dict['bandsep'] = 15

        elif StdStar_code in ['hd19445', 'sp0305+261', 'sp0305261', 'SP0305+261']:
            calibration_dict['file name'] = 'hd19445_oke1983_40a'
            calibration_dict['bandwidth'] = 'INDEF'
            calibration_dict['bandsep'] = 'INDEF'

        elif StdStar_code in ['hd84937', 'HD84937']:
            calibration_dict['file name'] = 'hd84937_oke1983_40a'  # 'hd84937_oke1983_40a'
            calibration_dict['bandwidth'] = 'INDEF'
            calibration_dict['bandsep'] = 'INDEF'

        return calibration_dict

    def zerocombine(self):

        self.task_data_dict['input'] = self.task_attributes['input']
        self.task_data_dict['output'] = self.task_attributes['output']
        self.task_data_dict['combine'] = self.task_attributes['combine']
        self.task_data_dict['reject'] = 'minmax'
        self.task_data_dict['ccdtype'] = '""'
        self.task_data_dict['process'] = 'no'
        self.task_data_dict['delete'] = 'no'

        return

    def ccdproc(self):

        # Load default configuraion
        self.task_data_dict['images'] = self.task_attributes['input']
        self.task_data_dict['output'] = self.task_attributes['output']  # Need a list of files here
        self.task_data_dict['ccdtype'] = '""'
        self.task_data_dict['readaxi'] = 'column'

        # for ccdproc we set all the defaults to no
        for task_attrib in ['fixpix', 'oversca', 'trim', 'zerocor', 'darkcor', 'flatcor', 'illumco', 'fringec',
                            'readcor', 'scancor']:
            if task_attrib in self.task_attributes:
                self.task_data_dict[task_attrib] = self.task_attributes[task_attrib]
            else:
                self.task_data_dict[task_attrib] = 'no'

        # Load specific files for each treatment
        for task_attrib in ['fixfile', 'biassec', 'trimsec', 'zero', 'dark', 'flat', 'illum', 'fringe']:
            if task_attrib in self.task_attributes:
                self.task_data_dict[task_attrib] = self.task_attributes[task_attrib]
            else:
                self.task_data_dict[task_attrib] = '""'

        return

    def flatcombine(self):

        self.task_data_dict['input'] = self.task_attributes['input']
        self.task_data_dict['output'] = self.task_attributes['output']  # Flat_combined
        self.task_data_dict['combine'] = self.task_attributes['combine']
        self.task_data_dict['reject'] = self.task_attributes['reject']
        self.task_data_dict['ccdtype'] = self.task_attributes['ccdtype']
        self.task_data_dict['scale'] = self.task_attributes['scale']
        self.task_data_dict['gain'] = self.task_attributes['gain']
        self.task_data_dict['snoise'] = self.task_attributes['snoise']
        # self.task_data_dict['dark'] = ''

        return

    def response(self):

        self.task_data_dict['calibrat'] = self.task_attributes['input']  # Object_flat or combined flat
        self.task_data_dict['normaliz'] = self.task_attributes['normalizing_flat']  # Combined_flat
        self.task_data_dict['response'] = self.task_attributes['output']  # Object_n_flat or n_combined_flat
        self.task_data_dict['functio'] = 'spline3'
        self.task_data_dict['order'] = self.task_attributes['order']
        self.task_data_dict['low_rej'] = '3.0'
        self.task_data_dict['high_rej'] = '3.0'
        self.task_data_dict['niterate'] = '1'
        self.task_data_dict['interact'] = 'no'
        self.task_data_dict['threshold'] = self.task_attributes['threshold']

        return

    def imcombine(self):

        self.task_data_dict['input'] = self.task_attributes['input']
        self.task_data_dict['output'] = self.task_attributes['output']
        self.task_data_dict['combine'] = self.task_attributes['combine']
        self.task_data_dict['scale'] = self.task_attributes['scale']
        self.task_data_dict['statsec'] = self.task_attributes['statsec']
        self.task_data_dict['reject'] = self.task_attributes['reject']
        self.task_data_dict['weight'] = self.task_attributes['weight']
        self.task_data_dict['gain'] = self.task_attributes['gain']
        self.task_data_dict['snoise'] = self.task_attributes['snoise']
        self.task_data_dict['sigma'] = self.task_attributes['output'].replace('.fits', '_sigma.fits')

        return

    def illumination(self):

        self.task_data_dict['images'] = self.task_attributes['input']
        self.task_data_dict['illumination'] = self.task_attributes['output']  # This is the illumFlat
        self.task_data_dict['interact'] = 'yes'
        self.task_data_dict['nbins'] = 5
        self.task_data_dict['low_rej'] = '3.0'
        self.task_data_dict['high_rej'] = '3.0'

        return

    def imcopy(self):

        self.task_data_dict['input'] = self.task_attributes['input']
        self.task_data_dict['output'] = self.task_attributes['output']
        self.task_data_dict['verbose'] = 'yes'

        return

    def imshift(self):

        self.task_data_dict['input'] = self.task_attributes['input']
        self.task_data_dict['output'] = self.task_attributes['output']
        self.task_data_dict['xshift'] = self.task_attributes['xshift']
        self.task_data_dict['yshift'] = self.task_attributes['yshift']

        return

    def identify(self):

        if self.task_attributes['color'] == 'Blue':
            coordlist_address = '/home/vital/Dropbox/Astrophysics/Tools/PyRaf/Telescope_ReductionFiles/WHT/CuNeCuAr_ArcLampBlue.csv'
        else:
            coordlist_address = '/home/vital/Dropbox/Astrophysics/Tools/PyRaf/Telescope_ReductionFiles/WHT/CuNeCuAr_ArcLampRed.csv'

        self.task_data_dict['images'] = self.task_attributes['input']
        self.task_data_dict['database'] = self.task_attributes['database']
        self.task_data_dict['section'] = 'middle column'
        self.task_data_dict['niterat'] = 1
        self.task_data_dict['fwidth'] = 7
        self.task_data_dict['coordlist'] = coordlist_address
        self.task_data_dict['match'] = 10
        self.task_data_dict['maxfeat'] = 75
        self.task_data_dict['fwidth'] = 7.5
        self.task_data_dict['cradius'] = 5
        self.task_data_dict['threshold'] = 10
        self.task_data_dict['minsep'] = 2

        return

    def reidentify(self):

        if self.task_attributes['color'] == 'Blue':
            coordlist_address = '/home/vital/Dropbox/Astrophysics/Tools/PyRaf/Telescope_ReductionFiles/WHT/CuNeCuAr_ArcLampBlue.csv'
        else:
            coordlist_address = '/home/vital/Dropbox/Astrophysics/Tools/PyRaf/Telescope_ReductionFiles/WHT/CuNeCuAr_ArcLampRed.csv'

        self.task_data_dict['referenc'] = self.task_attributes['referenc']
        self.task_data_dict['images'] = self.task_attributes['input']
        self.task_data_dict['database'] = self.task_attributes['database']
        self.task_data_dict['section'] = 'middle column'
        self.task_data_dict['coordlist'] = coordlist_address
        self.task_data_dict['interac'] = 'no'
        self.task_data_dict['overrid'] = 'yes'
        self.task_data_dict['trace'] = 'no'
        self.task_data_dict['nlost'] = 5
        self.task_data_dict['threshold'] = 10
        self.task_data_dict['match'] = 10
        self.task_data_dict['verbose'] = 'yes'

        return

    def fitcoords(self):

        self.task_data_dict['images'] = self.task_attributes['input']
        self.task_data_dict['fitname'] = self.task_attributes['fitname']
        self.task_data_dict['interac'] = 'no'

        return

    def refspectra(self):

        self.task_data_dict['images'] = self.task_attributes['input']
        self.task_data_dict['fitname'] = self.task_attributes['fitname']
        self.task_data_dict['interac'] = 'yes'

        return

    def dispcor(self):

        self.task_data_dict['input'] = self.task_attributes['input']
        self.task_data_dict['output'] = self.task_attributes['output']
        self.task_data_dict['lineari'] = self.task_attributes['lineari']
        self.task_data_dict['lineari'] = self.task_attributes['lineari']
        self.task_data_dict['flux'] = self.task_attributes['flux']
        self.task_data_dict['global'] = self.task_attributes['global']

        return

    def transform(self):

        self.task_data_dict['input'] = self.task_attributes['input']
        self.task_data_dict['output'] = self.task_attributes['output']
        self.task_data_dict['fitnames'] = self.task_attributes['fitnames']
        self.task_data_dict['flux'] = 'yes'  # is this right?

        return

    def background(self):

        self.task_data_dict['input'] = self.task_attributes['input']
        self.task_data_dict['output'] = self.task_attributes['output']
        self.task_data_dict['function'] = 'chebyshev'
        self.task_data_dict['order'] = self.task_attributes['order']
        self.task_data_dict['axis'] = self.task_attributes['axis']

        return

    def apall(self):

        self.task_data_dict['input'] = self.task_attributes['input']
        self.task_data_dict['output'] = self.task_attributes['output']
        self.task_data_dict['find'] = 'yes'
        self.task_data_dict['trace'] = self.task_attributes['trace']
        self.task_data_dict['resize'] = self.task_attributes['resize']
        self.task_data_dict['recente'] = self.task_attributes['recente']
        self.task_data_dict['referen'] = self.task_attributes['referen']
        self.task_data_dict['edit'] = self.task_attributes['edit']
        self.task_data_dict['extract'] = 'yes'
        self.task_data_dict['review'] = 'yes'
        self.task_data_dict['ylevel'] = self.task_attributes['ylevel']
        self.task_data_dict['nsum'] = self.task_attributes['nsum']
        self.task_data_dict['b_sampl'] = self.task_attributes['b_sample']
        self.task_data_dict['maxsep'] = 1000
        self.task_data_dict['b_order'] = self.task_attributes['b_order']
        self.task_data_dict['t_niter'] = 1
        self.task_data_dict['backgro'] = self.task_attributes['backgro']
        self.task_data_dict['weights'] = self.task_attributes['weights']
        self.task_data_dict['saturation'] = '32400'
        self.task_data_dict['gain'] = self.task_attributes['gain_key']
        self.task_data_dict['readnoi'] = self.task_attributes['readnois_key']
        self.task_data_dict['extras'] = self.task_attributes['extras']
        self.task_data_dict['line'] = self.task_attributes['line']
        self.task_data_dict['order'] = self.task_attributes['order']
        self.task_data_dict['interactive'] = self.task_attributes['interactive']

        return

    def standard(self):

        Observatory = 'lapalma'
        ExtinctionFileAddress = '/home/vital/Dropbox/Astrophysics/Tools/PyRaf/Telescope_ReductionFiles/WHT/a_ing_ext.dat'
        Standard_Stars_Folder = '/home/vital/Dropbox/Astrophysics/Tools/PyRaf/StandardStars_Calibration/'

        self.task_data_dict['input'] = self.task_attributes['input']
        self.task_data_dict['output'] = self.task_attributes['output']
        self.task_data_dict['samestar'] = 'yes'
        self.task_data_dict['beam_switch'] = 'no'
        self.task_data_dict['apertures'] = '""'
        self.task_data_dict['bandwidth'] = self.task_attributes['bandwidth']
        self.task_data_dict['bandsep'] = self.task_attributes['bandsep']
        self.task_data_dict['fnuzero'] = 3.68000e-20
        self.task_data_dict['extinction'] = ExtinctionFileAddress
        self.task_data_dict['caldir'] = Standard_Stars_Folder
        self.task_data_dict['observatory'] = Observatory
        self.task_data_dict['interact'] = 'yes'
        self.task_data_dict['star_name'] = self.task_attributes[
            'star_name']  # MUST NOT INCLUDE EXTENSION IN THE STAR CALIBRATION FILE, NEITHER CAPITALS!!!!
        self.task_data_dict['airmass'] = self.task_attributes['airmass']
        self.task_data_dict['exptime'] = self.task_attributes['exptime']
        self.task_data_dict['answer'] = 'yes'

        return

    def sensfunc(self):

        Observatory = 'lapalma'
        ExtinctionFileAddress = '/home/vital/Dropbox/Astrophysics/Tools/PyRaf/Telescope_ReductionFiles/WHT/a_ing_ext.dat'

        self.task_data_dict['standards'] = self.task_attributes['input']
        self.task_data_dict['sensitivity'] = self.task_attributes['output']
        self.task_data_dict['apertures'] = '""'
        self.task_data_dict['ignoreaps'] = "yes"
        self.task_data_dict['extinction'] = ExtinctionFileAddress
        self.task_data_dict['observatory'] = Observatory
        self.task_data_dict['functio'] = self.task_attributes['functio']
        self.task_data_dict['order'] = self.task_attributes['order']
        self.task_data_dict['interactive'] = 'yes'
        self.task_data_dict['graphs'] = self.task_attributes['graphs']
        self.task_data_dict['answer'] = 'yes'

        return

    def calibrate(self):

        Observatory = 'lapalma'
        ExtinctionFileAddress = '/home/vital/Dropbox/Astrophysics/Tools/PyRaf/Telescope_ReductionFiles/WHT/a_ing_ext.dat'

        self.task_data_dict['input'] = self.task_attributes['input']
        self.task_data_dict['output'] = self.task_attributes['output']
        self.task_data_dict['extinct'] = 'yes'
        self.task_data_dict['flux'] = 'yes'
        self.task_data_dict['extinction'] = ExtinctionFileAddress
        self.task_data_dict['observatory'] = Observatory
        self.task_data_dict['ignoreaps'] = 'yes'
        self.task_data_dict['sensitivity'] = self.task_attributes['senstivityCurve']
        self.task_data_dict['fnu'] = 'no'
        self.task_data_dict['airmass'] = self.task_attributes['airmass']
        self.task_data_dict['exptime'] = self.task_attributes['exptime']
        self.task_data_dict['mode'] = 'ql'

        return

    def continuum(self):

        self.task_data_dict['input'] = self.task_attributes['input']
        self.task_data_dict['output'] = self.task_attributes['output']

        return

    def sarith(self):

        self.task_data_dict['input1'] = self.task_attributes['input1']
        self.task_data_dict['op'] = self.task_attributes['op']
        self.task_data_dict['input2'] = self.task_attributes['input2']
        self.task_data_dict['output'] = self.task_attributes['output']
        self.task_data_dict['w1'] = 'INDEF'
        self.task_data_dict['w2'] = 'INDEF'
        self.task_data_dict['rebin'] = 'no'
        self.task_data_dict['verbose'] = 'yes'

        return

    def imarith(self):

        self.task_data_dict['operand1'] = self.task_attributes['operand1']
        self.task_data_dict['op'] = self.task_attributes['op']
        self.task_data_dict['operand2'] = self.task_attributes['operand2']
        self.task_data_dict['result'] = self.task_attributes['output']
        self.task_data_dict['verbose'] = 'yes'

        return

    def splot(self):

        self.task_data_dict['images'] = self.task_attributes['input']
        self.task_data_dict['xmin'] = self.task_attributes['xmin'] if 'xmin' in self.task_attributes else 'INDEF'
        self.task_data_dict['xmax'] = self.task_attributes['xmax'] if 'xmax' in self.task_attributes else 'INDEF'
        self.task_data_dict['ymax'] = self.task_attributes['ymax'] if 'ymax' in self.task_attributes else 'INDEF'

    def imstat(self):

        self.task_data_dict['images'] = self.task_attributes['input']
        self.task_data_dict['lower'] = 'INDEF'
        self.task_data_dict['upper'] = 'INDEF'

        return

    def dopcor(self):

        self.task_data_dict['input'] = self.task_attributes['input']
        self.task_data_dict['output'] = self.task_attributes['output']
        self.task_data_dict['redshift'] = self.task_attributes['redshift']
        self.task_data_dict['flux'] = self.task_attributes['flux']
        self.task_data_dict['verbose'] = 'yes'

        return