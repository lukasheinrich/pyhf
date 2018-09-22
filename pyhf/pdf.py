import copy
import logging
log = logging.getLogger(__name__)

from . import get_backend
from . import exceptions
from . import modifiers
from . import utils


class _ModelConfig(object):
    @classmethod
    def from_spec(cls,spec,poiname = 'mu', qualify_names = False):
        channels = []
        samples = []
        modifiers = []
        # hacky, need to keep track in which order we added the constraints
        # so that we can generate correctly-ordered data
        instance = cls()
        for channel in spec['channels']:
            channels.append(channel['name'])
            for sample in channel['samples']:
                samples.append(sample['name'])
                # we need to bookkeep a list of modifiers by type so that we
                # can loop over them on a type-by-type basis
                # types like histosys, normsys, etc...
                sample['modifiers_by_type'] = {}
                for modifier_def in sample['modifiers']:
                    if qualify_names:
                        fullname = '{}/{}'.format(modifier_def['type'],modifier_def['name'])
                        if modifier_def['name'] == poiname:
                            poiname = fullname
                        modifier_def['name'] = fullname
                    modifier = instance.add_or_get_modifier(channel, sample, modifier_def)
                    modifier.add_sample(channel, sample, modifier_def)
                    modifiers.append(modifier_def['name'])
                    sample['modifiers_by_type'].setdefault(modifier_def['type'],[]).append(modifier_def['name'])
        instance.channels = list(set(channels))
        instance.samples = list(set(samples))
        instance.modifiers = list(set(modifiers))
        instance.set_poi(poiname)
        return instance

    def __init__(self):
        # set up all other bookkeeping variables
        self.poi_index = None
        self.par_map = {}
        self.par_order = []
        self.auxdata = []
        self.auxdata_order = []
        self.next_index = 0

    def suggested_init(self):
        init = []
        for name in self.par_order:
            init = init + self.par_map[name]['modifier'].suggested_init
        return init

    def suggested_bounds(self):
        bounds = []
        for name in self.par_order:
            bounds = bounds + self.par_map[name]['modifier'].suggested_bounds
        return bounds

    def par_slice(self, name):
        return self.par_map[name]['slice']

    def modifier(self, name):
        return self.par_map[name]['modifier']

    def set_poi(self,name):
        if name not in self.modifiers:
            raise exceptions.InvalidModel("The paramter of interest '{0:s}' cannot be fit as it is not declared in the model specification.".format(name))
        s = self.par_slice(name)
        assert s.stop-s.start == 1
        self.poi_index = s.start

    def add_or_get_modifier(self, channel, sample, modifier_def):
        """
        Add a new modifier if it does not exist and return it
        or get the existing modifier and return it

        Args:
            channel: current channel object (e.g. from spec)
            sample: current sample object (e.g. from spec)
            modifier_def: current modifier definitions (e.g. from spec)

        Returns:
            modifier object

        """
        # get modifier class associated with modifier type
        try:
            modifier_cls = modifiers.registry[modifier_def['type']]
        except KeyError:
            log.exception('Modifier type not implemented yet (processing {0:s}). Current modifier types: {1}'.format(modifier_def['type'], modifiers.registry.keys()))
            raise exceptions.InvalidModifier()

        # if modifier is shared, check if it already exists and use it
        if modifier_cls.is_shared and modifier_def['name'] in self.par_map:
            log.info('using existing shared, {0:s}constrained modifier (name={1:s}, type={2:s})'.format('' if modifier_cls.is_constrained else 'un', modifier_def['name'], modifier_cls.__name__))
            modifier = self.par_map[modifier_def['name']]['modifier']
            if not type(modifier).__name__ == modifier_def['type']:
                raise exceptions.InvalidNameReuse('existing modifier is found, but it is of wrong type {} (instead of {}). Use unique modifier names or use qualify_names=True when constructing the pdf.'.format(type(modifier).__name__, modifier_def['type']))
            return modifier

        # did not return, so create new modifier and return it
        modifier = modifier_cls(sample['data'], modifier_def['data'])
        npars = modifier.n_parameters

        log.info('adding modifier %s (%s new nuisance parameters)', modifier_def['name'], npars)
        sl = slice(self.next_index, self.next_index + npars)
        self.next_index = self.next_index + npars
        self.par_order.append(modifier_def['name'])
        self.par_map[modifier_def['name']] = {
            'slice': sl,
            'modifier': modifier
        }
        if modifier.is_constrained:
            self.auxdata += self.modifier(modifier_def['name']).auxdata
            self.auxdata_order.append(modifier_def['name'])
        return modifier

def finalize_stats(modifier):
    tensorlib, _ = get_backend()
    inquad = tensorlib.sqrt(tensorlib.sum(tensorlib.power(tensorlib.astensor(modifier.uncertainties),2), axis=0))
    totals = tensorlib.sum(modifier.nominal_counts,axis=0)
    return tensorlib.divide(inquad,totals)

class Model(object):
    def __init__(self, spec, **config_kwargs):
        self.spec = copy.deepcopy(spec) #may get modified by config
        self.schema = config_kwargs.get('schema', utils.get_default_schema())
        # run jsonschema validation of input specification against the (provided) schema
        log.info("Validating spec against schema: {0:s}".format(self.schema))
        utils.validate(self.spec, self.schema)
        # build up our representation of the specification
        self.config = _ModelConfig.from_spec(self.spec,**config_kwargs)

        _allmods = []
        _allsamples = []
        _allchannels = []
        _allmods = []
        channel_nbins = {}

        for c in self.spec['channels']:
            _allchannels.append(c['name'])
            for s in c['samples']:
                channel_nbins[c['name']] = len(s['data'])
                _allsamples.append(s['name'])
                for mod in s['modifiers']:
                    _allmods.append((mod['name'],mod['type']))
        _allmods = list(set(_allmods))
        _allsamples = list(set(_allsamples))
        _allchannels = list(set(_allchannels))
        self.do_samples  = _allsamples[:]
        self.do_channels = _allchannels[:]
        self.do_mods = _allmods[:]
        self.channel_nbins = channel_nbins

        self.finalized_stats = {k:finalize_stats(self.config.modifier(k)) for k,v in self.config.par_map.items() if 'staterror' in k}
        self._make_mega()
        self.prep()

    def _make_mega(self):
        helper = {}
        for c in self.spec['channels']:
            for s in c['samples']:
                helper.setdefault(c['name'],{})[s['name']] = (c,s)

        mega_mods = {}
        import copy
        for m,mtype in self.do_mods:
            for s in self.do_samples:
                modspec = {'type': mtype, 'name': m}
                if mtype == 'histosys':
                    modspec.setdefault('data',{})['hi_data'] = []
                    modspec.setdefault('data',{})['lo_data'] = []
                    modspec.setdefault('data',{})['mask'] = []
                elif mtype == 'normsys':
                    modspec.setdefault('data',{})['hi'] = []
                    modspec.setdefault('data',{})['lo'] = []
                    modspec.setdefault('data',{})['mask'] = []
                elif mtype == 'normfactor':
                    modspec.setdefault('data',{})['mask'] = []
                elif mtype == 'shapefactor':
                    modspec.setdefault('data',{})['mask'] = []
                elif mtype == 'shapesys':
                    modspec.setdefault('data',{})['mask'] = []
                elif mtype == 'staterror':
                    modspec.setdefault('data',{})['uncrt'] = []
                    modspec.setdefault('data',{})['mask']  = []
                mega_mods.setdefault(s,{})[m] = copy.deepcopy(modspec)
                
        mega_samples = {}
        for s in self.do_samples:
            mega_nom = []
            for c in self.do_channels:
                defined_samp = helper.get(c,{}).get(s)
                defined_samp = None if not defined_samp else defined_samp[1]
                nom = defined_samp['data'] if defined_samp else [0.0]*self.channel_nbins[c]
                mega_nom += nom
                defined_mods = {x['name']:x for x in defined_samp['modifiers']} if defined_samp else {}
                for m,mtype in self.do_mods:
                    thismod = defined_mods.get(m)
                    if mtype == 'histosys':
                        lo_data = thismod['data']['lo_data'] if thismod else nom
                        hi_data = thismod['data']['hi_data'] if thismod else nom
                        maskval = True if thismod else False
                        mega_mods[s][m]['data']['lo_data'] += lo_data
                        mega_mods[s][m]['data']['hi_data'] += hi_data
                        mega_mods[s][m]['data']['mask']    += [maskval]*len(nom) #broadcasting
                        pass
                    elif mtype == 'normsys':
                        maskval = True if thismod else False
                        lo_factor = thismod['data']['lo'] if thismod else 1.0
                        hi_factor = thismod['data']['hi'] if thismod else 1.0
                        mega_mods[s][m]['data']['lo']   += [lo_factor]*len(nom) #broadcasting
                        mega_mods[s][m]['data']['hi']   += [hi_factor]*len(nom)
                        mega_mods[s][m]['data']['mask'] += [maskval]  *len(nom) #broadcasting
                    elif mtype == 'normfactor':
                        maskval = True if thismod else False
                        mega_mods[s][m]['data']['mask'] += [maskval]*len(nom) #broadcasting
                    elif mtype == 'staterror':
                        uncrt = thismod['data'] if thismod else [0.0]*len(nom)
                        maskval = [True if thismod else False]*len(nom)
                        mega_mods[s][m]['data']['mask']  += maskval
                        mega_mods[s][m]['data']['uncrt'] += uncrt
                    elif mtype == 'shapefactor':
                        maskval = True if thismod else False
                        mega_mods[s][m]['data']['mask'] += [maskval]*len(nom) #broadcasting
                    else:
                        raise RuntimeError
            sample_dict = {
                'name': 'mega_{}'.format(s),
                'nom': mega_nom,
                'modifiers': list(mega_mods[s].values())
            }
            mega_samples[s] = sample_dict
        self.mega_samples = mega_samples
        self.mega_mods    = mega_mods

    def prep(self):
        import numpy as np
        self.normsys_histoset = np.asarray([
            [
                [
                    self.mega_mods[s][m]['data']['lo'],
                    [1.]*len(self.mega_samples[s]['nom']),
                    self.mega_mods[s][m]['data']['hi'],
                ]
                for s in self.do_samples
            ] for m,mtype in self.do_mods if mtype == 'normsys' 
        ])
        self.normsys_mask = np.asarray([
            [
                [
                    self.mega_mods[s][m]['data']['mask'],
                ]
                for s in self.do_samples
            ] for m,mtype in self.do_mods if mtype == 'normsys' 
        ])
        self.normsys_default = np.ones(self.normsys_mask.shape)


        self.histosys_histoset = np.asarray([
            [
                [
                    self.mega_mods[s][m]['data']['lo_data'],
                    self.mega_samples[s]['nom'],
                    self.mega_mods[s][m]['data']['hi_data'],
                ]
                for s in self.do_samples
            ] for m,mtype in self.do_mods if mtype == 'histosys' 
        ])
        self.histosys_mask = np.asarray([
            [
                [
                    self.mega_mods[s][m]['data']['mask'],
                ]
                for s in self.do_samples
            ] for m,mtype in self.do_mods if mtype == 'histosys' 
        ])
        self.histosys_default = np.zeros(self.histosys_mask.shape)


        self.normfactor_mask = np.asarray([
            [
                [
                    self.mega_mods[s][m]['data']['mask'],
                ]
                for s in self.do_samples
            ] for m,mtype in self.do_mods if mtype == 'normfactor' 
        ])
        self.normfactor_default = np.ones(self.normfactor_mask.shape)
        self.staterror_mask = np.asarray([
            [
                [
                    self.mega_mods[s][m]['data']['mask'],
                ]
                for s in self.do_samples
            ] for m,mtype in self.do_mods if mtype == 'staterror' 
        ])
        self.staterror_default = np.ones(self.staterror_mask.shape)


    def _mtype_results(self,mtype,pars):
        """
        This method implements the computation of the modifier's application
        for a given modifier type, for each channel and sample within that
        type.

        In a follow up PR it will be further refactored to reverse the order of
        the three loops, such that the outer-most loop is over modifiers (which
        is the structure we are aiming for in #251)

        This will include additional code like

            if mtype in self.combined_mods.keys():
                return self.combined_mods[mtype].apply(pars)

        before the loops. This returns a bookkeeping dictionary of the following structure

            _mtype_results_dict == {
                channel1: {
                    sample1: [
                        mod1.apply(),
                        mod2.apply(),
                        ...
                    ],
                    sample2: [
                        mod1.apply(),
                        mod3.apply(),
                        mod5.apply(),
                        ...
                    ]
                },
                channel2: {
                    sample2: [
                        mod2.apply(),
                        mod3.apply(),
                        mod4.apply()
                    ],
                    ...
                },
                ...
            }
        """
        mtype_results = {}
        for channel in self.spec['channels']:
            for sample in channel['samples']:
                for mname in sample['modifiers_by_type'].get(mtype,[]):
                    modifier = self.config.modifier(mname)
                    modpars  = pars[self.config.par_slice(mname)]
                    mtype_results.setdefault(channel['name'],
                            {}).setdefault(sample['name'],
                            []).append(
                                modifier.apply(channel, sample, modpars)
                            )
        return mtype_results

    def expected_sample(self, channel, sample, pars):
        tensorlib, _ = get_backend()
        """
        Public API only, not efficient or fast. We compute all modification for
        all samples in this method even though we are only interested in the
        modifications in a single sample.

        Alternatively the _all_modifications() could take a list of
        channel/samples for which it should compute modificiations.
        """
        all_modifications = self._all_modifications(pars)
        return self._expected_sample(
            tensorlib.astensor(sample['data']), #nominal
            *all_modifications[channel['name']][sample['name']] #mods
        )

    def _expected_sample(self, nominal, factors, deltas):
        tensorlib, _ = get_backend()

        #the base value for each bin is either the nominal with deltas applied
        #if there were any otherwise just the nominal
        if len(deltas):
            #stack all bin-wise shifts in the yield value (delta) from the modifiers
            #on top of each other and sum through the first axis
            #will give us a overall shift to apply to the nominal histo
            all_deltas = tensorlib.sum(tensorlib.stack(deltas), axis=0)

            #stack nominal and deltas and sum through first axis again
            #to arrive at yield value after deltas (but before factor mods)
            nominal_and_deltas  = tensorlib.stack([nominal,all_deltas])
            nominal_plus_deltas = tensorlib.sum(nominal_and_deltas,axis=0)
            basefactor = [nominal_plus_deltas]
        else:
            basefactor = [nominal]

        factors += basefactor

        #multiplicative modifiers are either a single float that should be broadcast
        #to all bins or a list of floats (one for each bin of the histogram)
        binwise_factors = tensorlib.simple_broadcast(*factors)

        #now we arrange all factors on top of each other so that for each bin we
        #have all multiplicative factors
        stacked_factors_binwise = tensorlib.stack(binwise_factors)

        #binwise multiply all multiplicative factors such that we arrive
        #at a single number for each bin
        total_factors = tensorlib.product(stacked_factors_binwise, axis=0)
        return total_factors

    def _all_modifications(self, pars):
        """
        This function implements the calculation of all modifications by
        looping over all possible modifications and calling _mtype_results()
        for each one. The results are accumulated in a nested dict-like
        structure to keep track of factors/deltas that is then used by
        expected_actualdata()/expected_sample().

        The idea is that we compute all bin-values at once.. each bin is a
        product of various factors, but sum are per-channel the other
        per-channel

            b1 = shapesys_1   |      shapef_1   |
            b2 = shapesys_2   |      shapef_2   |
            ...             normfac1    ..     normfac2
            ...             (broad)     ..     (broad)
            bn = shapesys_n   |      shapef_1   |

        this can be achieved by `numpy`'s `broadcast_arrays` and `np.product`.
        The broadcast expands the scalars or one-length arrays to an array
        which we can then uniformly multiply

            >>> import numpy as np
            >>> np.broadcast_arrays([2],[3,4,5],[6],[7,8,9])
            [array([2, 2, 2]), array([3, 4, 5]), array([6, 6, 6]), array([7, 8, 9])]
            >>> ## also
            >>> np.broadcast_arrays(2,[3,4,5],6,[7,8,9])
            [array([2, 2, 2]), array([3, 4, 5]), array([6, 6, 6]), array([7, 8, 9])]
            >>> ## also
            >>> factors = [2,[3,4,5],6,[7,8,9]]
            >>> np.broadcast_arrays(*factors)
            [array([2, 2, 2]), array([3, 4, 5]), array([6, 6, 6]), array([7, 8, 9])]

        So that something like

            >>> import numpy as np
            >>> np.product(np.broadcast_arrays([2],[3,4,5],[6],[7,8,9]),axis=0)
            array([252, 384, 540])

        which is just `[ 2*3*6*7, 2*4*6*8, 2*5*6*9]`.

        Notice how some factors (for fixed channel c and sample s) depend on
        bin b and some don't (Eq 6 `CERN-OPEN-2012-016`_). The broadcasting lets
        you scale all bins the same way, such as when you have a ttbar
        normalization factor that scales all bins.

        Shape === affects each bin separately
        Non-shape === affects all bins the same way (just changes normalization, keeps shape the same)

        .. _`CERN-OPEN-2012-016`: https://cds.cern.ch/record/1456844?ln=en
        """
        # for each sample the expected ocunts are
        # counts = (multiplicative factors) * (normsys multiplier) * (histsys delta + nominal hist)
        #        = f1*f2*f3*f4* nomsysfactor(nom_sys_alphas) * hist(hist_addition(histosys_alphas) + nomdata)
        # nomsysfactor(nom_sys_alphas)   = 1 + sum(interp(1, anchors[i][0], anchors[i][0], val=alpha)  for i in range(nom_sys_alphas))
        # hist_addition(histosys_alphas) = sum(interp(nombin, anchors[i][0],
        # anchors[i][0], val=alpha) for i in range(histosys_alphas))
        #
        # Formula:
        #     \nu_{cb} (\phi_p, \alpha_p, \gamma_p) = \lambda_{cs} \gamma_{cb} \phi_{cs}(\alpha) \eta_{cs}(\alpha) \sigma_{csb}(\alpha)
        # \gamma == statsys, shapefactor
        # \phi == normfactor, overallsys
        # \sigma == histosysdelta + nominal

        # first, collect the factors from all modifiers

        all_modifications = {}
        mods_and_ops = [(x,getattr(modifiers,x).op_code) for x in modifiers.__all__]
        factor_mods = [x[0] for x in mods_and_ops if x[1]=='multiplication']
        delta_mods  = [x[0] for x in mods_and_ops if x[1]=='addition']

        all_results = {}
        for mtype in factor_mods + delta_mods:
            all_results[mtype] = self._mtype_results(mtype,pars)

        for channel in self.spec['channels']:
            for sample in channel['samples']:
                #concatenate list of lists using sum() and an initial value of []
                #to get a list of all prefactors that should be multiplied to the
                #base histogram (the deltas below + the nominal)
                factors = sum([
                    all_results.get(x,{}).get(channel['name'],{}).get(sample['name'],[])
                    for x in factor_mods
                ],[])

                #concatenate list of lists using sum() and an initial value of []
                #to get a list of all deltas that should be addded to the
                #nominal values
                deltas  = sum([
                    all_results.get(x,{}).get(channel['name'],{}).get(sample['name'],[])
                    for x in delta_mods
                ],[])

                all_modifications.setdefault(
                    channel['name'],{})[
                    sample['name']
                ] = (factors, deltas)
        return all_modifications

    def expected_auxdata(self, pars):
        # probably more correctly this should be the expectation value of the constraint_pdf
        # or for the constraints we are using (single par constraings with mean == mode), we can
        # just return the alphas

        tensorlib, _ = get_backend()
        # order matters! because we generated auxdata in a certain order
        auxdata = None
        for modname in self.config.auxdata_order:
            thisaux = self.config.modifier(modname).expected_data(
                pars[self.config.par_slice(modname)])
            tocat = [thisaux] if auxdata is None else [auxdata, thisaux]
            auxdata = tensorlib.concatenate(tocat)
        return auxdata

    def expected_actualdata(self,pars):
        import numpy as np

        histosys_alphaset = np.asarray([
            pars[self.config.par_slice(m)] for m,mtype in self.do_mods if mtype == 'histosys'
        ])
        normsys_alphaset = np.asarray([
            pars[self.config.par_slice(m)] for m,mtype in self.do_mods if mtype == 'normsys'
        ])
        from .interpolate import _hfinterp_code1,_hfinterp_code0
        results_norm   = _hfinterp_code1(self.normsys_histoset,normsys_alphaset)
        results_norm   = np.where(self.normsys_mask,results_norm,self.normsys_default)

        results_histo   = _hfinterp_code0(self.histosys_histoset,histosys_alphaset)
        results_histo   = np.where(self.histosys_mask,results_histo,self.histosys_default)


        statfactors = np.asarray([pars[self.config.par_slice(m)] for m,mtype in self.do_mods if mtype == 'staterror' ])
        results_staterr = self.staterror_mask * statfactors.reshape(statfactors.shape + (1,1))
        results_staterr = np.where(self.staterror_mask,results_staterr,self.staterror_default)

        normfactors = np.asarray([pars[self.config.par_slice(m)] for m,mtype in self.do_mods if mtype == 'normfactor' ])
        results_normfac = self.normfactor_mask * normfactors.reshape(normfactors.shape + (1,1))
        results_normfac = np.where(self.normfactor_mask,results_normfac,self.normfactor_default)


        thenom = np.asarray([self.mega_samples[s]['nom'] for s in self.do_samples])
        thenom = np.asarray(thenom).reshape((1,)+results_histo.shape[1:])

        allsum = np.concatenate([
            results_histo,
            thenom
        ])

        nom_plus_delta = np.sum(allsum,axis=0)
        nom_plus_delta = nom_plus_delta.reshape((1,)+nom_plus_delta.shape)

        # print(nom_plus_delta.shape,results_histo.shape)
        allfac = np.concatenate([
            results_norm,
            results_staterr,
            results_normfac,
            nom_plus_delta
        ])
        allfac.shape
        newbysample = np.product(allfac,axis=0)
        newresults = np.sum(newbysample,axis=0)
        return newresults[0] #only one alphas


    def expected_data(self, pars, include_auxdata=True):
        tensorlib, _ = get_backend()
        pars = tensorlib.astensor(pars)
        expected_actual = self.expected_actualdata(pars)

        if not include_auxdata:
            return expected_actual
        expected_constraints = self.expected_auxdata(pars)
        tocat = [expected_actual] if expected_constraints is None else [expected_actual,expected_constraints]
        return tensorlib.concatenate(tocat)

    def old_expected_actualdata(self, pars):
        tensorlib, _ = get_backend()
        pars = tensorlib.astensor(pars)
        data = []
        all_modifications = self._all_modifications(pars)
        cmap = {c['name'] : c for c in self.spec['channels']}
        for cname in self.do_channels:
            channel = cmap[cname]
            sample_stack = [
                self._expected_sample(
                    tensorlib.astensor(sample['data']), #nominal
                    *all_modifications[channel['name']][sample['name']] #mods
                )
                for sample in channel['samples']
            ]
            data.append(tensorlib.sum(tensorlib.stack(sample_stack),axis=0))
        return tensorlib.concatenate(data)

    def old_constraint_logpdf(self, auxdata, pars):
        tensorlib, _ = get_backend()
        # iterate over all constraints order doesn't matter....
        start_index = 0
        summands = None
        for cname in self.config.auxdata_order:
            modifier, modslice = self.config.modifier(cname), \
                self.config.par_slice(cname)
            modalphas = modifier.alphas(pars[modslice])
            end_index = start_index + int(modalphas.shape[0])
            thisauxdata = auxdata[start_index:end_index]
            start_index = end_index
            constraint_term = tensorlib.log(modifier.pdf(thisauxdata, modalphas))
            summands = constraint_term if summands is None else tensorlib.concatenate([summands,constraint_term])
        return tensorlib.sum(summands) if summands is not None else 0


    def constraint_logpdf(self, auxdata, pars):
        tensorlib, _ = get_backend()
        start_index = 0
        bytype = {}
        for cname in self.config.auxdata_order:
            modifier, modslice = self.config.modifier(cname), \
                self.config.par_slice(cname)
            modalphas = modifier.alphas(pars[modslice])
            end_index = start_index + int(modalphas.shape[0])
            thisauxdata = auxdata[start_index:end_index]
            start_index = end_index
            if modifier.pdf_type=='normal':
                if modifier.__class__.__name__ in ['histosys','normsys']:
                    kwargs = {'sigma': tensorlib.astensor([1])}
                elif modifier.__class__.__name__ in ['staterror']:
                    kwargs = {'sigma': self.finalized_stats[cname]}
            else:
                kwargs = {}
            callargs = [thisauxdata,modalphas] + [kwargs['sigma'] if kwargs else []]
            bytype.setdefault(modifier.pdf_type,[]).append(callargs)
        return self.__calculate_constraint(bytype)

    def __calculate_constraint(self,bytype):
        tensorlib, _ = get_backend()
        newsummands = None
        for k,c in bytype.items():
            c = tensorlib.astensor(c)
            #warning, call signature depends on pdf_type (2 for pois, 3 for normal)
            pdfval = getattr(tensorlib,k)(c[:,0],c[:,1],c[:,2])
            constraint_term = tensorlib.log(pdfval)
            newsummands = constraint_term if newsummands is None else tensorlib.concatenate([newsummands,constraint_term])

        if newsummands is None:
            return 0
        tosum = newsummands
        return tensorlib.sum(tosum)

    def logpdf(self, pars, data):
        tensorlib, _ = get_backend()
        pars, data = tensorlib.astensor(pars), tensorlib.astensor(data)
        cut = int(data.shape[0]) - len(self.config.auxdata)
        actual_data, aux_data = data[:cut], data[cut:]
        lambdas_data = self.expected_actualdata(pars)
        summands = tensorlib.log(tensorlib.poisson(actual_data, lambdas_data))

        tosum = summands
        mainpdf = tensorlib.sum(tosum)




        constraint = self.constraint_logpdf(aux_data, pars)
        # print('main',mainpdf)
        # print('cons',constraint)


        result = mainpdf + constraint
        return tensorlib.astensor(result) * tensorlib.ones((1)) #ensure (1,) array shape also for numpy

    def pdf(self, pars, data):
        tensorlib, _ = get_backend()
        return tensorlib.exp(self.logpdf(pars, data))
