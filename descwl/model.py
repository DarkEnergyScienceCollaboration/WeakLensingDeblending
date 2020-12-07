"""Model astronomical sources.
"""
from __future__ import print_function, division

import math
import inspect

import numpy as np
import numpy.linalg

import galsim


def sersic_second_moments(n,hlr,q,beta):
    """Calculate the second-moment tensor of a sheared Sersic radial profile.

    Args:
        n(int): Sersic index of radial profile. Only n = 1 and n = 4 are supported.
        hlr(float): Radius of 50% isophote before shearing, in arcseconds.
        q(float): Ratio b/a of Sersic isophotes after shearing.
        beta(float): Position angle of sheared isophotes in radians, measured anti-clockwise
            from the positive x-axis.

    Returns:
        numpy.ndarray: Array of shape (2,2) with values of the second-moments tensor
            matrix, in units of square arcseconds.

    Raises:
        RuntimeError: Invalid Sersic index n.
    """
    # Lookup the value of cn = 0.5*(r0/hlr)**2 Gamma(4*n)/Gamma(2*n)
    if n == 1:
        cn = 1.06502
    elif n == 4:
        cn = 10.8396
    else:
        raise RuntimeError('Invalid Sersic index n.')
    e_mag = (1.-q)/(1.+q)
    e_mag_sq = e_mag**2
    e1 = e_mag*math.cos(2*beta)
    e2 = e_mag*math.sin(2*beta)
    Q11 = 1 + e_mag_sq + 2*e1
    Q22 = 1 + e_mag_sq - 2*e1
    Q12 = 2*e2
    return np.array(((Q11,Q12),(Q12,Q22)))*cn*hlr**2/(1-e_mag_sq)**2
    #return np.array(((Q11,Q12),(Q12,Q22)))*cn*hlr**2

def moments_size_and_shape(Q):
    """Calculate size and shape parameters from a second-moment tensor.

    If the input is an array of second-moment tensors, the calculation is vectorized
    and returns a tuple of output arrays with the same leading dimensions (...).

    Args:
        Q(numpy.ndarray): Array of shape (...,2,2) containing second-moment tensors,
            which are assumed to be symmetric (only the [0,1] component is used).

    Returns:
        tuple: Tuple (sigma_m,sigma_p,a,b,beta,e1,e2) of :class:`numpy.ndarray` objects
            with shape (...). Refer to :ref:`analysis-results` for details on how
            each of these vectors is defined.
    """
    trQ = np.trace(Q,axis1=-2,axis2=-1)
    detQ = np.linalg.det(Q)
    sigma_m = np.power(detQ,0.25)
    sigma_p = np.sqrt(0.5*trQ)
    asymQx = Q[...,0,0] - Q[...,1,1]
    asymQy = 2*Q[...,0,1]
    asymQ = np.sqrt(asymQx**2 + asymQy**2)
    a = np.sqrt(0.5*(trQ + asymQ))
    b = np.sqrt(0.5*(trQ - asymQ))
    beta = 0.5*np.arctan2(asymQy,asymQx)
    e_denom = trQ + 2*np.sqrt(detQ)
    e1 = asymQx/e_denom
    e2 = asymQy/e_denom
    return sigma_m,sigma_p,a,b,beta,e1,e2

def sheared_second_moments(Q,g1,g2):
    """Apply shear to a second moments matrix.

    The shear M = ((1-g1,-g2),(-g2,1+g1)) is applied to each Q by calculating
    Q' = (M**-1).Q.(M**-1)^t where M**-1 = ((1+g1,g2),(g2,1-g1))/\|M\|.
    If the input is an array of second-moment tensors, the calculation is vectorized
    and returns a tuple of output arrays with the same leading dimensions (...).

    Args:
        Q(numpy.ndarray): Array of shape (...,2,2) containing second-moment tensors,
            which are assumed to be symmetric.
        g1(float): Shear ellipticity component g1 (+) with \|g\| = (a-b)/(a+b).
        g2(float): Shear ellipticity component g2 (x) with \|g\| = (a-b)/(a+b).

    Returns:
        numpy.ndarray: Array with the same shape as the input Q with the shear
            (g1,g2) applied to each 2x2 second-moments submatrix.
    """
    detM = 1 - g1**2 - g2**2
    Minv = np.array(((1+g1,g2),(g2,1-g1)))/detM
    return np.einsum('ia,...ab,jb',Minv,Q,Minv)

class SourceNotVisible(Exception):
    """Custom exception to indicate that a source has no visible model components.
    """
    pass

class Galaxy(object):
    """Source model for a galaxy.

    Galaxies are modeled using up to three components: a disk (Sersic n=1), a bulge
    (Sersic n=4), and an AGN (PSF-like). Not all components are required.  All components
    are assumed to have the same centroid and the extended (disk,bulge) components are
    assumed to have the same position angle.

    Args:
        identifier(int): Unique integer identifier for this galaxy in the source catalog.
        redshift(float): Catalog redshift of this galaxy.
        ab_magnitude(float): Catalog AB magnitude of this galaxy in the filter band being
            simulated.
        ri_color(float): Catalog source color calculated as (r-i) AB magnitude difference.
        cosmic_shear_g1(float): Cosmic shear ellipticity component g1 (+) with \|g\| = (a-b)/(a+b).
        cosmic_shear_g2(float): Cosmic shear ellipticity component g2 (x) with \|g\| = (a-b)/(a+b).
        dx_arcsecs(float): Horizontal offset of catalog entry's centroid from image center
            in arcseconds.
        dy_arcsecs(float): Vertical offset of catalog entry's centroid from image center
            in arcseconds.
        beta_radians(float): Position angle beta of Sersic components in radians, measured
            anti-clockwise from the positive x-axis. Ignored if disk_flux and bulge_flux are
            both zero.
        disk_flux(float): Total flux in detected electrons of Sersic n=1 component.
        disk_hlr_arcsecs(float): Half-light radius sqrt(a*b) of circularized 50% isophote
            for Sersic n=1 component, in arcseconds. Ignored if disk_flux is zero.
        disk_q(float): Ratio b/a of 50% isophote semi-minor (b) to semi-major (a) axis
            lengths for Sersic n=1 component. Ignored if disk_flux is zero.
        bulge_flux(float): Total flux in detected electrons of Sersic n=4 component.
        bulge_hlr_arcsecs(float): Half-light radius sqrt(a*b) of circularized 50% isophote
            for Sersic n=4 component, in arcseconds. Ignored if bulge_flux is zero.
        bulge_q(float): Ratio b/a of 50% isophote semi-minor (b) to semi-major (a) axis
            lengths for Sersic n=4 component. Ignored if bulge_flux is zero.
        agn_flux(float): Total flux in detected electrons of PSF-like component.
    """
    def __init__(self,identifier,redshift,ab_magnitude,ri_color,
        cosmic_shear_g1,cosmic_shear_g2,
        dx_arcsecs,dy_arcsecs,beta_radians,disk_flux,disk_hlr_arcsecs,disk_q,
        bulge_flux,bulge_hlr_arcsecs,bulge_q,agn_flux):
        self.identifier = identifier
        self.redshift = redshift
        self.ab_magnitude = ab_magnitude
        self.ri_color = ri_color
        self.dx_arcsecs = dx_arcsecs
        self.dy_arcsecs = dy_arcsecs
        self.cosmic_shear_g1 = cosmic_shear_g1
        self.cosmic_shear_g2 = cosmic_shear_g2
        components = [ ]
        # Initialize second-moments tensor. Note that we can only add the tensors for the
        # n = 1,4 components, as we do below, since they have the same centroid.
        self.second_moments = np.zeros((2,2))
        total_flux = disk_flux + bulge_flux + agn_flux
        self.disk_fraction = disk_flux/total_flux
        self.bulge_fraction = bulge_flux/total_flux
        if disk_flux > 0:
            disk = galsim.Exponential(flux = disk_flux, half_light_radius = disk_hlr_arcsecs).shear(q = disk_q, beta = beta_radians*galsim.radians)
            components.append(disk)
            self.second_moments += self.disk_fraction*sersic_second_moments(
                n=1,hlr=disk_hlr_arcsecs,q=disk_q,beta=beta_radians)

        if bulge_flux > 0:
            bulge = galsim.DeVaucouleurs(
                flux = bulge_flux, half_light_radius = bulge_hlr_arcsecs).shear(
                q = bulge_q, beta = beta_radians*galsim.radians)
            components.append(bulge)
            self.second_moments += self.bulge_fraction*sersic_second_moments(
                n=1,hlr=bulge_hlr_arcsecs,q=bulge_q,beta=beta_radians)
        # GalSim does not currently provide a "delta-function" component to model the AGN
        # so we use a very narrow Gaussian. See this GalSim issue for details:
        # https://github.com/GalSim-developers/GalSim/issues/533
        if agn_flux > 0:
            agn = galsim.Gaussian(flux = agn_flux, sigma = 1e-8)
            components.append(agn)

        # Combine the components into our final profile.
        self.profile = galsim.Add(components)
        # Apply transforms to build the final model.

        self.model = self.get_transformed_model()

        # Shear the second moments, if necessary.
        if self.cosmic_shear_g1 != 0 or self.cosmic_shear_g2 != 0:
            self.second_moments = sheared_second_moments(
                self.second_moments,self.cosmic_shear_g1,self.cosmic_shear_g2)

    def get_transformed_model(self,dx=0.,dy=0.,ds=0.,dg1=0.,dg2=0.):
        """Apply transforms to our model.

        The nominal model returned by `get_transformed_model()` is available via
        the `model` attribute.

        Args:
            dx(float): Amount to shift centroid in x, in arcseconds.
            dy(float): Amount to shift centroid in y, in arcseconds.
            ds(float): Relative amount to scale the galaxy profile in the
                radial direction while conserving flux, before applying shear
                or convolving with the PSF.
            dg1(float): Amount to adjust the + shear applied to the galaxy profile,
                with \|g\| = (a-b)/(a+b), before convolving with the PSF.
            dg2(float): Amount to adjust the x shear applied to the galaxy profile,
                with \|g\| = (a-b)/(a+b), before convolving with the PSF.

        Returns:
            galsim.GSObject: New model constructed using our source profile with
                the requested transforms applied.
        """

        return (self.profile
            .dilate(1 + ds)
            .shear(g1 = self.cosmic_shear_g1 + dg1,g2 = self.cosmic_shear_g2 + dg2)
            .shift(dx = self.dx_arcsecs + dx,dy = self.dy_arcsecs + dy))

class GalaxyBuilder(object):
    """Build galaxy source models.

    Args:
        survey(descwl.survey.Survey): Survey to use for flux normalization and cosmic shear.
        no_disk(bool): Ignore any Sersic n=1 component in the model if it is present in the catalog.
        no_bulge(bool): Ignore any Sersic n=4 component in the model if it is present in the catalog.
        no_agn(bool): Ignore any PSF-like component in the model if it is present in the catalog.
        verbose_model(bool): Provide verbose output from model building process.
    """
    def __init__(self,survey,no_disk,no_bulge,no_agn,verbose_model):
        if no_disk and no_bulge and no_agn:
            raise RuntimeError('Must build at least one galaxy component.')
        self.survey = survey
        self.no_disk = no_disk
        self.no_bulge = no_bulge
        self.no_agn = no_agn
        self.verbose_model = verbose_model

    def from_catalog(self,entry,dx_arcsecs,dy_arcsecs,filter_band):
        """Build a :class:Galaxy object from a catalog entry.

        Fluxes are distributed between the three possible components (disk,bulge,AGN) assuming
        that each component has the same spectral energy distribution, so that the resulting
        proportions are independent of the filter band.

        Args:
            entry(astropy.table.Row): A single row from a galaxy :mod:`descwl.catalog`.
            dx_arcsecs(float): Horizontal offset of catalog entry's centroid from image center
                in arcseconds.
            dy_arcsecs(float): Vertical offset of catalog entry's centroid from image center
                in arcseconds.
            filter_band(str): The LSST filter band to use for calculating flux, which must
                be one of 'u','g','r','i','z','y'.

        Returns:
            :class:`Galaxy`: A newly created galaxy source model.

        Raises:
            SourceNotVisible: All of the galaxy's components are being ignored.
            RuntimeError: Catalog entry is missing AB flux value in requested filter band.
        """
        # Calculate the object's total flux in detected electrons.
        try:
            ab_magnitude = entry[filter_band + '_ab']
            ri_color = entry['r_ab'] - entry['i_ab']
        except KeyError:
            raise RuntimeError('Catalog entry is missing required AB magnitudes.')
        total_flux = self.survey.get_flux(ab_magnitude)
        # Calculate the flux of each component in detected electrons.
        total_fluxnorm = entry['fluxnorm_disk'] + entry['fluxnorm_bulge'] + entry['fluxnorm_agn']
        disk_flux = 0. if self.no_disk else entry['fluxnorm_disk']/total_fluxnorm*total_flux
        bulge_flux = 0. if self.no_bulge else entry['fluxnorm_bulge']/total_fluxnorm*total_flux
        agn_flux = 0. if self.no_agn else entry['fluxnorm_agn']/total_fluxnorm*total_flux
        # Is there any flux to simulate?
        if disk_flux + bulge_flux + agn_flux == 0:
            raise SourceNotVisible
        # Calculate the position of angle of the Sersic components, which are assumed to be the same.
        if disk_flux > 0:
            beta_radians = math.radians(entry['pa_disk'])
            if bulge_flux > 0:
                assert entry['pa_disk'] == entry['pa_bulge'],'Sersic components have different beta.'
        elif bulge_flux > 0:
            beta_radians = math.radians(entry['pa_bulge'])
        else:
            # This might happen if we only have an AGN component.
            beta_radians = None
        # Calculate shapes hlr = sqrt(a*b) and q = b/a of Sersic components.
        if disk_flux > 0:
            a_d,b_d = entry['a_d'],entry['b_d']
            disk_hlr_arcsecs = math.sqrt(a_d*b_d)
            disk_q = b_d/a_d
        else:
            disk_hlr_arcsecs,disk_q = None,None
        if bulge_flux > 0:
            a_b,b_b = entry['a_b'],entry['b_b']
            bulge_hlr_arcsecs = math.sqrt(a_b*b_b)
            bulge_q = b_b/a_b
            bulge_beta = math.radians(entry['pa_bulge'])
        else:
            bulge_hlr_arcsecs,bulge_q = None,None
        # Look up extra catalog metadata.
        identifier = entry['galtileid']
        redshift = entry['redshift']
        if self.verbose_model:
            print('Building galaxy model for id=%d with z=%.3f' % (identifier,redshift))
            print('flux = %.3g detected electrons (%s-band AB = %.1f)' % (
                total_flux,filter_band,ab_magnitude))
            print('centroid at (%.6f,%.6f) arcsec relative to image center, beta = %.6f rad' % (
                dx_arcsecs,dy_arcsecs,beta_radians))
            if disk_flux > 0:
                print(' disk: frac = %.6f, hlr = %.6f arcsec, q = %.6f' % (
                    disk_flux/total_flux,disk_hlr_arcsecs,disk_q))
            if bulge_flux > 0:
                print('bulge: frac = %.6f, hlr = %.6f arcsec, q = %.6f' % (
                    bulge_flux/total_flux,bulge_hlr_arcsecs,bulge_q))
            if agn_flux > 0:
                print('  AGN: frac = %.6f' % (agn_flux/total_flux))
        return Galaxy(identifier,redshift,ab_magnitude,ri_color,
            self.survey.cosmic_shear_g1,self.survey.cosmic_shear_g2,
            dx_arcsecs,dy_arcsecs,beta_radians,disk_flux,disk_hlr_arcsecs,disk_q,
            bulge_flux,bulge_hlr_arcsecs,bulge_q,agn_flux)

    @staticmethod
    def add_args(parser):
        """Add command-line arguments for constructing a new :class:`GalaxyBuilder`.

        The added arguments are our constructor parameters with '_' replaced by '-' in the names.

        Args:
            parser(argparse.ArgumentParser): Arguments will be added to this parser object using its
                add_argument method.
        """
        parser.add_argument('--no-disk', action = 'store_true',
            help = 'Ignore any Sersic n=1 component in the model if it is present in the catalog.')
        parser.add_argument('--no-bulge', action = 'store_true',
            help = 'Ignore any Sersic n=4 component in the model if it is present in the catalog.')
        parser.add_argument('--no-agn', action = 'store_true',
            help = 'Ignore any PSF-like component in the model if it is present in the catalog.')
        parser.add_argument('--verbose-model', action = 'store_true',
            help = 'Provide verbose output from model building process.')

    @classmethod
    def from_args(cls,survey,args):
        """Create a new :class:`GalaxyBuilder` object from a set of arguments.

        Args:
            survey(descwl.survey.Survey): Survey to build source models for.
            args(object): A set of arguments accessed as a :py:class:`dict` using the
                built-in :py:func:`vars` function. Any extra arguments beyond those defined
                in :func:`add_args` will be silently ignored.

        Returns:
            :class:`GalaxyBuilder`: A newly constructed Reader object.
        """
        # Look up the named constructor parameters.
        pnames = (inspect.getargspec(cls.__init__)).args[1:]
        # Get a dictionary of the arguments provided.
        args_dict = vars(args)
        # Filter the dictionary to only include constructor parameters.
        filtered_dict = { key:args_dict[key] for key in (set(pnames) & set(args_dict)) }
        return cls(survey,**filtered_dict)

class Star(object):
    """Source model for a star.

    Stars are modeled using a PSF-like component.

    Args:
        identifier(int): Unique integer identifier for this galaxy in the source catalog.
        redshift(float): Catalog redshift of this galaxy.
        ab_magnitude(float): Catalog AB magnitude of this galaxy in the filter band being
            simulated.
        ri_color(float): Catalog source color calculated as (r-i) AB magnitude difference.
        dx_arcsecs(float): Horizontal offset of catalog entry's centroid from image center
            in arcseconds.
        dy_arcsecs(float): Vertical offset of catalog entry's centroid from image center
            in arcseconds.
        star_flux(float): Total flux in detected electrons of PSF-like component.
    """
    def __init__(self,identifier,redshift,ab_magnitude,ri_color,
        dx_arcsecs,dy_arcsecs,star_flux):
        self.identifier = identifier
        self.redshift = redshift
        self.ab_magnitude = ab_magnitude
        self.ri_color = ri_color
        self.dx_arcsecs = dx_arcsecs
        self.dy_arcsecs = dy_arcsecs
        components = [ ]
        total_flux = star_flux
        # GalSim does not currently provide a "delta-function" component to model the AGN
        # so we use a very narrow Gaussian. See this GalSim issue for details:
        # https://github.com/GalSim-developers/GalSim/issues/533
        if star_flux > 0:
            star = galsim.Gaussian(flux = star_flux, sigma = 1e-8)
            components.append(star)
        # Combine the components into our final profile.
        self.profile = galsim.Add(components)
        # Apply transforms to build the final model.
        self.model = self.get_transformed_model()

    def get_transformed_model(self,dx=0.,dy=0.,ds=0.,dg1=0.,dg2=0.):
        """Apply transforms to our model.

        The nominal model returned by `get_transformed_model()` is available via
        the `model` attribute.

        Args:
            dx(float): Amount to shift centroid in x, in arcseconds.
            dy(float): Amount to shift centroid in y, in arcseconds.
            ds(float): Relative amount to scale the galaxy profile in the
                radial direction while conserving flux, before applying shear
                or convolving with the PSF.
            dg1(float): Amount to adjust the + shear applied to the galaxy profile,
                with \|g\| = (a-b)/(a+b), before convolving with the PSF.
            dg2(float): Amount to adjust the x shear applied to the galaxy profile,
                with \|g\| = (a-b)/(a+b), before convolving with the PSF.

        Returns:
            galsim.GSObject: New model constructed using our source profile with
                the requested transforms applied.
        """
        return (self.profile
            .shift(dx = self.dx_arcsecs + dx,dy = self.dy_arcsecs + dy))

class StarBuilder(object):
    """Build star source models.

    Args:
        survey(descwl.survey.Survey): Survey to use for flux normalization and cosmic shear.
        verbose_star-model(bool): Provide verbose output from model building process.
    """
    def __init__(self,survey,verbose_model):
        self.survey = survey
        self.verbose_model = verbose_model

    def from_catalog(self,entry,dx_arcsecs,dy_arcsecs,filter_band):
        """Build a :class:Star object from a catalog entry.

        Args:
            entry(astropy.table.Row): A single row from a star :mod:`descwl.catalog`.
            dx_arcsecs(float): Horizontal offset of catalog entry's centroid from image center
                in arcseconds.
            dy_arcsecs(float): Vertical offset of catalog entry's centroid from image center
                in arcseconds.
            filter_band(str): The LSST filter band to use for calculating flux, which must
                be one of 'u','g','r','i','z','y'.

        Returns:
            :class:`Star`: A newly created star source model.

        Raises:
            SourceNotVisible: All of the star's components are being ignored.
            RuntimeError: Catalog entry is missing AB flux value in requested filter band.
        """
        # Calculate the object's total flux in detected electrons.
        try:
            ab_magnitude = entry[filter_band + '_ab']
            ri_color = entry['r_ab'] - entry['i_ab']
        except KeyError:
            raise RuntimeError('Catalog entry is missing required AB magnitudes.')
        total_flux = self.survey.get_flux(ab_magnitude)
        # Calculate the flux of each component in detected electrons.
        total_fluxnorm = entry['fluxnorm_star']
        star_flux = total_flux
        # Is there any flux to simulate?
        if star_flux == 0:
            raise SourceNotVisible
        # Calculate the position of angle of the Sersic components, which are assumed to be the same.
        identifier = entry['startileid']
        redshift = entry['redshift']
        if self.verbose_model:
            print('Building star model for id=%d with z=%.3f' % (identifier,redshift))
            print('flux = %.3g detected electrons (%s-band AB = %.1f)' % (
                total_flux,filter_band,ab_magnitude))
            print('centroid at (%.6f,%.6f) arcsec relative to image center' % (
                dx_arcsecs,dy_arcsecs))

        return Star(identifier,redshift,ab_magnitude,ri_color,
            dx_arcsecs,dy_arcsecs,star_flux)

    @staticmethod
    def add_args(parser):
        """Add command-line arguments for constructing a new :class:`StarBuilder`.

        The added arguments are our constructor parameters with '_' replaced by '-' in the names.

        Args:
            parser(argparse.ArgumentParser): Arguments will be added to this parser object using its
                add_argument method.
        """

        parser.add_argument('--verbose-star-model', action = 'store_true',
            help = 'Provide verbose output from model building process.')

    @classmethod
    def from_args(cls,survey,args):
        """Create a new :class:`StarBuilder` object from a set of arguments.

        Args:
            survey(descwl.survey.Survey): Survey to build source models for.
            args(object): A set of arguments accessed as a :py:class:`dict` using the
                built-in :py:func:`vars` function. Any extra arguments beyond those defined
                in :func:`add_args` will be silently ignored.

        Returns:
            :class:`StarBuilder`: A newly constructed ReaderStar object.
        """
        # Look up the named constructor parameters.
        pnames = (inspect.getargspec(cls.__init__)).args[1:]
        # Get a dictionary of the arguments provided.
        args_dict = vars(args)
        # Filter the dictionary to only include constructor parameters.
        filtered_dict = { key:args_dict[key] for key in (set(pnames) & set(args_dict)) }
        return cls(survey,**filtered_dict)
