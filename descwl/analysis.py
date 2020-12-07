"""Perform weak-lensing analysis of simulated sources.
"""
from __future__ import print_function, division

import numpy as np
import scipy.spatial

import astropy.table

import galsim

import lmfit

import descwl.model

from distutils.version import LooseVersion

from six import iteritems
def grl_equilibration(fish):
    """Algorithm for equilibrating fisher matrices of any shape. This is useful when inverting fisher matrices that have particularly high condition number.

    Note: The 1e4 is the 'magic number' that was obtained by looking at the histograms of fisher elemenets of the isolated galaixes. 
    """
    dim = fish.shape[0]
    eqi = np.eye(dim)
    for i in range(dim): 
        if i%6==0:
            eqi[i,i] = 1e4
    return eqi.dot(fish.dot(eqi)) #creates a copy. 

def make_positions():
    """Create dictionary of mapping from corresponding partial names to position in datacube.
    """

    slice_labels = OverlapResults.slice_labels

    positions = {}

    for i,pname_i in enumerate(slice_labels[1:]):
        positions[pname_i] = i+1
        for j,pname_j in enumerate(slice_labels[1:]):
            if(j>=i):
                positions[pname_i,pname_j] = ((9 - i) * i) // 2 + j + 6 #double partials
    positions[slice_labels[0]] = 0 #nominal image. 
    return positions

def make_inv_positions():
    return {value:key for key,value in iteritems(make_positions())}

class OverlapResults(object):
    """Results of analyzing effects of overlapping sources on weak lensing.

    Results are normally obtained by calling :meth:`OverlapAnalyzer.finalize` after
    simulating an image or using :meth:`descwl.output.Reader` to load saved
    simulation results.

    Indices returned by the `find` methods below can be used to lookup analysis
    results via `table[index]` and the corresponding simulated datacube using
    `stamps[index]` and `bounds[index]`.

    Args:
        survey(descwl.survey.Survey): Simulated survey that results are based on.
        table(astropy.table.Table): Table of analysis results with one row per galaxy.
        stamps(array): Array of :class:`numpy.ndarray` postage-stamp datacubes. Might be None.
        bounds(array): Array of galsim.BoundsI objects giving pixel coordinates of
            each datacube in the full simulated image. Might be None.
        num_slices(int): Number of datacube slices for each stamp. Ignored if stamps is None.
        is_star(bool): True if the object is a star.
    Raises:
        RuntimeError: Image datacubes have unexpected number of slices.
    """
    def __init__(self,survey,table,stamps,bounds,num_slices,is_star):
        self.survey = survey
        self.table = table
        if self.table is not None:
            self.num_objects = len(self.table)
            self.locals = { name: self.table[name] for name in self.table.colnames }
        self.stamps = stamps
        self.bounds = bounds
        self.num_slices = num_slices
        self.is_star = is_star
        if len(self.stamps) > 0:
             #The '21' immediately below refers to the case when second partials are required (21 is 1 + 5 + 15, where the first term refers to the galaxy image itself, the 5 is the number of first partials excluding flux and 15 is the number of second partials excluding flux and using that partial derivatives commute.)
            if self.num_slices not in (1, 3, 6, 21):
                raise RuntimeError('Image datacubes have unexpected number of slices (%d).'
                    % self.num_slices)
        self.noise_seed = None

    slice_labels = ['dflux','dx','dy','ds','dg1','dg2']

    def add_noise(self,noise_seed):
        """Add Poisson noise to the simulated survey image.

        Args:
            noise_seed(int): Random seed to use.

        Raises:
            RuntimeError: Noise has already been added.
        """
        if self.noise_seed is not None:
            raise RuntimeError('Noise has already been added to the simulated image.')
        self.noise_seed = noise_seed
        generator = galsim.random.BaseDeviate(seed = self.noise_seed)
        noise = galsim.PoissonNoise(rng = generator, sky_level = self.survey.mean_sky_level)
        self.survey.image.addNoise(noise)

    def select(self,*selectors,**options):
        """Select objects.

        This function is implemented using :func:`eval` so should only be used when the
        source of the selector strings is trusted.

        Args:
            selectors(str,...): A sequence of one or more selection strings, each
                consisting of a boolean expression of column table names, e.g.
                'snr_iso > 5'.  The special strings 'ALL' and 'NONE' do what you
                would expect.
            mode(str): The string 'and' or 'or' to specify how multiple selectors
                should be combined.
            format(str): The string 'mask' or 'index' to specify the format of the
                returned array.  A mask is an array of booleans with one entry per
                source. The alternative index array of integers lists the selected
                indices in increasing order and might be empty.  The mask format
                is useful for performing your own logical operations on the
                returned array. The index format is useful for iterating over the
                selected objects. Both formats can be used to slice the catalog
                table to extract only the selected rows.

        Returns:
            :class:`numpy.ndarray`: A numpy array of booleans or integers, depending
                on the value of the format input argument.

        Raises:
            RuntimeError: A selector uses an undefined variable name, the method was
                passed an unexpected mode or format string, or these results have
                no catalog table.
        """
        if self.table is None:
            raise RuntimeError('Cannot select() without a catalog table.')
        mode = options.get('mode','and')
        if mode not in ('and','or'):
            raise RuntimeError('Unexpected mode "%s".' % mode)
        format = options.get('format','index')
        if format not in ('mask','index'):
            raise RuntimeError('Unexpected format "%s".' % format)

        selection = self._select('ALL') if mode == 'and' else self._select('NONE')
        for selector in selectors:
            update = self._select(selector)
            if mode == 'and':
                selection = np.logical_and(selection,update)
            else:
                selection = np.logical_or(selection,update)

        if format == 'index':
            selection = np.arange(self.num_objects)[selection]
        return selection

    def _select(self,selector):
        # This private method implements select() for a single selector.
        if selector == 'ALL':
            return np.ones(self.num_objects,dtype=bool)
        elif selector == 'NONE':
            return np.zeros(self.num_objects,dtype=bool)
        else:
            try:
                return eval(selector,self.locals)
            except NameError as e:
                raise RuntimeError('%s in selector %r.' % (e.message,selector))

    def get_stamp(self,index,datacube_index=0):
        """Return the simulated postage stamp for a single galaxy.

        Args:
            index(int): Index of the requested galaxy in our results. Use
                meth:`find_galaxy` to get the index for an identifier.
            datacube_index(int): Which slice of the datacube to return.

        Returns:
            galsim.ImageView: A GalSim image for the requested galaxy. Note that the
                return value is a read-only view of our internal stamps array.

        Raises:
            RuntimeError: No such galaxy in the results.
        """
        if index < 0 or index >= len(self.stamps):
            raise RuntimeError('No such galaxy with index=%d.' % index)
        bounds = self.bounds[index]
        datacube = self.stamps[index]
        if callable(datacube):
            datacube = datacube()
            self.stamps[index] = datacube
        image = galsim.Image(datacube[datacube_index],
            xmin = bounds.xmin,ymin = bounds.ymin,
            scale = self.survey.pixel_scale,make_const = False)
        return image

    def get_subimage(self,indices,datacube_index=0):
        """Return simulated subimage of a set of objects.

        Args:
            indices(iterable): Indices of the objects to include in the subimage.
                Our :meth:`select` method returns a suitable argument to use here
                by default (format = 'index').

        Returns:
            galsim.Image: Image of the selected objects or None if indices is empty.

        Raises:
            RuntimeError: An index is out of range.
        """
        if len(indices) == 0:
            return None
        subimage_bounds = galsim.BoundsI()
        try:
            for index in indices:
                subimage_bounds += self.bounds[index]
        except IndexError:
            raise RuntimeError('Index out of range %d' % index)
        subimage = galsim.Image(bounds = subimage_bounds, scale = self.survey.pixel_scale)
        for index in indices:
            stamp = self.get_stamp(index,datacube_index)
            overlap = subimage_bounds & stamp.bounds
            subimage[overlap] += stamp[overlap]
        return subimage

    def get_fisher_images(self,index1,index2,background):
        """Return Fisher-matrix images for a pair of galaxies.

        Fisher matrix images are derived from the partial derivatives with respect to
        six parameters: total flux in detected electrons, centroid positions in x and y
        in arcseconds, flux-preserving radial scale (dimensionless), and shear g1,g2
        with \|g\| = (a-b)/(a+b). Use the :ref:`prog-fisher` program to display Fisher
        matrix images.

        Args:
            index1(int): Index of the first galaxy to use.
            index2(int): Index of the second galaxy to use, which might the same as index1.
            background(galsim.Image): Background image that combines all sources that overlap
                the two galaxies and completely contains them.

        Returns:
            tuple: Tuple (images,overlap) where images is a :class:`numpy.ndarray` with shape
                (npar,npar,height,width), where npar=6 and (height,width) are the dimensions
                of the overlap between the two galaxies, and overlap gives the bounding box of the
                overlap in the full survey image. Returns None,None if the two galaxies do not overlap.

        Raises:
            RuntimeError: Invalid index1 or index2, or galaxies are not contained with the
                background image, or no partial derivative images are available.
        """
        #npar = len(self.slice_labels) #this are the actual number of partials
        if self.num_slices not in (3, 6, 21):
            raise RuntimeError('No partial derivative images are available.')
        # Calculate the overlap bounds.
        try:
            overlap = self.bounds[index1] & self.bounds[index2]
        except IndexError:
            raise RuntimeError('Invalid index1=%d or index2=%d.' % (index1,index2))
        # Check that the background image contains each galaxy.
        if not background.bounds.includes(self.bounds[index1]):
            raise RuntimeError('Galaxy %d is not contained within the background image.' % index1)
        if not background.bounds.includes(self.bounds[index2]):
            raise RuntimeError('Galaxy %d is not contained within the background image.' % index2)
        # Is there any overlap between the two galaxies?
        if overlap.area() == 0:
            return None,None
        product = self.get_stamp(index1, 0)[overlap]*self.get_stamp(index2, 0)[overlap]
        if not np.any(product.array):
            return None, None
        # Fill arrays of partial derivatives within the overlap region.
        width = overlap.xmax - overlap.xmin + 1
        height = overlap.ymax - overlap.ymin + 1
        npar1 = 3 if self.is_star[index1] else 6
        npar2 = 3 if self.is_star[index2] else 6
        partials1 = np.empty((npar1, height, width),dtype = np.float32)
        partials2 = np.empty((npar2, height, width),dtype = np.float32)
        for islice in range(npar1):
            partials1[islice] = self.get_stamp(index1,islice)[overlap].array
        for islice in range(npar2):
            partials2[islice] = self.get_stamp(index2,islice)[overlap].array
        partials1[0] /= self.table['flux'][index1]
        partials2[0] /= self.table['flux'][index2]
        # Normalize the Fisher images.
        mu0 = background[overlap].array + self.survey.mean_sky_level
        fisher_norm = mu0**-1 + 0.5*mu0**-2
        # Calculate the Fisher images as the outer product of the partial-derivative images.
        images = np.einsum('yx,iyx,jyx->ijyx',fisher_norm, partials1, partials2)
        return images, overlap


    def get_bias_tensor_images(self,index1,index2,background):
        """Return bias tensor images for a pair of galaxies.

        The bias tensor images are required in the calculation of the bias of parameters of
        the galaxies. Bias tensor images are derived from the first partials and second partials
        of the six parameters.

        We fill in only the second partials of parameters that both belong to same galaxy, because mixed second partials are 0. 


        Args:
            index1(int): Index of the first galaxy to use.
            index2(int): Index of the second galaxy to use, which might the same as index1.
            background(galsim.Image): Background image that combines all sources that overlap
                the two galaxies and completely contains them.

        Returns:
            tuple: Tuple (images,overlap) where images is a :class:`numpy.ndarray` with shape
                (npar,npar,npar,height,width), where npar=6 and (height,width) are the dimensions
                of the overlap between the two galaxies, and overlap gives the bounding box of the
                overlap in the full survey image. Returns None,None if the two galaxies do not overlap.

        Raises:
            RuntimeError: Invalid index1 or index2, or galaxies are not contained with the
                background image, or no partial derivative images are available.
        """
        npar = len(self.slice_labels)
        if self.num_slices != len(self.slice_labels) and self.num_slices != 21:
            raise RuntimeError('No partial derivative images are available.')
        # Calculate the overlap bounds.
        try:
            overlap = self.bounds[index1] & self.bounds[index2]
        except IndexError:
            raise RuntimeError('Invalid index1=%d or index2=%d.' % (index1,index2))
        # Check that the background image contains each galaxy.
        if not background.bounds.includes(self.bounds[index1]):
            raise RuntimeError('Galaxy %d is not contained within the background image.' % index1)
        if not background.bounds.includes(self.bounds[index2]):
            raise RuntimeError('Galaxy %d is not contained within the background image.' % index2)
        # Is there any overlap between the two galaxies?
        if overlap.area() == 0:
            return None,None
        product = self.get_stamp(index1,0)[overlap]*self.get_stamp(index2,0)[overlap]
        if not np.any(product.array):
            return None,None

        # Fill arrays of partial derivatives within the overlap region.
        width = overlap.xmax - overlap.xmin + 1
        height = overlap.ymax - overlap.ymin + 1
        partials1 = np.zeros((npar,height,width),dtype = np.float32) #correspond to index 1 galaxy
        second_partials2 = np.zeros((npar,npar,height,width),dtype = np.float32) #index 2 galaxy

        slice_labels = OverlapResults.slice_labels

        #dictionary for positions of partials.
        positions = make_positions()

        #fill in partial with respect to flux
        partials1[0] = self.get_stamp(index1,0)[overlap].array/self.table['flux'][index1]

        #fill partials and second partials,
        for islice in range(1,npar):
            datacube_index1 = islice
            partials1[islice] = self.get_stamp(index1,datacube_index1)[overlap].array

            #fill in second partials with respect to flux.
            second_partials2[islice][0] = (self.get_stamp(index2,islice)[overlap].array/
                                           self.table['flux'][index2])
            second_partials2[0][islice] = second_partials2[islice][0]


            for jslice in range(islice,npar):
                param_i = slice_labels[islice]
                param_j = slice_labels[jslice]
                datacube_index2 = positions[param_i,param_j]
                second_partials2[islice][jslice] = (self.get_stamp(index2,datacube_index2)[overlap]
                                                                                          .array)

                #complete second partial array with i,j-->j,i just in case. partials must commute. 
                if islice!=jslice:
                    second_partials2[jslice][islice] = second_partials2[islice][jslice]

        #implicitly assume that second_partials2[0][0] = 0 

        mu0 = background[overlap].array + self.survey.mean_sky_level
        fisher_norm = mu0**-1 + 0.5*mu0**-2
        images = np.einsum('yx,iyx,jkyx->ijkyx',fisher_norm,partials1,second_partials2)
        return images,overlap

    def get_bias(self, selected, covariance):
        """Return bias of the 6 parameters in vector form.

        The bias is obtained from contracting the bias tensor with the appropiate covariance
        matrix elements.

        We create reduced_covariance matrix which contains only the diagonal blocks from covariance matrix, the diagonal blocks correspond to the fisher elements formed by the same galaxy. This is necessary because kl has to refer to the same galaxy for the bias_tensor to be non-zero. 


        Args:
            selected(iterable): Array of integer indices for the sources to include in the
                calculated matrices.
            covariance(array): An array containing the covariance matrix of the selected galaxies.

        Returns:
            array: bias which is a vector with dimensions (nbias)
                   containing the biases of the selected galaxies.
        """
        nsel = len(selected)
        npar = len(self.slice_labels)
        nbias = nsel*npar

        bias_tensor = self.get_bias_tensor(selected, covariance)
        bias = np.zeros(nbias, dtype=np.float64)


        reduced_covariance = np.zeros((nsel*npar,npar), dtype=np.float64)
        for i in range(nsel):
            reduced_covariance[npar*i:npar*(i+1),:]=covariance[npar*i:npar*(i+1),npar*i:npar*(i+1)]

        bias = (-.5)*np.einsum('ij,kl,jkl->i',covariance, reduced_covariance, bias_tensor)

        return bias

    def get_bias_tensor(self,selected,covariance):
        """Return bias tensor from the selected galaxies.

        Uses the function get_bias_tensor_images() and then contracts these images to obtain the
        actual bias tensor.


        Args:
            selected(iterable): Array of integer indices for the sources to include in the
                calculated matrices.
            covariance(array): An array containing the covariance matrix of the selected galaxies.

        Returns:
            array: bias_tensor with dimensions (ntensor,ntensor,npar) containing
                   bias_tensor elements.
        """

        background = self.get_subimage(selected)
        nsel = len(selected)
        npar = len(self.slice_labels)
        ntensor = nsel*npar
        bias_tensor = np.zeros((ntensor,ntensor,npar), dtype=np.float64)
        for row,index1 in enumerate(selected):
            for col,index2 in enumerate(selected):
                images, overlap = self.get_bias_tensor_images(index1,index2,background)

                if overlap is None:
                    continue

                bias_tensor_sums = np.sum(images,axis=(3,4),dtype = np.float64)
                bias_tensor[npar*row:npar*(row+1),npar*col:npar*(col+1),:] = bias_tensor_sums

        return bias_tensor

    def get_matrices(self,selected, get_cond_num=False, equilibrate=False):
        """Return matrices derived the from Fisher-matrix images for a set of sources.

        If the (optionally equilibrated) Fisher matrix is not invertible or any variances are <= 0, we will drop
        the selected source with the lowest value of snr_iso and try again. This procedure
        is iterated until we get a valid covariance matrix. Matrix elements for all parameters
        of any sources that get dropped by this procedure will be set to zero and variances
        will be set to np.inf so that 1/np.sqrt(variance) = 0 (so snr_grp = snr_grpf = 0). Invalid covariances are
        generally associated with sources that are barely above the pixel SNR threshold,
        so this procedure should normally provide sensible values for the largest
        possible subset of the input selected sources. Use the :ref:`prog-fisher` program to
        visualize matrix elements and to further study examples of invalid covariances. 
        The Fisher matrix can be optionally equilibrated (using 'equilibrate=True' as an argument) before inversion in order to attempt to reduce its condition number.

        Args:
            selected(iterable): Array of integer indices for the sources to include in the
                calculated matrices.

        Returns:
            tuple: Tuple `(fisher,covariance,variance,correlation)` of :class:`numpy.ndarray`
                where `variance` has shape (npar,) and all other arrays are symmetric with
                shape (npar,npar), where npar = 6*len(selected). Matrix elements will be
                zero for any parameters associated with dropped sources, as described above.
            float: (only if get_cond_num is set to True) condition number of first fisher matrix which is able to be 
            inverted with the above procedure.
        """
        background = self.get_subimage(selected)
        nsel = len(selected)
        npar = []
        for i, idx in enumerate(selected):
             if self.is_star[idx]:
                 npar.append(3)
             else:
                 npar.append(len(self.slice_labels))
        nfisher = np.sum(npar)
        fisher = np.zeros((nfisher, nfisher), dtype=np.float64)
        i0 = 0
        i1 = 0
        for row, index1 in enumerate(selected):
            i1 += npar[row]
            j0 = 0
            j1 = 0
            for col, index2 in enumerate(selected[:row+1]):
                j0 = j1
                j1+= npar[col]
                images, overlap = self.get_fisher_images(index1, index2, background)
                if overlap is None:
                    continue
                fisher_sums = np.sum(images, axis=(2, 3), dtype=np.float64)
                fisher[i0:i1, j0:j1] = fisher_sums
                if row != col:
                    fisher[j0:j1, i0:i1] = fisher_sums.T
            i0 = i1
        # Sort indices into the selected array by increasing snr_iso.
        priority = np.arange(nsel)[np.argsort(self.table['snr_iso'][selected])]
        # Start by trying to use all sources in the group.
        keep = np.ones(nfisher, dtype=bool)
        num_dropped = 0
        idx0 = 0
        idx1 = 0
        while np.any(keep):
            try:
                keep_flat = keep.flatten()
                # Advanced indexing like this makes a copy, not a view.
                reduced_fisher = fisher[keep_flat,:][:,keep_flat]
                if equilibrate: 
                    #equilibrate the fisher matrix.
                    ereduced_fisher = grl_equilibration(reduced_fisher)
                    reduced_cond_num_grp = np.linalg.cond(ereduced_fisher)

                    #attempt to invert equilibrated fisher matrix and equilibrate again to get back to correct "units".
                    reduced_covariance = grl_equilibration(np.linalg.inv(ereduced_fisher))
                else: 
                    reduced_cond_num_grp = np.linalg.cond(reduced_fisher)
                    reduced_covariance = np.linalg.inv(reduced_fisher)
                reduced_variance = np.diagonal(reduced_covariance).copy()
                assert np.min(reduced_variance) > 0,'Expected variance > 0'
                reduced_correlation = reduced_covariance/np.sqrt(
                    np.outer(reduced_variance, reduced_variance))
                break
            except (np.linalg.LinAlgError, AssertionError) as e:
                # We can't calculate a covariance for this set of objects, so drop the next
                # lowest SNR member of the set and try again.
                idx1 += npar[priority[num_dropped]]
                keep[idx0:idx1] = np.zeros(idx1-idx0, dtype=np.bool)
                idx0 = idx1
                num_dropped += 1
        if num_dropped == 0:
            if get_cond_num:
                return (reduced_fisher, reduced_covariance, reduced_variance, reduced_correlation), reduced_cond_num_grp
            else:
                return reduced_fisher, reduced_covariance, reduced_variance, reduced_correlation
        else:
            fisher = np.zeros((nfisher, nfisher), dtype=np.float64)
            covariance = np.zeros((nfisher, nfisher), dtype=np.float64)
            correlation = np.zeros((nfisher, nfisher), dtype=np.float64)
            variance = np.empty((nfisher,), dtype=np.float64)
            variance[:] = np.inf
            # Build matrices with zeros for any sources that had to be dropped. Is there
            # a more elegant way to do this? We cannot simply assign to a submatrix
            # since that requires advancing slicing, which creates a copy, not a view.
            keep_map = np.arange(nsel-num_dropped)
            next = 0
            i0 = 0
            i1 = 0
            i0k = 0
            i1k = 0
            for row in range(nsel):
                if not keep[i0]: continue
                i1 += npar[row]
                row_slice = slice(i0, i1)
                krow = keep_map[row]
                i1k += npar[krow]
                krow_slice = slice(i0k, i1k)
                variance[row_slice] = reduced_variance[krow_slice]
                j0 = 0
                j1 = 0
                j0k = 0
                j1k = 0
                for col in range(row+1):
                    j1 += npar[col]
                    if not keep[j0]: continue
                    col_slice = slice(j0, j1)
                    kcol = keep_map[col]
                    j1k += npar[kcol]
                    kcol_slice = slice(j0k, j1k)
                    fisher[row_slice, col_slice] = reduced_fisher[krow_slice, kcol_slice]
                    covariance[row_slice, col_slice] = reduced_covariance[krow_slice, kcol_slice]
                    correlation[row_slice, col_slice] = reduced_correlation[krow_slice, kcol_slice]
                    if row == col: continue
                    fisher[col_slice, row_slice] = reduced_fisher[kcol_slice, krow_slice]
                    covariance[col_slice, row_slice] = reduced_covariance[kcol_slice, krow_slice]
                    correlation[col_slice, row_slice] = reduced_correlation[kcol_slice, krow_slice]
                    j0 = j1
                    j0k = j1k
                i0 = i1
                i0k = i1k
            if get_cond_num: 
                return (fisher, covariance, variance, correlation), reduced_cond_num_grp
            else: 
                return fisher, covariance, variance, correlation


    def match_sextractor(self,catalog_name,column_name = 'match'):
        """Match detected objects to simulated sources.

        Args:
            catalog_name(str): Name of an ASCII catalog in SExtractor-compatible format, and containing
                X_IMAGE, Y_IMAGE columns and `num_found` rows.
            column_name(str): Name of a column in our `table` data member to fill with the detected
                object index matched to each simulated source, or -1 if no match is found. Overwrites
                any exisiting column or creates a new one.  Does nothing if column_name is None or ''.

        Returns:
            tuple: Tuple `detected,matched,indices,distance` where `detected` is the detected catalog as
                a :class:`astropy.table.Table`, `matched` is an array of `num_found` booleans
                indicating whether each object was matched with a simulated object, `indices` is an array
                of `num_matched = np.count_nonzero(matched)` integers giving row numbers in our `table`
                attribute for each simulated match, and `distance` is an array of `num_matched` separation
                distances in arcseconds between matched and simulated objects.
        """
        # Read the catalog of detected objects.
        detected = astropy.table.Table.read(catalog_name,format='ascii')
        assert 'X_IMAGE' in detected.colnames, 'Missing required column X_IMAGE'
        assert 'Y_IMAGE' in detected.colnames, 'Missing required column Y_IMAGE'
        # Build a KD tree for the simulated source positions (in arcsecs) relative to the image center.
        num_truth = len(self.table)
        xy_truth = np.empty((num_truth,2))
        xy_truth[:,0] = self.table['dx']
        xy_truth[:,1] = self.table['dy']
        kdtree = scipy.spatial.cKDTree(xy_truth)
        # Convert detected object centroids from pixels relative to bottom-left corner to arcsecs
        # into arcsecs relative to the image center.
        num_found = len(detected)
        xy_found = np.empty((num_found,2))
        width,height = self.survey.image_width,self.survey.image_height
        xy_found[:,0] = (detected['X_IMAGE'] - 0.5*width - 0.5)*self.survey.pixel_scale
        xy_found[:,1] = (detected['Y_IMAGE'] - 0.5*height - 0.5)*self.survey.pixel_scale
        # Find nearest simulated source to each detected source.
        min_distance,truth_indices = kdtree.query(xy_found)
        matched = (truth_indices < num_truth)
        indices = truth_indices[matched]
        distance = min_distance[matched]
        # Add a table column with indices of matched detected objects or -1 for unmatched sources.
        if column_name:
            match_lookup = np.empty(len(self.table),dtype = int)
            match_lookup[:] = -1
            match_lookup[indices] = np.arange(len(detected))[matched]
            self.table[column_name] = match_lookup
            # Make this column available via our locals dictionary.
            self.locals[column_name] = self.table[column_name]
        return detected,matched,indices,distance

class OverlapAnalyzer(object):
    """Analyze impact of overlapping sources on weak lensing.

    Args:
        survey(descwl.survey.Survey): Simulated survey to describe with FITS header keywords.
    """
    def __init__(self,survey,no_hsm,no_lmfit,no_fisher, calculate_bias, no_analysis, add_noise, equilibrate, detection_threshold, alpha=1):
        self.survey = survey
        self.models = [ ]
        self.stamps = [ ]
        self.bounds = [ ]
        self.is_star = [ ]
        self.alpha = alpha
        self.no_hsm = no_hsm
        self.no_lmfit = no_lmfit
        self.no_fisher = no_fisher
        self.calculate_bias = calculate_bias
        self.no_analysis = no_analysis
        self.add_noise = add_noise
        self.equilibrate = equilibrate
        self.detection_threshold = detection_threshold # cut on snr_grpf

    def add_galaxy(self,model,stamps,bounds):
        """Add one galaxy to be analyzed.

        Args:
            model(:class:`descwl.model.Galaxy`): The galaxy model used for rendering.
            stamps(:class:`numpy.ndarray`): Array of shape (nstamp,width,height) containing
                pixel values for nstamp stamps of dimensions (width,height).
            bounds(galsim.BoundsI): Bounds of the stamps in the full simulated survey image.
        """
        self.models.append(model)
        self.stamps.append(stamps)
        self.bounds.append(bounds)
        self.is_star.append(False)

    def add_star(self,model,stamps,bounds):
        """
        Args:
            model(:class:`descwl.model.Star`): The star model used for rendering
            stamps(:class:`numpy.ndarray`): Array of shape (nstamp,width,height) containing
                pixel values for nstamp stamps of dimensions (width,height).
            bounds(galsim.BoundsI): Bounds of the stamps in the full simulated survey image
        """
        self.models.append(model)
        self.stamps.append(stamps)
        self.bounds.append(bounds)
        self.is_star.append(True)

    def fit_galaxies(self,indices,observed_image,fixed_parameters = None):
        """Simultaneously fit a set of galaxy parameters to an observed image.

        Fits are performed on noise-free images, so there are no meaningful errors to report
        and only best-fit parameter values are returned.  This method is intended to support
        systematics studies where shifts in best-fit parameters are the quantities of interest.
        See the :meth:`descwl.render.GalaxyRenderer.draw` method for details on how the fit
        parameters are defined and how each galaxy model is rendered as these parameters
        are varied.

        Args:
            indices(iterable): List of num_galaxies integer galaxy indices to include in the fit.
                Index values correspond to the order in which galaxies were added using
                :meth:`add_galaxy`.
            observed_image(galsim.Image): A GalSim image that defines the observed pixels
                that will be fit and the region into which galaxy models will be renderered.
            fixed_parameters(dict): An optional dictionary of parameter values to fix in the
                fit, or None if all parameters should be floating.

        Returns:
            numpy.ndarray: Array with shape (num_galaxies,6) containing the best-fit values
                of the following six parameters: df,dx,dy,ds,dg1,dg2. See the
                :meth:`descwl.render.GalaxyRenderer.draw` method for details on how these
                parameters are defined.

        Raises:
            RuntimeError: Invalid galaxy index or fit did not find a minimum.
        """
        num_galaxies = len(indices)
        # Define the fit parameters we will use.
        parameters = lmfit.Parameters()
        for i in range(num_galaxies):
            parameters.add('df_%d' % i,value = 0.)
            parameters.add('dx_%d' % i,value = 0.)
            parameters.add('dy_%d' % i,value = 0.)
            parameters.add('ds_%d' % i,value = 0.)
            parameters.add('dg1_%d' % i,value = 0.,min = -0.2,max = +0.2)
            parameters.add('dg2_%d' % i,value = 0.,min = -0.2,max = +0.2)
        # Fix parameters as requested.
        if fixed_parameters:
            for name,value in iteritems(fixed_parameters):
                parameters[name].value = value
                parameters[name].vary = False
        # Initialize rendering.
        model_image = observed_image.copy()
        overlap = [ ]
        for i,galaxy in enumerate(indices):
            try:
                bbox = self.bounds[galaxy]
            except (TypeError,ValueError):
                raise RuntimeError('Invalid galaxy index in fit_galaxies: %r' % galaxy)
            overlap.append(model_image.bounds & bbox)
        data = observed_image.array.flatten()
        sigmas = np.sqrt(data + self.survey.mean_sky_level)
        # Define a function that evaluates the pixel residuals to be minimized.
        def residuals(p):
            model_image.array[:] = 0.
            for i,galaxy in enumerate(indices):
                model_stamp = self.models[galaxy].renderer.draw(df = p['df_%d'%i].value,
                    dx = p['dx_%d'%i].value,dy = p['dy_%d'%i].value,ds = p['ds_%d'%i].value,
                    dg1 = p['dg1_%d'%i].value,dg2 = p['dg2_%d'%i].value)
                model_image[overlap[i]] += model_stamp[overlap[i]]
            return (data - model_image.array.flat)/sigmas
        # Do the minimization.
        minimizer = lmfit.Minimizer(residuals,parameters,nan_policy='omit')
        minimum = minimizer.minimize()
        if not minimum.success:
            raise RuntimeError('fit_galaxies did not find a minimum.')
        # Copy the best-fit parameter values into the returned array.
        bestfit_values = np.empty((num_galaxies,6))
        if(LooseVersion(lmfit.__version__) > LooseVersion('0.8.3')):
            for i in range(num_galaxies):
                bestfit_values[i,0] = minimum.params['df_%d'%i].value
                bestfit_values[i,1] = minimum.params['dx_%d'%i].value
                bestfit_values[i,2] = minimum.params['dy_%d'%i].value
                bestfit_values[i,3] = minimum.params['ds_%d'%i].value
                bestfit_values[i,4] = minimum.params['dg1_%d'%i].value
                bestfit_values[i,5] = minimum.params['dg2_%d'%i].value
        else:
            for i in range(num_galaxies):
                bestfit_values[i,0] = parameters['df_%d'%i].value
                bestfit_values[i,1] = parameters['dx_%d'%i].value
                bestfit_values[i,2] = parameters['dy_%d'%i].value
                bestfit_values[i,3] = parameters['ds_%d'%i].value
                bestfit_values[i,4] = parameters['dg1_%d'%i].value
                bestfit_values[i,5] = parameters['dg2_%d'%i].value
        return bestfit_values

    def fit_stars(self,indices,observed_image,fixed_parameters = None):
        """Simultaneously fit a set of star parameters to an observed image.

        Fits are performed on noise-free images, so there are no meaningful errors to report
        and only best-fit parameter values are returned.  This method is intended to support
        systematics studies where shifts in best-fit parameters are the quantities of interest.
        See the :meth:`descwl.render_star.StarRenderer.draw` method for details on how the fit
        parameters are defined and how each galaxy model is rendered as these parameters
        are varied.

        Args:
            indices(iterable): List of num_stars integer star indices to include in the fit.
                Index values correspond to the order in which stars were added using
                :meth:`add_star`.
            observed_image(galsim.Image): A GalSim image that defines the observed pixels
                that will be fit and the region into which star models will be renderered.
            fixed_parameters(dict): An optional dictionary of parameter values to fix in the
                fit, or None if all parameters should be floating.

        Returns:
            numpy.ndarray: Array with shape (num_stars,6) containing the best-fit values
                of the following six parameters: df,dx,dy,ds,dg1,dg2. See the
                :meth:`descwl.render.StarRenderer.draw` method for details on how these
                parameters are defined.

        Raises:
            RuntimeError: Invalid galaxy index or fit did not find a minimum.
        """
        num_stars = len(indices)
        # Define the fit parameters we will use.
        parameters = lmfit.Parameters()
        for i in range(num_stars):
            parameters.add('df_%d' % i,value = 0.)
            parameters.add('dx_%d' % i,value = 0.)
            parameters.add('dy_%d' % i,value = 0.)
            parameters.add('ds_%d' % i,value = 0.)
            parameters.add('dg1_%d' % i,value = 0.,min=-2e-5,max=2e-5)
            parameters.add('dg2_%d' % i,value = 0.,min=-2e-5,max=2e-5)
            parameters['ds_%d' %i].value = 1e-5
            parameters['ds_%d' %i].vary = False
            parameters['dg1_%d' %i].value = 1e-5
            parameters['dg1_%d' %i].vary = False
            parameters['dg2_%d' %i].value = 1e-5
            parameters['dg2_%d' %i].vary = False
            # Fix parameters as requested.
        if fixed_parameters:
            for name,value in iteritems(fixed_parameters):
                parameters[name].value = value
                parameters[name].vary = False
        # Initialize rendering.
        model_image = observed_image.copy()
        overlap = [ ]
        for i,star in enumerate(indices):
            try:
                bbox = self.bounds[star]
            except (TypeError,ValueError):
                raise RuntimeError('Invalid star index in fit_dystd: %r' % star)
            overlap.append(model_image.bounds & bbox)
        data = observed_image.array.flatten()
        sigmas = np.sqrt(data + self.survey.mean_sky_level)
        # Define a function that evaluates the pixel residuals to be minimized.
        def residuals(p):
            model_image.array[:] = 0.
            for i,star in enumerate(indices):
                model_stamp = self.models[star].renderer.draw(df = p['df_%d'%i].value,
                    dx = p['dx_%d'%i].value,dy = p['dy_%d'%i].value,ds = p['ds_%d'%i].value,
                    dg1 = p['dg1_%d'%i].value,dg2 = p['dg2_%d'%i].value)
                model_image[overlap[i]] += model_stamp[overlap[i]]
            return (data - model_image.array.flat)/sigmas
        # Do the minimization.
        minimizer = lmfit.Minimizer(residuals,parameters,nan_policy='omit')
        minimum = minimizer.minimize()
        if not minimum.success:
            raise RuntimeError('fit_stars did not find a minimum.')
        # Copy the best-fit parameter values into the returned array.
        bestfit_values = np.empty((num_stars,6))
        for i in range(num_stars):
            bestfit_values[i,0] = parameters['df_%d'%i].value
            bestfit_values[i,1] = parameters['dx_%d'%i].value
            bestfit_values[i,2] = parameters['dy_%d'%i].value
            bestfit_values[i,3] = parameters['ds_%d'%i].value
            bestfit_values[i,4] = parameters['dg1_%d'%i].value
            bestfit_values[i,5] = parameters['dg2_%d'%i].value
        return bestfit_values

    def finalize(self,verbose,trace):
        """Finalize analysis of all added stars and galaxies.

        Args:
            verbose(bool): Print a summary of analysis results.
            trace(callable): Function to call for tracing resource usage. Will be
                called with a brief :class:`str` description of each checkpoint.

        Returns:
            :class:`OverlapResults`: Overlap analysis results.
        """

        trace('OverlapAnalyzer.finalize begin')
        # Define columns and allocate space for our table data.
        num_galaxies = len(self.models)
        dtype=[
                ('db_id',np.int64),
                ('grp_id',np.int64),
                ('grp_size',np.int16),
                ('grp_rank',np.int16),
                ('visible',np.int16),
                # Stamp bounding box.
                ('xmin',np.int32),
                ('xmax',np.int32),
                ('ymin',np.int32),
                ('ymax',np.int32),
                # Source properties.
                ('f_disk', np.float32),
                ('f_bulge', np.float32),
                ('dx',np.float32),
                ('dy',np.float32),
                ('z',np.float32),
                ('ab_mag',np.float32),
                ('ri_color',np.float32),
                ('flux',np.float32),
                ('sigma_m',np.float32),
                ('sigma_p',np.float32),
                ('e1',np.float32),
                ('e2',np.float32),
                ('a',np.float32),
                ('b',np.float32),
                ('beta',np.float32),
                ('psf_sigm',np.float32),
                # Pixel-level properties.
                ('purity',np.float32),
                ('snr_sky',np.float32),
                ('is_star', np.bool)
            ]

        #from here on, only add relevant entries. 
        if not self.no_fisher and not self.no_analysis: 
            dtype.extend([
            ('snr_iso',np.float32),
            ('snr_grp',np.float32),
            ('snr_isof',np.float32),
            ('snr_grpf',np.float32),
            ('ds',np.float32),
            ('dg1',np.float32),
            ('dg2',np.float32),
            ('ds_grp',np.float32),
            ('dg1_grp',np.float32),
            ('dg2_grp',np.float32),
            ('cond_num', np.float32), #condition number of individual galaxy fisher matrix. 
            ('cond_num_grp', np.float32), #condition number (using 2-norm) from fisher matrix of corresponding group.
            ])

        if self.calculate_bias and not self.no_analysis:
            dtype.extend([
            #fisher bias calculation results. 
            ('bias_f', np.float32),
            ('bias_s', np.float32),
            ('bias_g1',np.float32),
            ('bias_g2',np.float32),
            ('bias_x',np.float32),
            ('bias_y',np.float32),
            ('bias_f_grp', np.float32),
            ('bias_s_grp', np.float32),
            ('bias_g1_grp',np.float32),
            ('bias_g2_grp',np.float32),
            ('bias_x_grp',np.float32),
            ('bias_y_grp',np.float32)
            ])

        if not self.no_hsm and not self.no_analysis:
            dtype.extend([
            # HSM analysis results.
            ('hsm_sigm',np.float32),
            ('hsm_e1',np.float32),
            ('hsm_e2',np.float32)
            ])

        if not self.no_lmfit and not self.no_analysis:
            dtype.extend([
            # Systematics fit results.
            ('g1_fit',np.float32),
            ('g2_fit',np.float32),
            ])

        #initially all entries are unspecified with np.nan. 
        data = np.full(num_galaxies, np.nan, dtype=dtype)

        #skip all analysis. 
        if self.no_analysis:
            # Return empty data table with only a few of the columns. 
            table = astropy.table.Table(data, copy=False)
            num_slices, h, w = self.stamps[0].shape
            results = OverlapResults(self.survey, table, self.stamps,
                                     self.bounds, num_slices, self.is_star)
            return results

        trace('allocated table of %ld bytes for %d galaxies' % (data.nbytes,num_galaxies))

        # Initialize integer arrays of bounding box limits.
        for i in range(num_galaxies):
            data['xmin'][i] = self.bounds[i].xmin
            data['ymin'][i] = self.bounds[i].ymin
            data['xmax'][i] = self.bounds[i].xmax
            data['ymax'][i] = self.bounds[i].ymax

        # Find overlapping bounding boxes.
        x_overlap = np.logical_and(
            data['xmin'][:,np.newaxis] <= data['xmax'][np.newaxis,:],
            data['ymin'][:,np.newaxis] <= data['ymax'][np.newaxis,:])
        y_overlap = np.logical_and(
            data['xmax'][:,np.newaxis] >= data['xmin'][np.newaxis,:],
            data['ymax'][:,np.newaxis] >= data['ymin'][np.newaxis,:])
        overlapping_bounds = x_overlap & y_overlap

        # Calculate isolated galaxy quantities and identify overlapping groups.
        # At this stage, we use small integers for grp_id values, but these are later
        # reset to be the db_id of each group's leader.
        data['grp_id'] = np.arange(num_galaxies)
        for index in range(num_galaxies):
            trace('index %d' % index)
            model,stamps,bounds = self.models[index],self.stamps[index],self.bounds[index]
            data['db_id'][index] = model.identifier
            data['dx'][index] = model.dx_arcsecs
            data['dy'][index] = model.dy_arcsecs
            data['z'][index] = model.redshift
            data['ab_mag'][index] = model.ab_magnitude
            data['ri_color'][index] = model.ri_color
            data['flux'][index] = model.model.flux
            # Is this galaxy's centroid visible in the survey image?
            data['visible'][index] = 1 if self.survey.image.bounds.includes(bounds.center.x, bounds.center.y) else 0
            if(getattr(model,'disk_fraction',None)!=None):
                data['f_disk'][index] = model.disk_fraction
                data['f_bulge'][index] = model.bulge_fraction
                # Calculate this galaxy's sizes and shapes from its second-moments tensor.
                sigma_m,sigma_p,a,b,beta,e1,e2 = descwl.model.moments_size_and_shape(
                    model.second_moments)
                # Save values to the analysis results.
                data['sigma_m'][index] = sigma_m
                data['sigma_p'][index] = sigma_p
                data['a'][index] = a
                data['b'][index] = b
                data['e1'][index] = e1
                data['e2'][index] = e2
                data['beta'][index] = beta
                # Re-calculate sizes and shapes with the PSF second moments added.
                sigma_m_psf,sigma_p_psf,a_psf,b_psf,beta_psf,e1_psf,e2_psf = descwl.model.moments_size_and_shape(
                    model.second_moments + self.survey.psf_second_moments)
                # Save the PSF-convolved sigma(-) since this can be directly compared with the HSM size.
                data['psf_sigm'][index] = sigma_m_psf
            else:
                data['f_disk'][index] = 0
                data['f_bulge'][index] = 0
                data['sigma_m'][index] = 0
                data['sigma_p'][index] = 0
                data['a'][index] = 0
                data['b'][index] = 0
                data['e1'][index] = 0
                data['e2'][index] = 0
                data['beta'][index] = 0
                # Re-calculate sizes and shapes with the PSF second moments added.
                sigma_m_psf,sigma_p_psf,a_psf,b_psf,beta_psf,e1_psf,e2_psf = descwl.model.moments_size_and_shape(
                    self.survey.psf_second_moments)
                # Save the PSF-convolved sigma(-) since this can be directly compared with the HSM size.
                data['psf_sigm'][index] = sigma_m_psf
            # Loop over earlier galaxies with overlapping bounding boxes.
            for pre_index in np.arange(index)[overlapping_bounds[index,:index]]:
                pre_bounds = self.bounds[pre_index]
                overlap = bounds & pre_bounds
                assert overlap.area() > 0
                # Are there any overlapping pixels with non-zero flux from both sources?
                pre_stamps = self.stamps[pre_index]
                overlap_flux_product = np.sum(
                    stamps[0,
                        overlap.ymin-bounds.ymin:overlap.ymax-bounds.ymin+1,
                        overlap.xmin-bounds.xmin:overlap.xmax-bounds.xmin+1]*
                    pre_stamps[0,
                        overlap.ymin-pre_bounds.ymin:overlap.ymax-pre_bounds.ymin+1,
                        overlap.xmin-pre_bounds.xmin:overlap.xmax-pre_bounds.xmin+1])
                if overlap_flux_product == 0:
                    continue
                # Decide which group joins the other.
                grp_id_new = min(data['grp_id'][index],data['grp_id'][pre_index])
                grp_id_old = max(data['grp_id'][index],data['grp_id'][pre_index])
                # Reassign all galaxies in grp_id_old to grp_id_new.
                data['grp_id'][data['grp_id'] == grp_id_old] = grp_id_new

        # How many groups do we have?
        grp_id_set = set(data['grp_id'])
        num_groups = len(grp_id_set)
        if verbose:
            print('Simulated %d galaxies in %d overlap groups.' % (num_galaxies,num_groups))

        # Initialize our results object so we can use its methods (but be careful not
        # to use a method that needs something in table that we have not filled in yet).
        table = astropy.table.Table(data, copy = False)
        num_slices, h, w = self.stamps[0].shape
        results = OverlapResults(self.survey, table, self.stamps, self.bounds, num_slices, self.is_star)


        sky = self.survey.mean_sky_level
        dflux_index = results.slice_labels.index('dflux')
        ds_index = results.slice_labels.index('ds')
        dg1_index = results.slice_labels.index('dg1')
        dg2_index = results.slice_labels.index('dg2')
        dx_index = results.slice_labels.index('dx')
        dy_index = results.slice_labels.index('dy')
        # Loop over groups to calculate pixel-level quantities.
        for i,grp_id in enumerate(grp_id_set):
            trace('grp_id %d is %d of %d' % (grp_id,i,len(grp_id_set)))
            grp_members = (data['grp_id'] == grp_id)
            grp_size = np.count_nonzero(grp_members)
            data['grp_size'][grp_members] = grp_size
            group_indices = np.arange(num_galaxies)[grp_members]
            group_image = results.get_subimage(group_indices)

            if not self.no_fisher:
                # Check that we have partial derivatives available.
                if num_slices not in (3, 6, 21):
                    raise RuntimeError('Missing required partial derivative images for Fisher matrix analysis.')

                (fisher,covariance,variance,correlation), cond_num_grp = results.get_matrices(group_indices, get_cond_num=True, equilibrate=self.equilibrate)
                if self.calculate_bias:
                    bias = results.get_bias(group_indices, covariance.copy())
            base = 0
            for index,galaxy in enumerate(group_indices):
                flux = data['flux'][galaxy]
                signal = results.get_stamp(galaxy)
                if self.add_noise: #add noise for hsm, if requested. 
                    sigaux = signal.copy()
                    generator = galsim.random.BaseDeviate(seed = 1)
                    noise = galsim.PoissonNoise(rng = generator, sky_level = self.survey.mean_sky_level)
                    sigaux.addNoise(noise)
                # Calculate this galaxy's purity.
                signal_plus_background = group_image[signal.bounds]
                data['purity'][galaxy] = (
                    np.sum(signal.array**2)/np.sum(signal.array*signal_plus_background.array))

                # Run the HSM analysis on this galaxy's stamp (ignoring overlaps).
                if not self.no_hsm:
                    try:
                        if self.add_noise:
                            hsm_results = galsim.hsm.EstimateShear(sigaux,self.survey.psf_image)
                        else:
                            hsm_results = galsim.hsm.EstimateShear(signal,self.survey.psf_image)
                        data['hsm_sigm'][galaxy] = hsm_results.moments_sigma*self.survey.pixel_scale
                        data['hsm_e1'][galaxy] = hsm_results.corrected_e1
                        data['hsm_e2'][galaxy] = hsm_results.corrected_e2
                    except RuntimeError as e:
                        # Usually "Unphysical situation: galaxy convolved with PSF is smaller than PSF!"
                        # due to truncation of a faint galaxy at the limiting isophote.  Try to just
                        # calculate the PSF-convolved size in this case.
                        try:
                            if self.add_noise:
                                hsm_results = galsim.hsm.FindAdaptiveMom(sigaux)
                                data['hsm_sigm'][galaxy] = hsm_results.moments_sigma*self.survey.pixel_scale
                            else:
                                hsm_results = galsim.hsm.FindAdaptiveMom(signal)
                                data['hsm_sigm'][galaxy] = hsm_results.moments_sigma*self.survey.pixel_scale
                        except RuntimeError as e:
                            print(str(e))

                # Calculate the SNR this galaxy would have without any overlaps and
                # assuming that we are in the sky-dominated limit.
                data['snr_sky'][galaxy] = np.sqrt(np.sum(signal.array**2)/sky)
                # Calculate this galaxy's SNR in various ways.
                
                if not self.no_fisher:
                    data['snr_grp'][galaxy] = flux*np.sqrt(fisher[base+dflux_index,base+dflux_index])
                    # Variances will be np.inf if this galaxy was dropped from the group for the
                    # covariance calculation, leading to snr_grpf = 0 and infinite errors on s,g1,g2.
                    data['snr_grpf'][galaxy] = flux/np.sqrt(variance[base+dflux_index])
                    if not self.is_star[galaxy]:
                        data['ds_grp'][galaxy] = np.sqrt(variance[base+ds_index])
                        data['dg1_grp'][galaxy] = np.sqrt(variance[base+dg1_index])
                        data['dg2_grp'][galaxy] = np.sqrt(variance[base+dg2_index])
                    data['cond_num_grp'][galaxy] = cond_num_grp #add calculated condition number for galaxy's group.

                    if self.calculate_bias and not self.is_star[galaxy]:
                        data['bias_f_grp'][galaxy] = bias[base+dflux_index]
                        data['bias_s_grp'][galaxy] = bias[base+ds_index]
                        data['bias_g1_grp'][galaxy] = bias[base+dg1_index]
                        data['bias_g2_grp'][galaxy] = bias[base+dg2_index]
                        data['bias_x_grp'][galaxy] = bias[base+dx_index]
                        data['bias_y_grp'][galaxy] = bias[base+dy_index]

                    if grp_size == 1:
                        data['snr_iso'][galaxy] = data['snr_grp'][galaxy]
                        data['snr_isof'][galaxy] = data['snr_grpf'][galaxy]
                        if not self.is_star[galaxy]:
                            data['ds'][galaxy] = data['ds_grp'][galaxy]
                            data['dg1'][galaxy] = data['dg1_grp'][galaxy]
                            data['dg2'][galaxy] = data['dg2_grp'][galaxy]
                        data['cond_num'][galaxy] = data['cond_num_grp'][galaxy]

                        if self.calculate_bias and not self.is_star[galaxy]:
                            data['bias_f'][galaxy] = data['bias_f_grp'][galaxy]
                            data['bias_s'][galaxy] = data['bias_s_grp'][galaxy]
                            data['bias_g1'][galaxy] = data['bias_g1_grp'][galaxy]
                            data['bias_g2'][galaxy] = data['bias_g2_grp'][galaxy]
                            data['bias_x'][galaxy] = data['bias_x_grp'][galaxy]
                            data['bias_y'][galaxy] = data['bias_y_grp'][galaxy]

                    else:
                        # Redo the Fisher matrix analysis but ignoring overlapping sources.
                        (iso_fisher,iso_covariance,iso_variance,iso_correlation), cond_num  = results.get_matrices([galaxy], get_cond_num=True, equilibrate=self.equilibrate)

                        # snr_iso and snr_isof will be zero if the Fisher matrix is not invertible or
                        # yields any negative variances. Errors on s,g1,g2 will be np.inf.
                        data['snr_iso'][galaxy] = flux*np.sqrt(iso_fisher[dflux_index,dflux_index])
                        data['snr_isof'][galaxy] = flux/np.sqrt(iso_variance[dflux_index])
                        if not self.is_star[galaxy]:
                            data['ds'][galaxy] = np.sqrt(iso_variance[ds_index])
                            data['dg1'][galaxy] = np.sqrt(iso_variance[dg1_index])
                            data['dg2'][galaxy] = np.sqrt(iso_variance[dg2_index])
                        data['cond_num'][galaxy] = cond_num

                        if self.calculate_bias and not self.is_star[galaxy]:
                            iso_bias = results.get_bias([galaxy], iso_covariance.copy())
                            data['bias_f'][galaxy] = iso_bias[dflux_index]
                            data['bias_s'][galaxy] = iso_bias[ds_index]
                            data['bias_g1'][galaxy] = iso_bias[dg1_index]
                            data['bias_g2'][galaxy] = iso_bias[dg2_index]
                            data['bias_x'][galaxy] = iso_bias[dx_index]
                            data['bias_y'][galaxy] = iso_bias[dy_index]
                if self.is_star[galaxy]:
                    base += 3
                else:
                    base += len(results.slice_labels)
            # Order group members by decreasing isolated S/N (if available), otherwise use snr_sky.
            #this assumes that snr_sky is close to snr_iso (although not necessarily the same.)
            if not self.no_fisher:
                sorted_indices = group_indices[np.argsort(data['snr_iso'][grp_members])[::-1]]
            else: 
                sorted_indices = group_indices[np.argsort(data['snr_sky'][grp_members])[::-1]]

            data['grp_rank'][sorted_indices] = np.arange(grp_size,dtype = np.int16)
            # Replace group ID with ID of galaxy with largest S/N.
            group_leader = data['db_id'][sorted_indices[0]]
            data['grp_id'][grp_members] = group_leader

            if not self.no_fisher and not self.no_lmfit:
                alpha = self.alpha
                data['g1_fit'][sorted_indices] = 0.
                data['g2_fit'][sorted_indices] = 0.

                if self.detection_threshold is None: #adjust detection threshold if not specified.
                    self.detection_threshold = 6. #LSST default. 
                    if self.survey.survey_name in ['HSC', 'DES']:
                        self.detection_threshold = 5.

                detected = (data['snr_grpf'][sorted_indices] > self.detection_threshold)
                if np.count_nonzero(detected) > 0 and grp_size > 1:
                    use_count = np.zeros(grp_size,dtype = int)
                    # Loop over galaxies in order of decreasing snr_iso.
                    for i1,g1 in enumerate(sorted_indices):
                        # Skip sources below our detection threshold, which instead are added
                        # to a detected object below (unless they only directly overlap other
                        # undetected sources).
                        if not detected[i1]:
                            continue
                        stamp1 = results.get_stamp(g1)
                        deblended = stamp1.copy()
                        # Loop over other galaxies in this group in order of decreasing snr_iso.
                        for i2,g2 in enumerate(sorted_indices):
                            if i1 == i2 or not overlapping_bounds[g1,g2]:
                                continue
                            stamp2 = results.get_stamp(g2)
                            bbox = stamp1.bounds & stamp2.bounds
                            assert bbox.area() > 0
                            if not detected[i2]:
                                # This object is below our detection threshold so add its full
                                # flux to the first detected source that it overlaps.
                                if use_count[i2] == 0:
                                    deblended[bbox] += stamp2[bbox]
                                    use_count[i2] += 1
                            else:
                                overlap = stamp1[bbox]*stamp2[bbox]/group_image[bbox]
                                # Re-assign a fraction of the overlapping flux in the deblended image.
                                if i1 < i2:
                                    deblended[bbox] -= alpha*overlap
                                else:
                                    deblended[bbox] += alpha*overlap
                        # Fit the deblended image of this galaxy.
                        use_count[i1] += 1
                        if(getattr(model,'disk_fraction',None)!=None):
                            try:
                                bestfit = self.fit_galaxies([g1],deblended)
                                data['g1_fit'][g1] = bestfit[0,4]
                                data['g2_fit'][g1] = bestfit[0,5]
                            except RuntimeError as e:
                                print(str(e))
                        else:
                            try:
                                bestfit = self.fit_stars([g1],deblended)
                                data['g1_fit'][g1] = 0
                                data['g2_fit'][g1] = 0
                            except RuntimeError as e:
                                print(str(e))

        trace('OverlapAnalyzer.finalize end')
        return results

    @staticmethod
    def add_args(parser):
        """Add command-line arguments for constructing a new :class:`Reader`.

        The added arguments are our constructor parameters with '_' replaced by '-' in the names.
        Note that constructor parameter defaults are specified here rather than in the constructor,
        so that they are included in command-line help.

        Args:
            parser(argparse.ArgumentParser): Arguments will be added to this parser object using its
                add_argument method.
        """
        parser.add_argument('--alpha', type = float, default = 1, metavar = 'ALPHA',
            help = 'Fraction of flux given to the other object. Values range from -1 (overlapping flux to faintest source) to +1 \
            (overlapping flux to brightest source)')
        parser.add_argument('--no-hsm', action='store_true', help='Skip HSM fitting')
        parser.add_argument('--add-lmfit', action='store_true', help='Perform LMFIT fitting')
        parser.add_argument('--add-noise', action='store_true', help='Add Noise for HSM fitting')
        parser.add_argument('--equilibrate', action='store_true', help='Whether to equilibrate the fisher matrices before inversion. This might reduce the matrices\' condition number and improve numerical stability.')
        parser.add_argument('--detection-threshold', default=None, type=float, help='Specify threshold in terms of snr_grpf which determines sources that are skipped. Sources below our detection threshold are instead added to the highest snr_iso detected object it overlaps with. Default value is 5 for HSC,DES and 6 for LSST or other surveys.')

    @classmethod
    def from_args(cls,args):
        """Create a new :class:`Reader` object from a set of arguments.

        Args:
            args(object): A set of arguments accessed as a :py:class:`dict` using the
                built-in :py:func:`vars` function. Any extra arguments beyond those defined
                in :func:`add_args` will be silently ignored.

        Returns:
            :class:`Reader`: A newly constructed Reader object.
        """
        # Look up the named constructor parameters.
        pnames = (inspect.getargspec(cls.__init__)).args[1:]
        # Get a dictionary of the arguments provided.
        args_dict = vars(args)
        # Filter the dictionary to only include constructor parameters.
        filtered_dict = { key:args_dict[key] for key in (set(pnames) & set(args_dict)) }
        return cls(**filtered_dict)
