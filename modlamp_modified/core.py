# -*- coding: utf-8 -*-
"""
.. currentmodule:: modlamp.core

.. moduleauthor:: modlab Alex Mueller ETH Zurich <alex.mueller@pharma.ethz.ch>

Core helper functions and classes for other modules. The two main classes are:

=============================    =======================================================================================
Class                            Characteristics
=============================    =======================================================================================
:py:class:`BaseSequence`         Base class inheriting to all sequence classes in the module :py:mod:`modlamp.sequences`
:py:class:`BaseDescriptor`       Base class inheriting to the two descriptor classes in :py:mod:`modlamp.descriptors`
=============================    =======================================================================================
"""

import os
import random
import re

import numpy as np
import pandas as pd
import collections
import operator
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle

__author__ = "Alex Müller, Gisela Gabernet"
__docformat__ = "restructuredtext en"


class BaseSequence(object):
    """Base class for sequence classes in the module :mod:`modlamp.sequences`.
    It contains amino acid probabilities for different sequence generation classes.
    
    The following amino acid probabilities are used: (extracted from the
    `APD3 <http://aps.unmc.edu/AP/statistic/statistic.php>`_, March 17, 2016)

    ===  ====    ======   =========    ==========
    AA   rand    AMP      AMPnoCM      randnoCM
    ===  ====    ======   =========    ==========
    A    0.05    0.0766   0.0812275    0.05555555
    C    0.05    0.071    0.0          0.0
    D    0.05    0.026    0.0306275    0.05555555
    E    0.05    0.0264   0.0310275    0.05555555
    F    0.05    0.0405   0.0451275    0.05555555
    G    0.05    0.1172   0.1218275    0.05555555
    H    0.05    0.021    0.0256275    0.05555555
    I    0.05    0.061    0.0656275    0.05555555
    K    0.05    0.0958   0.1004275    0.05555555
    L    0.05    0.0838   0.0884275    0.05555555
    M    0.05    0.0123   0.0          0.0
    N    0.05    0.0386   0.0432275    0.05555555
    P    0.05    0.0463   0.0509275    0.05555555
    Q    0.05    0.0251   0.0297275    0.05555555
    R    0.05    0.0545   0.0591275    0.05555555
    S    0.05    0.0613   0.0659275    0.05555555
    T    0.05    0.0455   0.0501275    0.05555555
    V    0.05    0.0572   0.0618275    0.05555555
    W    0.05    0.0155   0.0201275    0.05555555
    Y    0.05    0.0244   0.0290275    0.05555555
    ===  ====    ======   =========    ==========
    
    """

    def __init__(self, seqnum, lenmin=7, lenmax=28):
        """
        :param seqnum: number of sequences to generate
        :param lenmin: minimal length of the generated sequences
        :param lenmax: maximal length of the generated sequences
        :return: attributes :py:attr:`seqnum`, :py:attr:`lenmin` and :py:attr:`lenmax`.
        :Example:
        
        >>> b = BaseSequence(10, 7, 28)
        >>> b.seqnum
        10
        >>> b.lenmin
        7
        >>> b.lenmax
        28
        """
        self.sequences = list()
        self.names = list()
        self.lenmin = int(lenmin)
        self.lenmax = int(lenmax)
        self.seqnum = int(seqnum)

        # AA classes:
        self.AA_hyd = ['G', 'A', 'L', 'I', 'V']
        self.AA_basic = ['K', 'R']
        self.AA_acidic = ['D', 'E']
        self.AA_aroma = ['W', 'Y', 'F']
        self.AA_polar = ['S', 'T', 'Q', 'N']
        # AA labels:
        self.AAs = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        # AA probability from the APD3 database:
        self.prob_AMP = [0.0766, 0.071, 0.026, 0.0264, 0.0405, 0.1172, 0.021, 0.061, 0.0958, 0.0838, 0.0123, 0.0386,
                         0.0463, 0.0251, 0.0545, 0.0613, 0.0455, 0.0572, 0.0155, 0.0244]
        # AA probability from the APD2 database without Cys and Met (synthesis reasons)
        self.prob_AMPnoCM = [0.081228, 0., 0.030627, 0.031027, 0.045128, 0.121828, 0.025627, 0.065628, 0.100428,
                             0.088428, 0., 0.043228, 0.050928, 0.029728, 0.059128, 0.065927, 0.050128, 0.061828,
                             0.020128, 0.029028]
        # equal AA probabilities:
        self.prob = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                     0.05, 0.05, 0.05, 0.05]
        # equal AA probabilities but 0 for Cys and Met:
        self.prob_randnoCM = [0.05555555555, 0.0, 0.05555555555, 0.05555555555, 0.05555555555, 0.05555555555,
                              0.05555555555, 0.05555555555, 0.05555555555, 0.05555555555, 0.0, 0.05555555555,
                              0.05555555555, 0.05555555555, 0.05555555555, 0.05555555555, 0.05555555555, 0.05555555555,
                              0.05555555555, 0.05555555555]

        # AA probability from the linear CancerPPD peptides:
        self.prob_ACP = [0.14526966, 0., 0.00690031, 0.00780824, 0.06991102, 0.04957327, 0.01725077, 0.05647358,
                         0.27637552, 0.17759216, 0.00998729, 0.00798983, 0.01307427, 0.00381333, 0.02941711,
                         0.02651171, 0.0154349, 0.04013074, 0.0406755, 0.00581079]

        # AA probabilities for perfect amphipathic helix of different arc sizes
        self.prob_amphihel = [[0.04545455, 0., 0.04545454, 0.04545455, 0., 0.04545455, 0.04545455, 0., 0.25, 0., 0.,
                               0.04545454, 0.04545455, 0.04545454, 0.25, 0.04545454, 0.04545454, 0., 0., 0.04545454],
                              [0., 0., 0., 0., 0.16666667, 0., 0., 0.16666667, 0., 0.16666667, 0., 0., 0., 0., 0., 0.,
                               0., 0.16666667, 0.16666667, (1. - 0.16666667 * 5)]]

        # helical ACP AA probabilities, depending on the position of the AA in the helix.
        self.prob_ACPhel = np.array([[0.0483871, 0., 0., 0.0483871, 0.01612903, 0.12903226, 0.03225807, 0.09677419,
                                      0.19354839, 0.5, 0.0483871, 0.11290323, 0.1, 0.18518519, 0.07843137, 0.12,
                                      0.17073172, 0.16666667],
                                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.01612903, 0., 0., 0., 0., 0.,
                                      0.02439024,
                                      0.19444444],
                                     [0., 0.01612903, 0., 0.27419355, 0.01612903, 0., 0., 0.01612903, 0., 0., 0., 0.,
                                      0.,
                                      0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 0., 0., 0., 0.06451613, 0., 0.01612903, 0.0483871, 0.01612903, 0.,
                                      0.01851852, 0., 0., 0., 0.],
                                     [0.16129032, 0.0483871, 0.30645161, 0., 0.0483871, 0., 0., 0.01612903, 0.,
                                      0.01612903,
                                      0., 0.09677419, 0.06666667, 0.01851852, 0., 0.02, 0.14634146, 0.],
                                     [0.64516129, 0., 0.17741936, 0.14516129, 0., 0.01612903, 0.25806452, 0.11290323,
                                      0.06451613, 0.08064516, 0.22580645, 0.03225807, 0.06666667, 0.2037037, 0.1372549,
                                      0.1, 0., 0.05555556],
                                     [0., 0., 0., 0.01612903, 0., 0., 0.01612903, 0., 0.03225807, 0., 0., 0.20967742,
                                      0.,
                                      0., 0., 0.16, 0., 0.],
                                     [0.0483871, 0.11290323, 0.01612903, 0.08064516, 0.33870968, 0.27419355, 0.,
                                      0.0483871, 0.14516129, 0.06451613, 0.03225807, 0.06451613, 0.18333333, 0., 0.,
                                      0.1, 0.26829268, 0.],
                                     [0., 0.03225807, 0.01612903, 0.12903226, 0.12903226, 0., 0.38709677, 0.33870968,
                                      0.0483871, 0.03225807, 0.41935484, 0.08064516, 0., 0.03703704, 0.29411765,
                                      0.04, 0.02439024, 0.02777778],
                                     [0.0483871, 0.70967742, 0.12903226, 0.0483871, 0.09677419, 0.32258064, 0.20967742,
                                      0.06451613, 0.11290323, 0.06451613, 0.03225807, 0.03225807, 0.28333333,
                                      0.24074074,
                                      0.03921569, 0.28, 0.07317073, 0.22222222],
                                     [0., 0.01612903, 0.01612903, 0.0483871, 0.01612903, 0.03225807, 0., 0., 0., 0.,
                                      0., 0., 0.03333333, 0., 0.01960784, 0.02, 0., 0.],
                                     [0., 0.01612903, 0., 0., 0., 0., 0., 0., 0.01612903, 0., 0.03225807, 0., 0., 0.,
                                      0.01960784, 0.02, 0., 0.],
                                     [0., 0., 0.14516129, 0.01612903, 0.03225807, 0.01612903, 0., 0., 0., 0.,
                                      0.01612903, 0., 0., 0.12962963, 0.17647059, 0., 0., 0.],
                                     [0., 0., 0.01612903, 0.01612903, 0., 0., 0.01612903, 0., 0.01612903, 0., 0.,
                                      0.01612903, 0., 0.01851852, 0., 0., 0., 0.],
                                     [0., 0.01612903, 0.01612903, 0., 0.01612903, 0., 0.01612903, 0., 0.01612903,
                                      0.01612903, 0.01612903, 0.01612903, 0., 0.01851852, 0.01960784, 0., 0.04878049,
                                      0.],
                                     [0.01612903, 0., 0.01612903, 0.12903226, 0.03225807, 0.03225807, 0.0483871,
                                      0.17741936, 0., 0.03225807, 0.09677419, 0.0483871, 0.01666667, 0., 0.15686274,
                                      0.1, 0., 0.05555556],
                                     [0.01612903, 0.01612903, 0., 0.01612903, 0.0483871, 0.01612903, 0., 0.01612903, 0.,
                                      0.01612903, 0.01612903, 0.11290323, 0., 0.01851852, 0.03921569, 0.02, 0.,
                                      0.05555556],
                                     [0.01612903, 0.01612903, 0.01612903, 0.01612903, 0.20967742, 0.16129032,
                                      0.01612903,
                                      0.0483871, 0.33870968, 0.16129032, 0., 0.14516129, 0.25, 0.11111111, 0.01960784,
                                      0.02, 0.21951219, 0.22222222],
                                     [0., 0., 0.12903226, 0.01612903, 0., 0., 0., 0., 0.01612903, 0., 0., 0., 0., 0.,
                                      0.,
                                      0., 0.02439024, 0.],
                                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.01612903, 0., 0., 0., 0., 0., 0.]])

    def save_fasta(self, filename, names=False):
        """Method to save generated sequences in a ``.FASTA`` formatted file.

        :param filename: output filename in which the sequences from :py:attr:`sequences` are safed in fasta format.
        :param names: {bool} whether sequence names from :py:attr:`names` should be saved as sequence identifiers
        :return: a FASTA formatted file containing the generated sequences
        :Example:
        
        >>> b = BaseSequence(2)
        >>> b.sequences = ['KLLSLSLALDLLS', 'KLPERTVVNSSDF']
        >>> b.names = ['Sequence1', 'Sequence2']
        >>> b.save_fasta('/location/of/fasta/file.fasta', names=True)
        """
        if names:
            save_fasta(filename, self.sequences, self.names)
        else:
            save_fasta(filename, self.sequences)

    def mutate_AA(self, nr, prob):
        """Method to mutate with **prob** probability a **nr** of positions per sequence randomly.

        :param nr: number of mutations to perform per sequence
        :param prob: probability of mutating a sequence
        :return: mutated sequences in the attribute :py:attr:`sequences`.
        :Example:

        >>> b = BaseSequence(1)
        >>> b.sequences = ['IAKAGRAIIK']
        >>> b.mutate_AA(3, 1.)
        >>> b.sequences
        ['NAKAGRAWIK']
        """
        for s in range(len(self.sequences)):
            # mutate: yes or no? prob = mutation probability
            mutate = np.random.choice([1, 0], 1, p=[prob, 1 - float(prob)])
            if mutate == 1:
                seq = list(self.sequences[s])
                cnt = 0
                while cnt < nr:  # mutate "nr" AA
                    seq[random.choice(range(len(seq)))] = random.choice(self.AAs)
                    cnt += 1
                self.sequences[s] = ''.join(seq)

    def filter_duplicates(self):
        """Method to filter duplicates in the sequences from the class attribute :py:attr:`sequences`

        :return: filtered sequences list in the attribute :py:attr:`sequences` and corresponding names.
        :Example:
        
        >>> b = BaseSequence(4)
        >>> b.sequences = ['KLLKLLKKLLKLLK', 'KLLKLLKKLLKLLK', 'KLAKLAKKLAKLAK', 'KLAKLAKKLAKLAK']
        >>> b.filter_duplicates()
        >>> b.sequences
        ['KLLKLLKKLLKLLK', 'KLAKLAKKLAKLAK']

        .. versionadded:: v2.2.5
        """
        if not self.names:
            self.names = ['Seq_' + str(i) for i in range(len(self.sequences))]
        df = pd.DataFrame(list(zip(self.sequences, self.names)), columns=['Sequences', 'Names'])
        df = df.drop_duplicates('Sequences', 'first')  # keep first occurrence of duplicate
        self.sequences = list(df['Sequences'])
        self.names = list(df['Names'])

    def keep_natural_aa(self):
        """Method to filter out sequences that do not contain natural amino acids. If the sequence contains a character
        that is not in ``['A','C','D,'E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']``.

        :return: filtered sequence list in the attribute :py:attr:`sequences`. The other attributes are also filtered
            accordingly (if present).
        :Example:
        
        >>> b = BaseSequence(2)
        >>> b.sequences = ['BBBsdflUasUJfBJ', 'GLFDIVKKVVGALGSL']
        >>> b.keep_natural_aa()
        >>> b.sequences
        ['GLFDIVKKVVGALGSL']
        """
        natural_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                      'Y']

        seqs = []
        names = []

        for i, s in enumerate(self.sequences):
            seq = list(s.upper())
            if all(c in natural_aa for c in seq):
                seqs.append(s.upper())
                if hasattr(self, 'names') and self.names:
                    names.append(self.names[i])

        self.sequences = seqs
        self.names = names

    def filter_aa(self, amino_acids):
        """Method to filter out corresponding names and descriptor values of sequences with given amino acids in the
        argument list *aminoacids*.

        :param amino_acids: {list} amino acids to be filtered
        :return: filtered list of sequences names in the corresponding attributes.
        :Example:
        
        >>> b = BaseSequence(3)
        >>> b.sequences = ['AAALLLIIIKKK', 'CCEERRT', 'LLVVIIFFFQQ']
        >>> b.filter_aa(['C'])
        >>> b.sequences
        ['AAALLLIIIKKK', 'LLVVIIFFFQQ']
        """

        pattern = re.compile('|'.join(amino_acids))
        seqs = []
        names = []

        for i, s in enumerate(self.sequences):
            if not pattern.search(s):
                seqs.append(s)
                if hasattr(self, 'names') and self.names:
                    names.append(self.names[i])

        self.sequences = seqs
        self.names = names

    def clean(self):
        """Method to clean / clear / empty the attributes :py:attr:`sequences` and :py:attr:`names`.

        :return: freshly initialized, empty class attributes.
        """
        self.__init__(self.seqnum, self.lenmin, self.lenmax)


class BaseDescriptor(object):
    """
    Base class inheriting to both peptide descriptor classes :py:class:`modlamp.descriptors.GlobalDescriptor` and
    :py:class:`modlamp.descriptors.PeptideDescriptor`.
    """

    def __init__(self, seqs):
        """
        :param seqs: a ``.FASTA`` file with sequences, a list / array of sequences or a single sequence as string to
            calculate the descriptor values for.
        :return: initialized attributes :py:attr:`sequences` and :py:attr:`names`.
        :Example:

        >>> AMP = BaseDescriptor('KLLKLLKKLLKLLK','pepCATS')
        >>> AMP.sequences
        ['KLLKLLKKLLKLLK']
        >>> seqs = BaseDescriptor('/Path/to/file.fasta', 'eisenberg')  # load sequences from .fasta file
        >>> seqs.sequences
        ['AFDGHLKI','KKLQRSDLLRTK','KKLASCNNIPPR'...]
        """
        if type(seqs) == list and seqs[0].isupper():
            self.sequences = [s.strip() for s in seqs]
            self.names = []
        elif type(seqs) == np.ndarray and seqs[0].isupper():
            self.sequences = [s.strip() for s in seqs.tolist()]
            self.names = []
        elif type(seqs) == str and seqs.isupper():
            self.sequences = [seqs.strip()]
            self.names = []
        elif os.path.isfile(seqs):
            if seqs.endswith('.fasta'):  # read .fasta file
                self.sequences, self.names = read_fasta(seqs)
            elif seqs.endswith('.csv'):  # read .csv file with sequences every line
                with open(seqs) as f:
                    self.sequences = list()
                    cntr = 0
                    self.names = []
                    for line in f:
                        if line.isupper():
                            self.sequences.append(line.strip())
                            self.names.append('seq_' + str(cntr))
                            cntr += 1
            else:
                print("Sorry, currently only .fasta or .csv files can be read!")
        else:
            print("%s does not exist, is not a valid list of AA sequences or is not a valid sequence string" % seqs)

        self.descriptor = np.array([[]])
        self.target = np.array([], dtype='int')
        self.scaler = None
        self.featurenames = []

    def read_fasta(self, filename):
        """Method for loading sequences from a ``.FASTA`` formatted file into the attributes :py:attr:`sequences` and
        :py:attr:`names`.

        :param filename: {str} ``.FASTA`` file with sequences and headers to read
        :return: {list} sequences in the attribute :py:attr:`sequences` with corresponding sequence names in
            :py:attr:`names`.
        """
        self.sequences, self.names = read_fasta(filename)

    def save_fasta(self, filename, names=False):
        """Method for saving sequences from :py:attr:`sequences` to a ``.FASTA`` formatted file.

        :param filename: {str} filename of the output ``.FASTA`` file
        :param names: {bool} whether sequence names from self.names should be saved as sequence identifiers
        :return: a FASTA formatted file containing the generated sequences
        """
        if names:
            save_fasta(filename, self.sequences, self.names)
        else:
            save_fasta(filename, self.sequences)

    def count_aa(self, scale='relative', average=False, append=False):
        """Method for producing the amino acid distribution for the given sequences as a descriptor

        :param scale: {'absolute' or 'relative'} defines whether counts or frequencies are given for each AA
        :param average: {boolean} whether the averaged amino acid counts for all sequences should be returned
        :param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the
            attribute :py:attr:`descriptor`.
        :return: the amino acid distributions for every sequence individually in the attribute :py:attr:`descriptor`
        :Example:

        >>> AMP = PeptideDescriptor('ACDEFGHIKLMNPQRSTVWY') # aa_count() does not depend on the descriptor scale
        >>> AMP.count_aa()
        >>> AMP.descriptor
        array([[ 0.05,  0.05,  0.05,  0.05,  0.05,  0.05,  0.05,  0.05,  0.05, ... ]])
        >>> AMP.descriptor.shape
        (1, 20)

        .. seealso:: :py:func:`modlamp.core.count_aa()`
        """
        desc = list()
        for seq in self.sequences:
            od = count_aas(seq, scale)
            desc.append(list(od.values()))

        desc = np.array(desc)
        self.featurenames = list(od.keys())

        if append:
            self.descriptor = np.hstack((self.descriptor, desc))
        elif average:
            self.descriptor = np.mean(desc, axis=0)
        else:
            self.descriptor = desc

    def count_ngrams(self, n):
        """Method for producing n-grams of all sequences in self.sequences

        :param n: {int or list of ints} defines whether counts or frequencies are given for each AA
        :return: {dict} dictionary with n-grams as keys and their counts in the sequence as values in :py:attr:`descriptor`
        :Example:

        >>> D = PeptideDescriptor('GLLDFLSLAALSLDKLVKKGALS')
        >>> D.count_ngrams([2, 3])
        >>> D.descriptor
        {'LS': 3, 'LD': 2, 'LSL': 2, 'AL': 2, ..., 'LVK': 1}

        .. seealso:: :py:func:`modlamp.core.count_ngrams()`
        """
        ngrams = dict()
        for seq in self.sequences:
            d = count_ngrams(seq, n)
            for k, v in d.items():
                if k in ngrams.keys():
                    ngrams[k] += v
                else:
                    ngrams[k] = v
        self.descriptor = ngrams

    def feature_scaling(self, stype='standard', fit=True):
        """Method for feature scaling of the calculated descriptor matrix.

        :param stype: {'standard' or 'minmax'} type of scaling to be used
        :param fit: {boolean} defines whether the used scaler is first fitting on the data (True) or
            whether the already fitted scaler in :py:attr:`scaler` should be used to transform (False).
        :return: scaled descriptor values in :py:attr:`descriptor`
        :Example:

        >>> D.descriptor
        array([[0.155],[0.34],[0.16235294],[-0.08842105],[0.116]])
        >>> D.feature_scaling(type='minmax',fit=True)
        array([[0.56818182],[1.],[0.5853447],[0.],[0.47714988]])
        """
        if stype in ['standard', 'minmax']:
            if stype == 'standard':
                self.scaler = StandardScaler()
            elif stype == 'minmax':
                self.scaler = MinMaxScaler()

            if fit:
                self.descriptor = self.scaler.fit_transform(self.descriptor)
            else:
                self.descriptor = self.scaler.transform(self.descriptor)
        else:
            print("Unknown scaler type!\nAvailable: 'standard', 'minmax'")

    def feature_shuffle(self):
        """Method for shuffling feature columns randomly.

        :return: descriptor matrix with shuffled feature columns in :py:attr:`descriptor`
        :Example:

        >>> D.descriptor
        array([[0.80685625,167.05234375,39.56818125,-0.26338667,155.16888667,33.48778]])
        >>> D.feature_shuffle()
        array([[155.16888667,-0.26338667,167.05234375,0.80685625,39.56818125,33.48778]])
        """
        self.descriptor = shuffle(self.descriptor.transpose()).transpose()

    def sequence_order_shuffle(self):
        """Method for shuffling sequence order in the attribute :py:attr:`sequences`.

        :return: sequences in :py:attr:`sequences` with shuffled order in the list.
        :Example:

        >>> D.sequences
        ['LILRALKGAARALKVA','VKIAKIALKIIKGLG','VGVRLIKGIGRVARGAI','LRGLRGVIRGGKAIVRVGK','GGKLVRLIARIGKGV']
        >>> D.sequence_order_shuffle()
        >>> D.sequences
        ['VGVRLIKGIGRVARGAI','LILRALKGAARALKVA','LRGLRGVIRGGKAIVRVGK','GGKLVRLIARIGKGV','VKIAKIALKIIKGLG']
        """
        self.sequences = shuffle(self.sequences)

    def random_selection(self, num):
        """Method to randomly select a specified number of sequences (with names and descriptors if present) out of a given
        descriptor instance.

        :param num: {int} number of entries to be randomly selected
        :return: updated instance
        :Example:

        >>> h = Helices(7, 28, 100)
        >>> h.generate_helices()
        >>> desc = PeptideDescriptor(h.sequences, 'eisenberg')
        >>> desc.calculate_moment()
        >>> len(desc.sequences)
        100
        >>> len(desc.descriptor)
        100
        >>> desc.random_selection(10)
        >>> len(desc.descriptor)
        10
        >>> len(desc.descriptor)
        10

        .. versionadded:: v2.2.3
        """

        sel = np.random.choice(len(self.sequences), size=num, replace=False)
        self.sequences = np.array(self.sequences)[sel].tolist()
        if hasattr(self, 'descriptor') and self.descriptor.size:
            self.descriptor = self.descriptor[sel]
        if hasattr(self, 'names') and self.names:
            self.names = np.array(self.names)[sel].tolist()
        if hasattr(self, 'target') and self.target.size:
            self.target = self.target[sel]

    def minmax_selection(self, iterations, distmetric='euclidean', seed=0):
        """Method to select a specified number of sequences according to the minmax algorithm.

        :param iterations: {int} Number of sequences to retrieve.
        :param distmetric: Distance metric to calculate the distances between the sequences in descriptor space.
            Choose from 'euclidean' or 'minkowsky'.
        :param seed: {int} Set a random seed for numpy to pick the first sequence.
        :return: updated instance

        .. seealso:: **SciPy** http://docs.scipy.org/doc/scipy/reference/spatial.distance.html
        """

        # Storing M into pool, where selections get deleted
        pool = self.descriptor  # Store pool where selections get deleted
        minmaxidx = list()  # Store original indices of selections to return

        # Randomly selecting first peptide into the sele
        np.random.seed(seed)
        idx = int(np.random.random_integers(0, len(pool), 1))
        sele = pool[idx:idx + 1, :]
        minmaxidx.append(int(*np.where(np.all(self.descriptor == pool[idx:idx + 1, :], axis=1))))

        # Deleting peptide in selection from pool
        pool = np.delete(pool, idx, axis=0)

        for i in range(iterations - 1):
            # Calculating distance from sele to the rest of the peptides
            dist = distance.cdist(pool, sele, distmetric)

            # Choosing maximal distances for every sele instance
            maxidx = np.argmax(dist, axis=0)
            maxcols = np.max(dist, axis=0)

            # Choosing minimal distance among the maximal distances
            minmax = np.argmin(maxcols)
            maxidx = int(maxidx[minmax])

            # Adding it to selection and removing from pool
            sele = np.append(sele, pool[maxidx:maxidx + 1, :], axis=0)
            pool = np.delete(pool, maxidx, axis=0)
            minmaxidx.append(int(*np.where(np.all(self.descriptor == pool[maxidx:maxidx + 1, :], axis=1))))

        self.sequences = np.array(self.sequences)[minmaxidx].tolist()
        if hasattr(self, 'descriptor') and self.descriptor.size:
            self.descriptor = self.descriptor[minmaxidx]
        if hasattr(self, 'names') and self.names:
            self.names = np.array(self.names)[minmaxidx].tolist()
        if hasattr(self, 'target') and self.target.size:
            self.target = self.descriptor[minmaxidx]

    def filter_sequences(self, sequences):
        """Method to filter out entries for given sequences in *sequences* out of a descriptor instance. All
        corresponding attribute values of these sequences (e.g. in :py:attr:`descriptor`, :py:attr:`name`) are deleted
        as well. The method returns an updated descriptor instance.

        :param sequences: {list} sequences to be filtered out of the whole instance, including corresponding data
        :return: updated instance without filtered sequences
        :Example:

        >>> sequences = ['KLLKLLKKLLKLLK', 'ACDEFGHIK', 'GLFDIVKKVV', 'GLFDIVKKVVGALG', 'GLFDIVKKVVGALGSL']
        >>> desc = PeptideDescriptor(sequences, 'pepcats')
        >>> desc.calculate_crosscorr(7)
        >>> len(desc.descriptor)
        5
        >>> desc.filter_sequences('KLLKLLKKLLKLLK')
        >>> len(desc.descriptor)
        4
        >>> desc.sequences
        ['ACDEFGHIK', 'GLFDIVKKVV', 'GLFDIVKKVVGALG', 'GLFDIVKKVVGALGSL']
        """
        indices = list()
        if isinstance(sequences, str):  # check if sequences is only one sequence string and convert it to a list
            sequences = [sequences]
        for s in sequences:  # get indices of queried sequences
            indices.append(self.sequences.index(s))

        self.sequences = np.delete(np.array(self.sequences), indices, 0).tolist()
        if hasattr(self, 'descriptor') and self.descriptor.size:
            self.descriptor = np.delete(self.descriptor, indices, 0)
        if hasattr(self, 'names') and self.names:
            self.names = np.delete(np.array(self.names), indices, 0).tolist()
        if hasattr(self, 'target') and self.target.size:
            self.target = np.delete(self.target, indices, 0)

    def filter_values(self, values, operator='=='):
        """Method to filter the descriptor matrix in the attribute :py:attr:`descriptor` for a given list of values (same
        size as the number of features in the descriptor matrix!) The operator option tells the method whether to
        filter for values equal, lower, higher ect. to the given values in the *values* array.

        :param values: {list} values to filter the attribute :py:attr:`descriptor` for
        :param operator: {str} filter criterion, available the operators ``==``, ``<``, ``>``, ``<=``and ``>=``.
        :return: descriptor matrix and updated sequences containing only entries with descriptor values given in
            *values* in the corresponding attributes.
        :Example:

        >>> desc.descriptor  # desc = BaseDescriptor instance
        array([[ 0.7666517 ],
               [ 0.38373498]])
        >>> desc.filter_values([0.5], '<')
        >>> desc.descriptor
        array([[ 0.38373498]])
        """
        dim = self.descriptor.shape[1]
        for d in range(dim):  # for all the features in self.descriptor
            if operator == '==':
                indices = np.where(self.descriptor[:, d] == values[d])[0]
            elif operator == '<':
                indices = np.where(self.descriptor[:, d] < values[d])[0]
            elif operator == '>':
                indices = np.where(self.descriptor[:, d] > values[d])[0]
            elif operator == '<=':
                indices = np.where(self.descriptor[:, d] <= values[d])[0]
            elif operator == '>=':
                indices = np.where(self.descriptor[:, d] >= values[d])[0]
            else:
                raise KeyError('available operators: ``==``, ``<``, ``>``, ``<=``and ``>=``')

            # filter descriptor matrix, sequence list and names list according to obtained indices
            self.sequences = np.array(self.sequences)[indices].tolist()
            if hasattr(self, 'descriptor') and self.descriptor.size:
                self.descriptor = self.descriptor[indices]
            if hasattr(self, 'names') and self.names:
                self.names = np.array(self.names)[indices].tolist()
            if hasattr(self, 'target') and self.target.size:
                self.target = self.target[indices]

    def filter_aa(self, amino_acids):
        """Method to filter out corresponding names and descriptor values of sequences with given amino acids in the
        argument list *aminoacids*.

        :param amino_acids: list of amino acids to be filtered
        :return: filtered list of sequences, descriptor values, target values and names in the corresponding attributes.
        :Example:

        >>> b = BaseSequence(3)
        >>> b.sequences = ['AAALLLIIIKKK', 'CCEERRT', 'LLVVIIFFFQQ']
        >>> b.filter_aa(['C'])
        >>> b.sequences
        ['AAALLLIIIKKK', 'LLVVIIFFFQQ']
        """

        pattern = re.compile('|'.join(amino_acids))
        seqs = []
        desc = []
        names = []
        target = []

        for i, s in enumerate(self.sequences):
            if not pattern.search(s):
                seqs.append(s)
                if hasattr(self, 'descriptor') and self.descriptor.size:
                    desc.append(self.descriptor[i])
                if hasattr(self, 'names') and self.names:
                    names.append(self.names[i])
                if hasattr(self, 'target') and self.target.size:
                    target.append(self.target[i])

        self.sequences = seqs
        self.names = names
        self.descriptor = np.array(desc)
        self.target = np.array(target, dtype='int')

    def filter_duplicates(self):
        """Method to filter duplicates in the sequences from the class attribute :py:attr:`sequences`

        :return: filtered sequences list in the attribute :py:attr:`sequences` and corresponding names.
        :Example:

        >>> b = BaseDescriptor(['KLLKLLKKLLKLLK', 'KLLKLLKKLLKLLK', 'KLAKLAKKLAKLAK', 'KLAKLAKKLAKLAK'])
        >>> b.filter_duplicates()
        >>> b.sequences
        ['KLLKLLKKLLKLLK', 'KLAKLAKKLAKLAK']

        .. versionadded:: v2.2.5
        """
        if not self.names:
            self.names = ['Seq_' + str(i) for i in range(len(self.sequences))]
        if not self.target:
            self.target = [0] * len(self.sequences)
        if not self.descriptor:
            self.descriptor = np.zeros(len(self.sequences))
        df = pd.DataFrame(np.array([self.sequences, self.names, self.descriptor, self.target]).T,
                          columns=['Sequences', 'Names', 'Descriptor', 'Target'])
        df = df.drop_duplicates('Sequences', 'first')  # keep first occurrence of duplicate
        self.sequences = list(df['Sequences'])
        self.names = list(df['Names'])
        self.descriptor = df['Descriptor'].get_values()
        self.target = df['Target'].get_values()

    def keep_natural_aa(self):
        """Method to filter out sequences that do not contain natural amino acids. If the sequence contains a character
        that is not in ['A','C','D,'E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'].

        :return: filtered sequence list in the attribute :py:attr:`sequences`. The other attributes are also filtered
            accordingly (if present).
        :Example:

        >>> b = BaseSequence(2)
        >>> b.sequences = ['BBBsdflUasUJfBJ', 'GLFDIVKKVVGALGSL']
        >>> b.keep_natural_aa()
        >>> b.sequences
        ['GLFDIVKKVVGALGSL']
        """

        natural_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                      'Y']

        seqs = []
        desc = []
        names = []
        target = []

        for i, s in enumerate(self.sequences):
            seq = list(s.upper())
            if all(c in natural_aa for c in seq):
                seqs.append(s.upper())
                if hasattr(self, 'descriptor') and self.descriptor.size:
                    desc.append(self.descriptor[i])
                if hasattr(self, 'names') and self.names:
                    names.append(self.names[i])
                if hasattr(self, 'target') and self.target.size:
                    target.append(self.target[i])

        self.sequences = seqs
        self.names = names
        self.descriptor = np.array(desc)
        self.target = np.array(target, dtype='int')

    def load_descriptordata(self, filename, delimiter=",", targets=False, skip_header=0):
        """Method to load any data file with sequences and descriptor values and save it to a new insatnce of the
        class :class:`modlamp.descriptors.PeptideDescriptor`.

        .. note:: Headers are not considered. To skip initial lines in the file, use the *skip_header* option.

        :param filename: {str} filename of the data file to be loaded
        :param delimiter: {str} column delimiter
        :param targets: {boolean} whether last column in the file contains a target class vector
        :param skip_header: {int} number of initial lines to skip in the file
        :return: loaded sequences, descriptor values and targets in the corresponding attributes.
        """
        data = np.genfromtxt(filename, delimiter=delimiter, skip_header=skip_header)
        data = data[:, 1:]  # skip sequences as they are "nan" when read as float
        seqs = np.genfromtxt(filename, delimiter=delimiter, dtype="str")
        seqs = seqs[:, 0]
        if targets:
            self.target = np.array(data[:, -1], dtype='int')
        self.sequences = seqs
        self.descriptor = data

    def save_descriptor(self, filename, delimiter=',', targets=None, header=None):
        """Method to save the descriptor values to a .csv/.txt file

        :param filename: filename of the output file
        :param delimiter: column delimiter
        :param targets: target class vector to be added to descriptor (same length as :py:attr:`sequences`)
        :param header: {str} header to be written at the beginning of the file (if ``None``: feature names are taken)
        :return: output file with peptide names and descriptor values
        """
        seqs = np.array(self.sequences, dtype='|S80')[:, np.newaxis]
        ids = np.array(self.names, dtype='|S80')[:, np.newaxis]
        if ids.shape == seqs.shape:
            names = np.hstack((ids, seqs))
        else:
            names = seqs
        if targets and len(targets) == len(self.sequences):
            target = np.array(targets)[:, np.newaxis]
            data = np.hstack((names, self.descriptor, target))
        else:
            data = np.hstack((names, self.descriptor))
        if not header:
            featurenames = [['Sequence']] + self.featurenames
            header = ', '.join([f[0] for f in featurenames])
        np.savetxt(filename, data, delimiter=delimiter, fmt='%s', header=header)


def load_scale(scalename):
    """Method to load scale values for a given amino acid scale

    :param scalename: amino acid scale name, for available scales see the
        :class:`modlamp.descriptors.PeptideDescriptor()` documentation.
    :return: amino acid scale values in dictionary format.
    """
    # predefined amino acid scales dictionary
    scales = {
        'aasi': {'A': [1.89], 'C': [1.73], 'D': [3.13], 'E': [3.14], 'F': [1.53], 'G': [2.67], 'H': [3], 'I': [1.97],
                 'K': [2.28], 'L': [1.74], 'M': [2.5], 'N': [2.33], 'P': [0.22], 'Q': [3.05], 'R': [1.91], 'S': [2.14],
                 'T': [2.18], 'V': [2.37], 'W': [2], 'Y': [2.01]},
        'abhprk': {'A': [0, 0, 0, 0, 0, 0], 'C': [0, 0, 0, 0, 0, 0], 'D': [1, 0, 0, 1, 0, 0], 'E': [1, 0, 0, 1, 0, 0],
                   'F': [0, 0, 1, 0, 1, 0], 'G': [0, 0, 0, 0, 0, 0], 'H': [0, 0, 0, 1, 1, 0], 'I': [0, 0, 1, 0, 0, 0],
                   'K': [0, 1, 0, 1, 0, 0], 'L': [0, 0, 1, 0, 0, 0], 'M': [0, 0, 1, 0, 0, 0], 'N': [0, 0, 0, 1, 0, 0],
                   'P': [0, 0, 0, 0, 0, 1], 'Q': [0, 0, 0, 1, 0, 0], 'R': [0, 1, 0, 1, 0, 0], 'S': [0, 0, 0, 1, 0, 0],
                   'T': [0, 0, 0, 1, 0, 0], 'V': [0, 0, 1, 0, 0, 0], 'W': [0, 0, 1, 0, 1, 0], 'Y': [0, 0, 0, 1, 1, 0]},
        'argos': {'I': [0.77], 'F': [1.2], 'V': [0.14], 'L': [2.3], 'W': [0.07], 'M': [2.3], 'A': [0.64], 'G': [-0.48],
                  'C': [0.25], 'Y': [-0.41], 'P': [-0.31], 'T': [-0.13], 'S': [-0.25], 'H': [-0.87], 'E': [-0.94],
                  'N': [-0.89], 'Q': [-0.61], 'D': [-1], 'K': [-1], 'R': [-0.68]},
        'bulkiness': {'A': [0.443], 'C': [0.551], 'D': [0.453], 'E': [0.557], 'F': [0.898], 'G': [0], 'H': [0.563],
                      'I': [0.985], 'K': [0.674], 'L': [0.985], 'M': [0.703], 'N': [0.516], 'P': [0.768], 'Q': [0.605],
                      'R': [0.596], 'S': [0.332], 'T': [0.677], 'V': [0.995], 'W': [1], 'Y': [0.801]},
        'charge_phys': {'A': [0.], 'C': [-.1], 'D': [-1.], 'E': [-1.], 'F': [0.], 'G': [0.], 'H': [0.1],
                        'I': [0.], 'K': [1.], 'L': [0.], 'M': [0.], 'N': [0.], 'P': [0.], 'Q': [0.],
                        'R': [1.], 'S': [0.], 'T': [0.], 'V': [0.], 'W': [0.], 'Y': [0.]},
        'charge_acid': {'A': [0.], 'C': [-.1], 'D': [-1.], 'E': [-1.], 'F': [0.], 'G': [0.], 'H': [1.],
                        'I': [0.], 'K': [1.], 'L': [0.], 'M': [0.], 'N': [0.], 'P': [0.], 'Q': [0.],
                        'R': [1.], 'S': [0.], 'T': [0.], 'V': [0.], 'W': [0.], 'Y': [0.]},
        'cougar': {'A': [0.25, 0.62, 1.89], 'C': [0.208, 0.29, 1.73], 'D': [0.875, -0.9, 3.13],
                   'E': [0.833, -0.74, 3.14], 'F': [0.042, 1.2, 1.53], 'G': [1, 0.48, 2.67], 'H': [0.083, -0.4, 3],
                   'I': [0.667, 1.4, 1.97], 'K': [0.708, -1.5, 2.28], 'L': [0.292, 1.1, 1.74], 'M': [0, 0.64, 2.5],
                   'N': [0.667, -0.78, 2.33], 'P': [0.875, 0.12, 0.22], 'Q': [0.792, -0.85, 3.05],
                   'R': [0.958, -2.5, 1.91], 'S': [0.875, -0.18, 2.14], 'T': [0.583, -0.05, 2.18],
                   'V': [0.375, 1.1, 2.37], 'W': [0.042, 0.81, 2], 'Y': [0.5, 0.26, 2.01]},
        'eisenberg': {'I': [1.4], 'F': [1.2], 'V': [1.1], 'L': [1.1], 'W': [0.81], 'M': [0.64], 'A': [0.62],
                      'G': [0.48], 'C': [0.29], 'Y': [0.26], 'P': [0.12], 'T': [-0.05], 'S': [-0.18], 'H': [-0.4],
                      'E': [-0.74], 'N': [-0.78], 'Q': [-0.85], 'D': [-0.9], 'K': [-1.5], 'R': [-2.5]},
        'ez': {'A': [-0.29, 10.22, 4.67], 'C': [0.95, 13.69, 5.77], 'D': [1.19, 14.25, 8.98], 'E': [1.3, 14.66, 4.16],
               'F': [-0.8, 19.67, 7.12], 'G': [-0.01, 13.86, 6], 'H': [0.75, 12.26, 2.77], 'I': [-0.56, 14.34, 10.69],
               'K': [1.66, 11.11, 2.09], 'L': [-0.64, 17.34, 8.61], 'M': [-0.28, 18.04, 7.13], 'N': [0.89, 12.78, 6.28],
               'P': [0.83, 18.09, 3.53], 'Q': [1.21, 10.46, 2.59], 'R': [1.55, 9.34, 4.68], 'S': [0.1, 13.86, 6],
               'T': [0.01, 13.86, 6], 'V': [-0.47, 11.35, 4.97], 'W': [-0.85, 11.65, 7.2], 'Y': [-0.42, 13.04, 6.2]},
        'flexibility': {'A': [0.25], 'C': [0.208], 'D': [0.875], 'E': [0.833], 'F': [0.042], 'G': [1], 'H': [0.083],
                        'I': [0.667], 'K': [0.708], 'L': [0.292], 'M': [0.], 'N': [0.667], 'P': [0.875], 'Q': [0.792],
                        'R': [0.958], 'S': [0.875], 'T': [0.583], 'V': [0.375], 'W': [0.042], 'Y': [0.5]},
        'grantham': {'A': [0, 8.1, 31], 'C': [2.75, 5.5, 55], 'D': [1.38, 13.0, 54], 'E': [0.92, 12.3, 83],
                     'F': [0, 5.2, 132], 'G': [0.74, 9.0, 3], 'H': [0.58, 10.4, 96], 'I': [0, 5.2, 111],
                     'K': [0.33, 11.3, 119], 'L': [0, 4.9, 111], 'M': [0, 5.7, 105], 'N': [1.33, 11.6, 56],
                     'P': [0.39, 8.0, 32.5], 'Q': [0.89, 10.5, 85], 'R': [0.65, 10.5, 124], 'S': [1.42, 9.2, 32],
                     'T': [0.71, 8.6, 61], 'V': [0, 5.9, 84], 'W': [0.13, 5.4, 170], 'Y': [0.20, 6.2, 136]},
        'gravy': {'I': [4.5], 'V': [4.2], 'L': [3.8], 'F': [2.8], 'C': [2.5], 'M': [1.9], 'A': [1.8], 'G': [-0.4],
                  'T': [-0.7], 'W': [-0.9], 'S': [-0.8], 'Y': [-1.3], 'P': [-1.6], 'H': [-3.2], 'E': [-3.5],
                  'Q': [-3.5], 'D': [-3.5], 'N': [-3.5], 'K': [-3.9], 'R': [-4.5]},
        'hopp-woods': {'A': [-0.5], 'C': [-1], 'D': [3], 'E': [3], 'F': [-2.5], 'G': [0], 'H': [-0.5], 'I': [-1.8],
                       'K': [3], 'L': [-1.8], 'M': [-1.3], 'N': [0.2], 'P': [0], 'Q': [0.2], 'R': [3], 'S': [0.3],
                       'T': [-0.4], 'V': [-1.5], 'W': [-3.4], 'Y': [-2.3]},
        'isaeci': {'A': [62.9, 0.05], 'C': [78.51, 0.15], 'D': [18.46, 1.25], 'E': [30.19, 1.31], 'F': [189.42, 0.14],
                   'G': [19.93, 0.02], 'H': [87.38, 0.56], 'I': [149.77, 0.09], 'K': [102.78, 0.53], 'L': [154.35, 0.1],
                   'M': [132.22, 0.34], 'N': [19.53, 1.36], 'P': [122.35, 0.16], 'Q': [17.87, 1.31], 'R': [52.98, 1.69],
                   'S': [19.75, 0.56], 'T': [59.44, 0.65], 'V': [120.91, 0.07], 'W': [179.16, 1.08],
                   'Y': [132.16, 0.72]},
        'janin': {'I': [1.2], 'F': [0.87], 'V': [1], 'L': [0.87], 'W': [0.59], 'M': [0.73], 'A': [0.59], 'G': [0.59],
                  'C': [1.4], 'Y': [-0.4], 'P': [-0.26], 'T': [-0.12], 'S': [0.02], 'H': [0.02], 'E': [-0.83],
                  'N': [-0.55], 'Q': [-0.83], 'D': [-0.69], 'K': [-2.4], 'R': [-1.8]},
        'kytedoolittle': {'I': [1.7], 'F': [1.1], 'V': [1.6], 'L': [1.4], 'W': [-0.14], 'M': [0.8], 'A': [0.77],
                          'G': [0.03], 'C': [1], 'Y': [-0.27], 'P': [-0.37], 'T': [-0.07], 'S': [-0.1], 'H': [-0.91],
                          'E': [-1], 'N': [-1], 'Q': [-1], 'D': [-1], 'K': [-1.1], 'R': [-1.3]},
        'levitt_alpha': {'A': [1.29], 'C': [1.11], 'D': [1.04], 'E': [1.44], 'F': [1.07], 'G': [0.56], 'H': [1.22],
                         'I': [0.97], 'K': [1.23], 'L': [1.3], 'M': [1.47], 'N': [0.9], 'P': [0.52], 'Q': [1.27],
                         'R': [0.96], 'S': [0.82], 'T': [0.82], 'V': [0.91], 'W': [0.99], 'Y': [0.72]},
        'mss': {'A': [13.02], 'C': [23.7067], 'D': [22.02], 'E': [20.0233], 'F': [23.5288], 'G': [1.01], 'H': [23.5283],
                'I': [22.3611], 'K': [18.9756], 'L': [19.6944], 'M': [21.92], 'N': [21.8567], 'P': [19.0242],
                'Q': [19.9689], 'R': [19.0434], 'S': [18.3533], 'T': [22.3567], 'V': [21.0267], 'W': [26.1975],
                'Y': [24.1954]},
        'msw': {'A': [-0.73, 0.2, -0.62], 'C': [-0.66, 0.26, -0.27], 'D': [0.11, -1, -0.96], 'E': [0.24, -0.39, -0.04],
                'F': [0.76, 0.85, -0.34], 'G': [-0.31, -0.28, -0.75], 'H': [0.84, 0.67, -0.78],
                'I': [-0.91, 0.83, -0.25], 'K': [-0.51, 0.08, 0.6], 'L': [-0.74, 0.72, -0.16], 'M': [-0.7, 1, -0.32],
                'N': [0.14, 0.2, -0.66], 'P': [-0.43, 0.73, -0.6], 'Q': [0.3, 1, -0.3], 'R': [-0.22, 0.27, 1],
                'S': [-0.8, 0.61, -1], 'T': [-0.58, 0.85, -0.89], 'V': [-1, 0.79, -0.58], 'W': [1, 0.98, -0.47],
                'Y': [0.97, 0.66, -0.16]},
        'pepcats': {'A': [1, 0, 0, 0, 0, 0], 'C': [1, 0, 1, 1, 0, 0], 'D': [0, 0, 1, 0, 0, 1], 'E': [0, 0, 1, 0, 0, 1],
                    'F': [1, 1, 0, 0, 0, 0], 'G': [0, 0, 0, 0, 0, 0], 'H': [1, 1, 0, 1, 1, 0], 'I': [1, 0, 0, 0, 0, 0],
                    'K': [1, 0, 0, 1, 1, 0], 'L': [1, 0, 0, 0, 0, 0], 'M': [1, 0, 1, 0, 0, 0], 'N': [0, 0, 1, 1, 0, 0],
                    'P': [1, 0, 0, 0, 0, 0], 'Q': [0, 0, 1, 1, 0, 0], 'R': [1, 0, 0, 1, 1, 0], 'S': [0, 0, 1, 1, 0, 0],
                    'T': [0, 0, 1, 1, 0, 0], 'V': [1, 0, 0, 0, 0, 0], 'W': [1, 1, 0, 1, 0, 0], 'Y': [1, 1, 1, 1, 0, 0]},
        'peparc': {'A': [1, 0, 0, 0, 0], 'C': [0, 1, 0, 0, 0], 'D': [0, 1, 0, 1, 0], 'E': [0, 1, 0, 1, 0],
                   'F': [1, 0, 0, 0, 0], 'G': [0, 0, 0, 0, 0], 'H': [0, 1, 1, 0, 0], 'I': [1, 0, 0, 0, 0],
                   'K': [0, 1, 1, 0, 0], 'L': [1, 0, 0, 0, 0], 'M': [1, 0, 0, 0, 0], 'N': [0, 1, 0, 0, 0],
                   'P': [0, 0, 0, 0, 1], 'Q': [0, 1, 0, 0, 0], 'R': [0, 1, 1, 0, 0], 'S': [0, 1, 0, 0, 0],
                   'T': [0, 1, 0, 0, 0], 'V': [1, 0, 0, 0, 0], 'W': [1, 0, 0, 0, 0], 'Y': [1, 0, 0, 0, 0]},
        'polarity': {'A': [0.395], 'C': [0.074], 'D': [1.], 'E': [0.914], 'F': [0.037], 'G': [0.506], 'H': [0.679],
                     'I': [0.037], 'K': [0.79], 'L': [0.], 'M': [0.099], 'N': [0.827], 'P': [0.383], 'Q': [0.691],
                     'R': [0.691], 'S': [0.531], 'T': [0.457], 'V': [0.123], 'W': [0.062], 'Y': [0.16]},
        'ppcali': {
            'A': [0.070781, 0.036271, 2.042, 0.083272, 0.69089, 0.15948, -0.80893, 0.24698, 0.86525, 0.68563, -0.24665,
                  0.61314, -0.53343, -0.50878, -1.3646, 2.2679, -1.5644, -0.75043, -0.65875],
            'C': [0.61013, -0.93043, -0.85983, -2.2704, 1.5877, -2.0066, -0.30314, 1.2544, -0.2832, -1.2844, -0.73449,
                  -0.11235, -0.41152, -0.0050164, 0.28307, 0.20522, -0.021084, -0.15627, -0.32689],
            'D': [-1.3215, 0.24063, -0.032754, -0.37863, 1.2051, 1.0001, 2.1827, 0.19212, -0.60529, 0.37639, -0.46451,
                  -0.46788, 1.4077, -2.1661, 0.72604, -0.12332, -0.8243, -0.082989, 0.053476],
            'E': [-0.87713, 1.4905, 1.0755, 0.35944, 1.567, 0.41365, 1.0944, 0.72634, -0.74957, 0.038939, 0.075057,
                  0.78637, -1.4543, 1.6667, -0.097439, -0.24293, 1.7687, 0.36174, -0.11585],
            'F': [1.3557, -0.10336, -0.4309, 0.41269, -0.083356, 0.83783, 0.095381, -0.65222, -0.3119, 0.43293, -1.0011,
                  -0.66855, -0.10242, 1.2066, 2.6234, 1.9981, -0.25016, 0.71979, 0.21569],
            'G': [-1.0818, -2.1561, 0.77082, -0.92747, -1.0748, 1.7997, -1.3708, 1.279, -1.2098, 0.46065, 0.43076,
                  0.20037, -0.2302, 0.2646, 0.57149, -0.68432, 0.19341, -0.061606, -0.08071],
            'H': [-0.050161, 0.69246, -0.88397, -0.64601, 0.24622, 0.10487, -1.1317, -2.3661, -0.89918, 0.46391,
                  -0.62359, 2.5478, -0.34737, -0.52062, 0.17522, -0.88648, -0.4755, 0.023187, -0.28261],
            'I': [1.4829, -0.46435, 0.50189, 0.55724, -0.51535, -0.29914, 0.97236, -0.15793, -0.98246, -0.54347,
                  0.97806, 0.37577, 1.618, 0.62323, -0.59359, -0.35483, -0.085017, 0.55825, -2.7542],
            'K': [-0.85344, 1.529, 0.27747, 0.32993, -1.1786, -0.16633, -1.0459, 0.44621, 0.41027, -2.5318, 0.91329,
                  0.53385, 0.61417, -1.111, 1.1323, 0.95105, 0.76769, -0.016115, 0.054995],
            'L': [1.2857, 0.039488, 1.5378, 0.87969, -0.21419, 0.40389, -0.20426, -0.14351, 0.61024, -1.1927, -2.2149,
                  -0.84248, -0.5061, -0.48548, 0.10791, -2.1503, -0.12006, -0.60222, 0.26546],
            'M': [1.137, 0.64388, 0.13724, -0.2988, 1.2288, 0.24981, -1.6427, -0.75868, -0.54902, 1.0571, 1.272,
                  -1.9104, 0.70919, -0.93575, -0.6314, -0.079654, 1.634, -0.0021923, 0.49825],
            'N': [-1.084, -0.176, -0.47062, -0.92245, -0.32953, 0.74278, 0.34551, -1.4605, 0.25219, -1.2107, -0.59978,
                  -0.79183, 1.3268, 1.9839, -1.6137, 0.5333, 0.033889, -1.0331, 0.83019],
            'P': [-1.1823, -1.6911, -1.1331, 3.073, 1.1942, -0.93426, -0.72985, -0.042441, -0.19264, -0.21603, -0.1239,
                  0.054016, 0.15241, -0.019691, -0.20543, 0.10206, 0.07671, -0.081968, 0.20348],
            'Q': [-0.57747, 0.97452, -0.077547, -0.0033488, 0.17184, -0.52537, -0.27362, -0.1366, 0.2057, -0.013066,
                  1.8834, -1.2736, -0.84991, 1.0445, 0.69027, -1.2866, -2.6776, 0.1683, 0.086105],
            'R': [-0.62245, 1.545, -0.61966, 0.19057, -1.7485, -1.3909, -0.47526, 1.3938, -0.84556, 1.7344, -1.6516,
                  -0.52678, 0.6791, 0.24374, -0.62551, -0.0028271, -0.053884, 0.14926, -0.17232],
            'S': [-0.86409, -0.77147, 0.38542, -0.59389, -0.53313, -0.47585, 0.31966, -0.89716, 1.8029, 0.26431,
                  -0.23173, -0.37626, -0.47349, -0.42878, -0.47297, -0.079826, 0.57043, 3.2057, -0.18413],
            'T': [-0.33027, -0.57447, 0.18653, -0.28941, -0.62681, -1.0737, 0.80363, -0.59525, 1.8786, 1.3971, 0.63929,
                  0.21281, -0.067048, 0.096271, 1.323, -0.36173, 1.2261, -2.2771, -0.65412],
            'V': [1.1675, -0.61554, 0.95405, 0.11662, -0.74473, -1.1482, 1.1309, 0.12079, -0.77171, 0.18597, 0.93442,
                  1.201, 0.3826, -0.091573, -0.31269, 0.074367, -0.22946, 0.24322, 2.9836],
            'W': [1.1881, 0.43789, -1.7915, 0.138, 0.43088, 1.6467, -0.11987, 1.7369, 2.0818, 0.33122, 0.31829, 1.1586,
                  0.67649, 0.30819, -0.55772, -0.54491, -0.17969, 0.24477, 0.38674],
            'Y': [0.54671, -0.1468, -1.5688, 0.19001, -1.2736, 0.66162, 1.1614, -0.18614, -0.70654, -0.43634, 0.44775,
                  -0.71366, -2.5907, -1.1649, -1.1576, 0.66572, 0.21019, -0.61016, -0.34844]},
        'refractivity': {'A': [0.102045615], 'C': [0.841053374], 'D': [0.282153774], 'E': [0.405831178],
                         'F': [0.691276746], 'G': [0], 'H': [0.512814484], 'I': [0.448154244], 'K': [0.50058782],
                         'L': [0.441570656], 'M': [0.508817305], 'N': [0.282153774], 'P': [0.256995062],
                         'Q': [0.405831178], 'R': [0.626851634], 'S': [0.149306372], 'T': [0.258876087],
                         'V': [0.327298378], 'W': [1], 'Y': [0.741359041]},
        't_scale': {'A': [-8.4, -8.01, -3.73, -3.65, -6.12, -1.59, 1.56],
                    'C': [-2.44, -1.96, 0.93, -2.35, 1.31, 2.29, -1.52],
                    'D': [-6.84, -0.94, 17.68, -0.03, 3.44, 9.07, 4.32],
                    'E': [-6.5, 16.2, 17.28, 3.11, -4.75, -2.54, 4.72],
                    'F': [21.59, -5.73, 1.03, -3.3, 2.64, -5.02, 1.7],
                    'G': [-8.48, -10.37, -5.14, -6.51, -11.84, -3.6, 2.01],
                    'H': [15.28, -3.67, 6.72, -6.38, 4.12, -1.55, -2.85],
                    'I': [-2.97, 4.64, -0.77, 11, 3.26, -4.36, -7.88],
                    'K': [2.7, 13.46, -14.03, -2.55, 2.77, 0.15, 3.19],
                    'L': [2.61, 5.96, 1.97, 2.59, -4.77, -4.84, -5.44],
                    'M': [3.38, 12.43, -4.77, 0.45, -1.55, -0.6, 3.26],
                    'N': [-3.11, -1.22, 6.26, -9.38, 9.94, 7.66, -4.81],
                    'P': [-5.35, -9.07, -1.52, -8.79, -8.73, 4.29, -9.91],
                    'Q': [-5.31, 15.64, 8.44, 1.03, -4.32, -4.4, -0.52],
                    'R': [-2.27, 18.9, -18.24, -3.47, 3.03, 6.64, 0.45],
                    'S': [-15.88, -11.21, -2.44, -3.61, 3.46, -0.37, 8.98],
                    'T': [-17.81, -13.64, -5.19, 10.57, 6.91, -4.43, 3.49],
                    'V': [-5.8, -6.15, -2.26, 9.87, 5.28, -1.49, -7.54],
                    'W': [21.68, -8.78, -2.53, 15.53, -8.15, 11.98, 3.23],
                    'Y': [23.9, -6.47, 0.31, -4.14, 4.08, -7.28, 3.59]},
        'tm_tend': {'A': [0.38], 'C': [-0.3], 'D': [-3.27], 'E': [-2.9], 'F': [1.98], 'G': [-0.19], 'H': [-1.44],
                    'I': [1.97], 'K': [-3.46], 'L': [1.82], 'M': [1.4], 'N': [-1.62], 'P': [-1.44], 'Q': [-1.84],
                    'R': [-2.57], 'S': [-0.53], 'T': [-0.32], 'V': [1.46], 'W': [1.53], 'Y': [0.49]},
        'z3': {'A': [0.07, -1.73, 0.09], 'C': [0.71, -0.97, 4.13], 'D': [3.64, 1.13, 2.36], 'E': [3.08, 0.39, -0.07],
               'F': [-4.92, 1.3, 0.45], 'G': [2.23, -5.36, 0.3], 'H': [2.41, 1.74, 1.11], 'I': [-4.44, -1.68, -1.03],
               'K': [2.84, 1.41, -3.14], 'L': [-4.19, -1.03, -0.98], 'M': [-2.49, -0.27, -0.41],
               'N': [3.22, 1.45, 0.84], 'P': [-1.22, 0.88, 2.23], 'Q': [2.18, 0.53, -1.14], 'R': [2.88, 2.52, -3.44],
               'S': [1.96, -1.63, 0.57], 'T': [0.92, -2.09, -1.4], 'V': [-2.69, -2.53, -1.29], 'W': [-4.75, 3.65, 0.85],
               'Y': [-1.39, 2.32, 0.01]},
        'z5': {'A': [0.24, -2.32, 0.6, -0.14, 1.3], 'C': [0.84, -1.67, 3.71, 0.18, -2.65],
               'D': [3.98, 0.93, 1.93, -2.46, 0.75], 'E': [3.11, 0.26, -0.11, -3.04, -0.25],
               'F': [-4.22, 1.94, 1.06, 0.54, -0.62], 'G': [2.05, -4.06, 0.36, -0.82, -0.38],
               'H': [2.47, 1.95, 0.26, 3.9, 0.09], 'I': [-3.89, -1.73, -1.71, -0.84, 0.26],
               'K': [2.29, 0.89, -2.49, 1.49, 0.31], 'L': [-4.28, -1.3, -1.49, -0.72, 0.84],
               'M': [-2.85, -0.22, 0.47, 1.94, -0.98], 'N': [3.05, 1.62, 1.04, -1.15, 1.61],
               'P': [-1.66, 0.27, 1.84, 0.7, 2], 'Q': [1.75, 0.5, -1.44, -1.34, 0.66],
               'R': [3.52, 2.5, -3.5, 1.99, -0.17], 'S': [2.39, -1.07, 1.15, -1.39, 0.67],
               'T': [0.75, -2.18, -1.12, -1.46, -0.4], 'V': [-2.59, -2.64, -1.54, -0.85, -0.02],
               'W': [-4.36, 3.94, 0.59, 3.44, -1.59], 'Y': [-2.54, 2.44, 0.43, 0.04, -1.47]},
        'seeva_v1': {'A': [-9.3791], 'C': [-5.1817], 'D': [-0.1271], 'E': [2.5735],
                      'F': [7.7645], 'G': [-12.1224], 'H': [5.7387], 'I': [-3.1927],
                      'K': [1.2657], 'L': [-3.7677], 'M': [0.214], 'N': [-0.9875],
                      'P': [-4.6786], 'Q': [2.608], 'R': [8.0397], 'S': [-7.4445],
                      'T': [-4.5941], 'V': [-4.5696], 'W': [17.9318], 'Y': [9.9091]},
        'seeva_v2': {'A': [-3.029], 'C': [-5.5323], 'D': [-6.1488], 'E': [-2.3191],
                      'F': [-1.549], 'G': [-2.5872], 'H': [-0.5249], 'I': [0.9509],
                      'K': [-0.2902], 'L': [-0.7821], 'M': [-1.3442], 'N': [2.8252],
                      'P': [3.958], 'Q': [8.5726], 'R': [13.97], 'S': [1.4173],
                      'T': [2.8154], 'V': [-1.104], 'W': [-8.7567], 'Y': [-0.5419]},
        'seeva_v3': {'A': [0.2782], 'C': [3.6333], 'D': [8.3529], 'E': [4.6611],
                      'F': [-3.8851], 'G': [0.6311], 'H': [8.23], 'I': [-6.2187],
                      'K': [-5.8614], 'L': [-7.2986], 'M': [-1.3392], 'N': [5.635],
                      'P': [-0.2926], 'Q': [3.9764], 'R': [0.1907], 'S': [0.4662],
                      'T': [-0.432], 'V': [-5.587], 'W': [-2.4151], 'Y': [-2.7252]},
        'seeva_v4': {'A': [-2.2037], 'C': [-3.832], 'D': [2.6402], 'E': [3.8592],
                      'F': [-7.9043], 'G': [-1.8514], 'H': [2.6188], 'I': [4.8593],
                      'K': [1.9111], 'L': [5.3532], 'M': [1.1488], 'N': [2.0161],
                      'P': [-4.6889], 'Q': [1.7393], 'R': [1.2128], 'S': [-2.8794],
                      'T': [-3.7572], 'V': [5.6152], 'W': [2.19], 'Y': [-8.0472]},
        'seeva_v5': {'A': [-2.8175], 'C': [0.7158], 'D': [4.7241], 'E': [1.1865],
                      'F': [4.8332], 'G': [0.0438], 'H': [0.627], 'I': [0.4378],
                      'K': [-0.8927], 'L': [3.6698], 'M': [-5.5161], 'N': [1.3561],
                      'P': [-5.3945], 'Q': [-2.9617], 'R': [1.7008], 'S': [-1.9138],
                      'T': [-0.5086], 'V': [3.2834], 'W': [-6.751], 'Y': [4.1777]},
        'seeva_v6': {'A': [1.2455], 'C': [-2.2536], 'D': [0.9121], 'E': [5.2224],
                      'F': [-2.7416], 'G': [0.4538], 'H': [2.9528], 'I': [-2.49],
                      'K': [1.5021], 'L': [0.9556], 'M': [3.452], 'N': [-9.3931],
                      'P': [-0.339], 'Q': [-2.6743], 'R': [2.91], 'S': [3.2798],
                      'T': [-1.5927], 'V': [-1.1698], 'W': [-2.9536], 'Y': [2.7216]},
        'seeva_v7': {'A': [-0.5021], 'C': [-0.5394], 'D': [6.1944], 'E': [-0.4608],
                      'F': [-0.7836], 'G': [-4.9694], 'H': [-1.7119], 'I': [4.8464],
                      'K': [4.1011], 'L': [-3.1443], 'M': [-0.088], 'N': [-0.2461],
                      'P': [4.7338], 'Q': [-4.3069], 'R': [0.7243], 'S': [-1.4355],
                      'T': [2.3161], 'V': [-2.3731], 'W': [-1.4521], 'Y': [-0.9027]},
        'seeva_v8': {'A': [0.2046], 'C': [-1.0613], 'D': [2.3408], 'E': [-1.6437],
                      'F': [-1.6514], 'G': [2.5515], 'H': [-2.0467], 'I': [-1.5906],
                      'K': [-2.543], 'L': [1.2008], 'M': [-4.7854], 'N': [-1.0184],
                      'P': [-1.3885], 'Q': [-3.7452], 'R': [5.0065], 'S': [3.8983],
                      'T': [4.182], 'V': [-0.5729], 'W': [5.1578], 'Y': [-2.4952]},
        'seeva_v9': {'A': [-3.4891], 'C': [-5.4109], 'D': [1.5623], 'E': [-1.3408],
                      'F': [-1.2269], 'G': [2.3068], 'H': [2.7224], 'I': [-0.5636],
                      'K': [-2.1851], 'L': [-1.7613], 'M': [-0.0516], 'N': [-1.7855],
                      'P': [2.954], 'Q': [1.1737], 'R': [-3.4959], 'S': [-1.6959],
                      'T': [4.8128], 'V': [4.8767], 'W': [0.2777], 'Y': [2.3201]},
        'seeva_v10': {'A': [0.7521], 'C': [2.9483], 'D': [2.2098], 'E': [-3.3456],
                      'F': [-0.032], 'G': [2.07], 'H': [-2.1317], 'I': [-1.2105],
                      'K': [-1.4722], 'L': [-2.9698], 'M': [5.4021], 'N': [-1.6069],
                      'P': [-1.7469], 'Q': [-0.5516], 'R': [4.2833], 'S': [-4.6864],
                      'T': [-0.3966], 'V': [3.3215], 'W': [-0.1929], 'Y': [-0.644]},
        'seeva_v11': {'A': [-0.8426], 'C': [-2.3553], 'D': [1.663], 'E': [1.8745],
                      'F': [1.5894], 'G': [0.1484], 'H': [-4.046], 'I': [-4.3605],
                      'K': [5.4347], 'L': [-1.6107], 'M': [1.5207], 'N': [1.1064],
                      'P': [-3.6024], 'Q': [2.3877], 'R': [-1.3541], 'S': [1.4693],
                      'T': [2.9086], 'V': [0.168], 'W': [-0.5572], 'Y': [-1.5419]},
        'seeva_v12': {'A': [2.8882], 'C': [-2.1416], 'D': [-1.0712], 'E': [-3.9647],
                      'F': [1.7892], 'G': [1.5706], 'H': [5.7973], 'I': [-1.1829],
                      'K': [4.3225], 'L': [-0.7877], 'M': [-0.9971], 'N': [0.6158],
                      'P': [-0.9059], 'Q': [-2.1856], 'R': [0.6505], 'S': [-1.1492],
                      'T': [-0.3461], 'V': [-0.3487], 'W': [-0.4563], 'Y': [-2.0972]},
        'vsw_v1': {'A': [11.6344], 'R': [11.87107], 'N': [5.34951], 'D': [4.02676],
                    'C': [5.65032], 'Q': [2.17649], 'E': [2.36689], 'G': [11.7823],
                    'H': [2.33867], 'I': [0.41167], 'L': [0.26884], 'K': [9.00644],
                    'M': [4.36305], 'F': [7.26373], 'P': [5.3069], 'S': [9.15545],
                    'T': [4.22009], 'W': [11.70248], 'Y': [8.54042], 'V': [3.18401]},
        'vsw_v2': {'A': [1.89677], 'R': [2.87032], 'N': [7.68317], 'D': [2.99296],
                    'C': [2.87927], 'Q': [2.40014], 'E': [0.15236], 'G': [13.6975],
                    'H': [0.36099], 'I': [6.40422], 'L': [8.11636], 'K': [2.09657],
                    'M': [1.66543], 'F': [4.36604], 'P': [3.18399], 'S': [2.32007],
                    'T': [0.27234], 'W': [0.16164], 'Y': [1.52583], 'V': [2.29446]},
        'vsw_v3': {'A': [1.97792], 'R': [2.74823], 'N': [4.11656], 'D': [3.35851],
                    'C': [2.99048], 'Q': [0.84542], 'E': [4.04847], 'G': [3.47023],
                    'H': [1.56522], 'I': [1.24429], 'L': [2.89694], 'K': [3.35485],
                    'M': [3.97675], 'F': [1.09058], 'P': [0.59462], 'S': [0.49944],
                    'T': [1.39095], 'W': [5.62047], 'Y': [1.74131], 'V': [0.49218]},
        'vsw_v4': {'A': [2.60584], 'R': [1.25733], 'N': [4.1737], 'D': [3.77013],
                    'C': [2.3435], 'Q': [3.57161], 'E': [0.80392], 'G': [0.20083],
                    'H': [1.07602], 'I': [1.62207], 'L': [0.98231], 'K': [2.39215],
                    'M': [1.02332], 'F': [1.62148], 'P': [4.27735], 'S': [2.26929],
                    'T': [2.53752], 'W': [4.91875], 'Y': [1.28473], 'V': [0.51649]},
        'vsw_v5': {'A': [1.71457], 'R': [1.14309], 'N': [4.24926], 'D': [1.92286],
                    'C': [0.87816], 'Q': [1.20118], 'E': [2.0369], 'G': [0.9653],
                    'H': [2.00162], 'I': [1.23439], 'L': [1.93394], 'K': [0.37787],
                    'M': [0.13041], 'F': [3.19585], 'P': [1.52466], 'S': [0.12948],
                    'T': [1.07003], 'W': [0.56439], 'Y': [0.10915], 'V': [4.51497]},
        'vsw_v6': {'A': [2.03136], 'R': [0.47736], 'N': [0.1885], 'D': [0.67174],
                    'C': [1.94477], 'Q': [1.09218], 'E': [0.99025], 'G': [3.07417],
                    'H': [1.04105], 'I': [1.42428], 'L': [3.1559], 'K': [1.32714],
                    'M': [0.81746], 'F': [0.09336], 'P': [1.51178], 'S': [0.372],
                    'T': [1.55739], 'W': [0.73184], 'Y': [0.95319], 'V': [0.21018]},
        'vsw_v7': {'A': [0.81805], 'R': [2.72172], 'N': [1.06525], 'D': [1.55729],
                    'C': [1.06906], 'Q': [0.11427], 'E': [1.08696], 'G': [0.43997],
                    'H': [1.30003], 'I': [0.04118], 'L': [0.05848], 'K': [0.4619],
                    'M': [0.53974], 'F': [0.40768], 'P': [3.06724], 'S': [0.85265],
                    'T': [1.33781], 'W': [2.21604], 'Y': [1.14174], 'V': [1.87421]},
        'vsw_v8': {'A': [0.64043], 'R': [1.76875], 'N': [0.12846], 'D': [1.21043],
                    'C': [1.5617], 'Q': [0.05163], 'E': [2.34517], 'G': [0.28232],
                    'H': [2.06728], 'I': [0.17942], 'L': [1.19562], 'K': [0.18649],
                    'M': [1.70327], 'F': [0.10993], 'P': [0.3204], 'S': [0.99715],
                    'T': [0.60839], 'W': [0.97258], 'Y': [1.10035], 'V': [0.56113]},
        'vsw_v9': {'A': [1.07973], 'R': [1.44027], 'N': [0.83938], 'D': [0.30098],
                    'C': [2.61873], 'Q': [0.88206], 'E': [0.16624], 'G': [0.70226],
                    'H': [1.57], 'I': [0.41872], 'L': [1.52982], 'K': [0.33156],
                    'M': [0.08417], 'F': [0.66747], 'P': [0.68273], 'S': [0.54608],
                    'T': [0.55772], 'W': [0.31995], 'Y': [0.57515], 'V': [0.92008]},
        'svger_v1': {'A': [-6.6644], 'C': [-4.2608], 'D': [-3.9242], 'E': [-0.5934], 
                      'F': [5.0975], 'G': [-7.5875], 'H': [1.6594], 'I': [-1.2956],
                      'K': [3.1284], 'L': [-1.3666], 'M': [0.5409], 'N': [-3.8935],
                      'P': [-3.9602], 'Q': [0.2527], 'R': [6.7376], 'S': [6.7376], 
                      'T': [-3.9294], 'V': [-2.7809], 'W': [9.8424], 'Y': [6.2601]},
         'svger_v2': {'A': [0.2868], 'C': [3.8876], 'D': [-1.0325], 'E': [-1.388],
                      'F': [4.1277], 'G': [1.5124], 'H': [2.5802], 'I': [-2.1638],
                      'K': [-1.401], 'L': [-2.4682], 'M': [2.0466], 'N': [-1.5514],
                      'P': [0.6597], 'Q': [-1.5231], 'R': [-3.8121], 'S': [-3.8121],
                      'T': [-0.1936], 'V': [-0.8673], 'W': [2.7928], 'Y': [2.3193]},
         'svger_v3': {'A': [0.3188], 'C': [-3.1306], 'D': [1.5044], 'E': [-1.2357],
                      'F': [-0.078], 'G': [-2.6784], 'H': [1.782], 'I': [2.9984],
                      'K': [-1.823], 'L': [2.9637], 'M': [-3.1255], 'N': [2.5195],
                      'P': [1.1578], 'Q': [-1.96], 'R': [-2.4341], 'S': [-2.4341], 
                      'T': [0.8037], 'V': [0.5357], 'W': [3.2729], 'Y': [1.0426]},
         'svger_v4': {'A': [0.6555], 'C': [-2.2351], 'D': [0.9481], 'E': [1.883], 
                      'F': [0.3976], 'G': [2.639], 'H': [1.7931], 'I': [-2.4099], 
                      'K': [-1.1742], 'L': [-1.741], 'M': [-3.8677], 'N': [0.1997],
                      'P': [-0.2066], 'Q': [1.9797], 'R': [0.2419], 'S': [0.2419],
                      'T': [-0.2626], 'V': [-0.5721], 'W': [-0.218], 'Y': [1.7078]},
         'svger_v5': {'A': [1.4012], 'C': [-1.4149], 'D': [-2.3026], 'E': [-2.4956],
                      'F': [1.836], 'G': [2.1595], 'H': [-0.3786], 'I': [0.8273], 
                      'K': [2.1995], 'L': [0.0155], 'M': [-1.7609], 'N': [-1.3031], 
                      'P': [0.9526], 'Q': [-1.4181], 'R': [0.5035], 'S': [0.5035],
                      'T': [0.9809], 'V': [0.6678], 'W': [-0.0868], 'Y': [-0.8866]},
         'svger_v6': {'A': [-0.3495], 'C': [-1.2429], 'D': [0.8944], 'E': [1.8804],
                      'F': [1.4935], 'G': [-0.185], 'H': [-2.1119], 'I': [0.788], 
                      'K': [1.2086], 'L': [0.1626], 'M': [0.3169], 'N': [-1.3293], 
                      'P': [-0.2108], 'Q': [0.1043], 'R': [-1.1635], 'S': [-1.1635],
                      'T': [0.3535], 'V': [0.0943], 'W': [-0.784], 'Y': [1.2441]}, 
         'svger_v7': {'A': [-9.379], 'C': [-6.5663], 'D': [-1.0962], 'E': [1.2998],
                      'F': [5.6974], 'G': [-12.5872], 'H': [4.2739], 'I': [-1.7578], 
                      'K': [1.0935], 'L': [-1.6986], 'M': [-1.6994], 'N': [-1.2873],
                      'P': [-2.0637], 'Q': [1.1021], 'R': [6.0612], 'S': [6.0612], 
                      'T': [-3.7454], 'V': [-4.1165], 'W': [12.339], 'Y': [8.0691]},
         'svger_v8': {'A': [1.05], 'C': [-0.5046], 'D': [-2.9195], 'E': [-3.1492],
                      'F': [3.1713], 'G': [0.3346], 'H': [1.0312], 'I': [1.612],
                      'K': [-0.1509], 'L': [1.4535], 'M': [-0.4697], 'N': [-1.8613],
                      'P': [3.9547], 'Q': [-2.0914], 'R': [-3.0783], 'S': [-3.0783],
                      'T': [-0.6444], 'V': [1.5961], 'W': [3.0831], 'Y': [0.6611]},
         'svger_v9': {'A': [-0.7519], 'C': [1.0932], 'D': [1.8876], 'E': [1.0722],
                      'F': [-0.5587], 'G': [-1.4802], 'H': [1.735], 'I': [-1.0091],
                      'K': [-1.629], 'L': [-1.2796], 'M': [0.3935], 'N': [1.2087], 
                      'P': [1.8923], 'Q': [0.3915], 'R': [-1.7358], 'S': [-1.7358],
                      'T': [0.8504], 'V': [-0.6374], 'W': [0.1657], 'Y': [0.1274]},
         'svger_v10': {'A': [-7.4283], 'C': [-4.5636], 'D': [-3.7729], 'E': [-0.147],
                       'F': [6.6287], 'G': [-8.031], 'H': [-0.1768], 'I': [-3.1848],
                       'K': [5.1296], 'L': [-3.6525], 'M': [0.239], 'N': [-4.2649],
                       'P': [-5.0009], 'Q': [0.792], 'R': [11.0625], 'S': [11.0625],
                       'T': [-4.8291], 'V': [-4.5968], 'W': [7.2301], 'Y': [7.5041]},
         'svger_v11': {'A': [2.6791], 'C': [0.4112], 'D': [-0.3285], 'E': [-2.2991], 
                       'F': [-1.1017], 'G': [3.2671], 'H': [-2.4065], 'I': [-0.7208],
                       'K': [-1.2543], 'L': [-0.4764], 'M': [-2.1326], 'N': [-0.0481],
                       'P': [0.5687], 'Q': [-2.2737], 'R': [3.4593], 'S': [3.4593], 
                       'T': [0.5004], 'V': [0.3457], 'W': [-1.8468], 'Y': [0.1979]},
         'svrg_v1': {'A': [-7.8944], 'R': [17.5769], 'N': [-2.826], 'D': [-4.5483], 
                      'C': [-6.7917], 'Q': [0.6728], 'E': [-1.8994], 'G': [-9.5317], 
                      'H': [-1.445], 'I': [-0.0686], 'L': [0.5394], 'K': [7.0222], 
                      'M': [-1.7032], 'F': [-5.3789], 'P': [5.9499], 'S': [-6.1313], 
                      'T': [-4.435], 'W': [14.4057], 'Y': [9.0291], 'V': [-2.5423]}, 
          'svrg_v2': {'A': [-0.8971], 'R': [-11.3412], 'N': [-2.4212], 'D': [-0.0121],
                      'C': [-1.2466], 'Q': [0.0303], 'E': [0.3917], 'G': [-1.5891], 
                      'H': [-0.035], 'I': [1.4303], 'L': [2.316], 'K': [-0.6611],
                      'M': [-0.4693], 'F': [0.5673], 'P': [4.5434], 'S': [-1.3685],
                      'T': [-0.8531], 'W': [6.1716], 'Y': [4.6574], 'V': [0.7864]},
          'svrg_v3': {'A': [3.5421], 'R': [1.1927], 'N': [-3.3507], 'D': [0.1875], 
                      'C': [1.9419], 'Q': [-2.2631], 'E': [-1.6013], 'G': [5.156],
                      'H': [-1.2536], 'I': [-4.3247], 'L': [-4.6425], 'K': [-1.1883],
                      'M': [-1.4478], 'F': [0.3261], 'P': [2.2261], 'S': [1.7606], 
                      'T': [0.0328], 'W': [1.103], 'Y': [4.8093], 'V': [-2.2061]},
          'svrg_v4': {'A': [0.242], 'R': [-1.8994], 'N': [-0.0584], 'D': [0.0109], 
                      'C': [0.7955], 'Q': [0.0619], 'E': [-0.1745], 'G': [0.6194],
                      'H': [-0.3629], 'I': [-1.6996], 'L': [-0.8513], 'K': [6.4838],
                      'M': [2.096], 'F': [-0.7033], 'P': [-2.5738], 'S': [-0.1093],
                      'T': [-0.5665], 'W': [0.9842], 'Y': [-0.6647], 'V': [-1.6301]},
          'svrg_v5': {'A': [-0.5321], 'R': [-0.755], 'N': [1.4997], 'D': [-0.9499], 
                      'C': [-1.0506], 'Q': [1.942], 'E': [0.2322], 'G': [-0.4203],
                      'H': [1.736], 'I': [0.764], 'L': [-1.7583], 'K': [0.581],
                      'M': [-0.1749], 'F': [-1.5043], 'P': [-0.7741], 'S': [-0.4152],
                      'T': [0.5941], 'W': [-3.2607], 'Y': [4.3628], 'V': [-0.1165]},
          'svrg_v6': {'A': [-0.2858], 'R': [-0.0905], 'N': [2.0161], 'D': [1.9142],
                      'C': [-0.7088], 'Q': [-1.1759], 'E': [-0.9452], 'G': [-0.6525],
                      'H': [-1.3811], 'I': [-0.3046], 'L': [0.5731], 'K': [-0.9637], 
                      'M': [-1.1594], 'F': [0.1277], 'P': [-3.7752], 'S': [1.8597], 
                      'T': [0.8999], 'W': [1.905], 'Y': [1.7911], 'V': [0.356]},
          'svrg_v7': {'A': [-0.6999], 'R': [-0.2088], 'N': [2.7727], 'D': [1.1343],
                      'C': [1.2641], 'Q': [0.7002], 'E': [0.8088], 'G': [-0.2503],
                      'H': [-0.5462], 'I': [-1.5165], 'L': [-0.7643], 'K': [-1.857], 
                      'M': [2.6295], 'F': [-1.3229], 'P': [0.8819], 'S': [-0.1317], 
                      'T': [-2.4932], 'W': [0.5675], 'Y': [0.0499], 'V': [-1.0181]},
          'svrg_v8': {'A': [0.4211], 'R': [0.5582], 'N': [-1.7785], 'D': [0.0063],
                      'C': [0.4602], 'Q': [-0.7694], 'E': [0.4639], 'G': [0.2995], 
                      'H': [-0.505], 'I': [1.1682], 'L': [0.8143], 'K': [-0.6708], 
                      'M': [2.0963], 'F': [1.831], 'P': [-2.0654], 'S': [-1.8226], 
                      'T': [-2.0566], 'W': [-0.6034], 'Y': [1.65], 'V': [0.5027]},
          'svrg_v9': {'A': [0.1111], 'R': [-0.0158], 'N': [0.855], 'D': [0.4384], 
                      'C': [-0.0698], 'Q': [-1.113], 'E': [0.0109], 'G': [-1.23],
                      'H': [-2.9701], 'I': [-0.6018], 'L': [0.1033], 'K': [1.1044],
                      'M': [0.3437], 'F': [0.9651], 'P': [1.6217], 'S': [-0.3163],
                      'T': [0.7035], 'W': [-1.6574], 'Y': [0.6518], 'V': [1.0652]},
          'svrg_v10': {'A': [-0.3389], 'R': [0.0701], 'N': [-1.034], 'D': [-0.2222], 
                       'C': [0.1137], 'Q': [-0.7182], 'E': [-1.2775], 'G': [-0.4339],
                       'H': [0.4717], 'I': [-0.6483], 'L': [1.1251], 'K': [-0.7456],
                       'M': [2.5924], 'F': [-1.4047], 'P': [0.04], 'S': [0.2079], 
                       'T': [2.3847], 'W': [-0.1957], 'Y': [0.2976], 'V': [-0.2843]},
          'svrg_v11': {'A': [-5.9082], 'R': [7.4386], 'N': [-3.2495], 'D': [-3.4151],
                       'C': [-3.4528], 'Q': [0.7412], 'E': [-0.6074], 'G': [-6.8679],
                       'H': [1.2963], 'I': [-0.6107], 'L': [-0.7406], 'K': [3.019], 
                       'M': [1.3496], 'F': [-3.2208], 'P': [6.3729], 'S': [-5.334],
                       'T': [-3.2382], 'W': [11.0178], 'Y': [7.4664], 'V': [-2.0567]},
          'svrg_v12': {'A': [0.7197], 'R': [-3.1167], 'N': [-2.6099], 'D': [-1.5233],
                       'C': [4.8887], 'Q': [-0.9023], 'E': [-0.935], 'G': [2.8835], 
                       'H': [-0.2267], 'I': [-3.1206], 'L': [-3.4737], 'K': [-0.349],
                       'M': [3.1447], 'F': [0.2478], 'P': [3.7134], 'S': [-0.3277], 
                       'T': [-0.3181], 'W': [0.4087], 'Y': [1.5601], 'V': [-0.6638]},
          'svrg_v13': {'A': [-0.4996], 'R': [4.6443], 'N': [-1.4085], 'D': [-0.4735],
                       'C': [0.7554], 'Q': [3.0675], 'E': [2.76], 'G': [1.4014], 
                       'H': [0.2235], 'I': [-2.2368], 'L': [-1.9935], 'K': [2.4237],
                       'M': [1.4995], 'F': [-1.5559], 'P': [-1.7188], 'S': [-1.1852],
                       'T': [-0.8146], 'W': [-3.3601], 'Y': [-1.4356], 'V': [-0.0932]},
          'svrg_v14': {'A': [1.1106], 'R': [-0.2724], 'N': [-0.0991], 'D': [0.8486], 
                       'C': [-2.7832], 'Q': [1.2869], 'E': [1.4287], 'G': [2.882],
                       'H': [0.0099], 'I': [-1.9152], 'L': [-1.5138], 'K': [-0.6669],
                       'M': [-4.1297], 'F': [-0.0085], 'P': [1.5139], 'S': [0.4623],
                       'T': [-0.0062], 'W': [0.4752], 'Y': [1.7758], 'V': [-0.399]},
          'svrg_v15': {'A': [1.2405], 'R': [0.8752], 'N': [-1.3005], 'D': [-2.1593],
                       'C': [-1.1356], 'Q': [-1.4086], 'E': [-2.2504], 'G': [1.5407],
                       'H': [-1.0526], 'I': [1.3986], 'L': [0.4502], 'K': [3.2261],
                       'M': [-0.9596], 'F': [0.8157], 'P': [1.1937], 'S': [-0.9085],
                       'T': [1.1008], 'W': [-0.9542], 'Y': [-0.5768], 'V': [0.8648]},
          'svrg_v16': {'A': [-0.2713], 'R': [-1.2578], 'N': [-1.1164], 'D': [0.6811],
                       'C': [-0.9146], 'Q': [0.0223], 'E': [2.5563], 'G': [-0.5376],
                       'H': [-3.0805], 'I': [1.0304], 'L': [0.8461], 'K': [0.2049], 
                       'M': [1.0745], 'F': [-0.5997], 'P': [1.2481], 'S': [-0.2604],
                       'T': [0.0714], 'W': [-1.0383], 'Y': [0.7847], 'V': [0.5566]},
          'hesh_v1': {'A': [0.59], 'R': [-0.64], 'N': [-0.76], 'D': [-0.88], 'C': [2.09],
                       'Q': [-1.15], 'E': [-0.73], 'G': [0.53], 'H': [-0.01], 'I': [1.16],
                       'L': [0.71], 'K': [-2.0], 'M': [0.93], 'F': [0.62], 'P': [-0.92], 
                       'S': [-0.2], 'T': [-0.1], 'W': [-0.33], 'Y': [-0.38], 'V': [1.47]},
           'hesh_v2': {'A': [-0.28], 'R': [-0.08], 'N': [-0.69], 'D': [-0.76], 'C': [-1.24],
                       'Q': [-0.42], 'E': [-0.79], 'G': [-0.99], 'H': [-0.46], 'I': [0.77],
                       'L': [1.04], 'K': [-0.09], 'M': [-0.16], 'F': [1.39], 'P': [0.44],
                       'S': [-0.95], 'T': [-0.66], 'W': [2.77], 'Y': [1.21], 'V': [-0.03]},
           'hesh_v3': {'A': [0.11], 'R': [-4.07], 'N': [0.35], 'D': [0.18], 'C': [-0.17],
                       'Q': [0.3], 'E': [-0.19], 'G': [-0.06], 'H': [0.55], 'I': [0.28], 
                       'L': [0.37], 'K': [0.94], 'M': [0.31], 'F': [-0.14], 'P': [0.55],
                       'S': [0.29], 'T': [0.17], 'W': [-0.19], 'Y': [0.36], 'V': [0.05]}, 
           'hesh_v4': {'A': [2.11], 'R': [0.07], 'N': [-0.43], 'D': [0.37], 'C': [-1.86],
                       'Q': [-0.71], 'E': [0.81], 'G': [0.69], 'H': [-1.08], 'I': [0.79], 
                       'L': [1.39], 'K': [0.89], 'M': [-0.93], 'F': [-0.13], 'P': [-1.04], 
                       'S': [-0.53], 'T': [-0.19], 'W': [-1.17], 'Y': [-0.02], 'V': [0.98]},
           'hesh_v5': {'A': [-1.54], 'R': [1.51], 'N': [-0.29], 'D': [-0.44], 'C': [-1.03],
                       'Q': [0.35], 'E': [0.18], 'G': [-1.67], 'H': [0.32], 'I': [-0.3], 
                       'L': [0.04], 'K': [0.98], 'M': [0.33], 'F': [0.86], 'P': [-0.22], 
                       'S': [-1.06], 'T': [-0.65], 'W': [1.85], 'Y': [1.55], 'V': [-0.78]},
           'hesh_v6': {'A': [-0.54], 'R': [1.04], 'N': [0.77], 'D': [0.95], 'C': [-1.09], 
                       'Q': [1.09], 'E': [1.11], 'G': [0.05], 'H': [-0.42], 'I': [-0.86], 
                       'L': [-0.74], 'K': [1.91], 'M': [-1.31], 'F': [-1.39], 'P': [0.52],
                       'S': [0.85], 'T': [0.57], 'W': [-1.25], 'Y': [-0.42], 'V': [-0.82]},
           'hesh_v7': {'A': [-0.45], 'R': [1.42], 'N': [0.5], 'D': [1.75], 'C': [-1.07],
                       'Q': [0.02], 'E': [1.86], 'G': [0.05], 'H': [1.15], 'I': [-0.9], 
                       'L': [-0.92], 'K': [1.48], 'M': [-0.92], 'F': [-0.98], 'P': [-0.02],
                       'S': [-0.36], 'T': [-0.45], 'W': [-0.75], 'Y': [-0.62], 'V': [-0.79]},
           'hesh_v8': {'A': [0.24], 'R': [2.05], 'N': [-0.08], 'D': [-2.31], 'C': [-0.25],
                       'Q': [0.2], 'E': [-2.04], 'G': [0.01], 'H': [0.81], 'I': [-0.1], 
                       'L': [-0.11], 'K': [2.06], 'M': [-0.1], 'F': [-0.3], 'P': [0.19], 
                       'S': [0.02], 'T': [0.13], 'W': [-0.27], 'Y': [-0.09], 'V': [-0.06]}, 
           'hesh_v9': {'A': [-0.82], 'R': [0.71], 'N': [0.46], 'D': [0.93], 'C': [2.27], 
                       'Q': [0.75], 'E': [-0.99], 'G': [-1.25], 'H': [0.4], 'I': [-1.26],
                       'L': [-1.2], 'K': [-0.48], 'M': [0.59], 'F': [0.68], 'P': [-1.31],
                       'S': [0.94], 'T': [0.69], 'W': [-0.48], 'Y': [0.3], 'V': [-0.94]},
           'hesh_v10': {'A': [-1.19], 'R': [0.66], 'N': [-0.75], 'D': [-0.09], 'C': [-0.2], 
                        'Q': [-0.54], 'E': [0.52], 'G': [-1.66], 'H': [0.85], 'I': [0.39],
                        'L': [0.42], 'K': [0.29], 'M': [0.45], 'F': [1.3], 'P': [-0.67], 
                        'S': [-1.65], 'T': [-1.19], 'W': [2.13], 'Y': [1.02], 'V': [-0.11]},
           'hesh_v11': {'A': [-0.95], 'R': [1.85], 'N': [1.59], 'D': [-0.17], 'C': [0.67], 
                        'Q': [-0.26], 'E': [1.62], 'G': [-0.87], 'H': [1.01], 'I': [-0.76], 
                        'L': [-0.86], 'K': [-1.67], 'M': [0.49], 'F': [-0.94], 'P': [-0.94],
                        'S': [0.33], 'T': [0.34], 'W': [0.39], 'Y': [-0.01], 'V': [-0.85]}, 
           'hesh_v12': {'A': [-0.58], 'R': [0.18], 'N': [0.61], 'D': [-0.14], 'C': [-0.08],
                        'Q': [-0.25], 'E': [0.53], 'G': [-0.48], 'H': [0.41], 'I': [-0.72],
                        'L': [-0.75], 'K': [3.86], 'M': [-0.26], 'F': [-0.66], 'P': [-0.69], 
                        'S': [0.01], 'T': [-0.06], 'W': [-0.07], 'Y': [-0.19], 'V': [-0.68]},
           'andn920101' :  {'A': [4.35], 'R': [4.38], 'N': [4.75], 'D': [4.76], 'C': [4.65], 'Q': [4.37], 'E': [4.29], 'G': [3.97], 'H': [4.63], 'I': [3.95], 'L': [4.17], 'K': [4.36], 'M': [4.52], 'F': [4.66], 'P': [4.44], 'S': [4.5], 'T': [4.35], 'W': [4.7], 'Y': [4.6], 'V': [3.95]} ,
           'argp820101' :  {'A': [0.61], 'R': [0.6], 'N': [0.06], 'D': [0.46], 'C': [1.07], 'Q': [0.0], 'E': [0.47], 'G': [0.07], 'H': [0.61], 'I': [2.22], 'L': [1.53], 'K': [1.15], 'M': [1.18], 'F': [2.02], 'P': [1.95], 'S': [0.05], 'T': [0.05], 'W': [2.65], 'Y': [1.88], 'V': [1.32]} ,
           'argp820102' :  {'A': [1.18], 'R': [0.2], 'N': [0.23], 'D': [0.05], 'C': [1.89], 'Q': [0.72], 'E': [0.11], 'G': [0.49], 'H': [0.31], 'I': [1.45], 'L': [3.23], 'K': [0.06], 'M': [2.67], 'F': [1.96], 'P': [0.76], 'S': [0.97], 'T': [0.84], 'W': [0.77], 'Y': [0.39], 'V': [1.08]} ,
           'argp820103' :  {'A': [1.56], 'R': [0.45], 'N': [0.27], 'D': [0.14], 'C': [1.23], 'Q': [0.51], 'E': [0.23], 'G': [0.62], 'H': [0.29], 'I': [1.67], 'L': [2.93], 'K': [0.15], 'M': [2.96], 'F': [2.03], 'P': [0.76], 'S': [0.81], 'T': [0.91], 'W': [1.08], 'Y': [0.68], 'V': [1.14]} ,
           'begf750101' :  {'A': [1.0], 'R': [0.52], 'N': [0.35], 'D': [0.44], 'C': [0.06], 'Q': [0.44], 'E': [0.73], 'G': [0.35], 'H': [0.6], 'I': [0.73], 'L': [1.0], 'K': [0.6], 'M': [1.0], 'F': [0.6], 'P': [0.06], 'S': [0.35], 'T': [0.44], 'W': [0.73], 'Y': [0.44], 'V': [0.82]} ,
           'begf750102' :  {'A': [0.77], 'R': [0.72], 'N': [0.55], 'D': [0.65], 'C': [0.65], 'Q': [0.72], 'E': [0.55], 'G': [0.65], 'H': [0.83], 'I': [0.98], 'L': [0.83], 'K': [0.55], 'M': [0.98], 'F': [0.98], 'P': [0.55], 'S': [0.55], 'T': [0.83], 'W': [0.77], 'Y': [0.83], 'V': [0.98]} ,
           'begf750103' :  {'A': [0.37], 'R': [0.84], 'N': [0.97], 'D': [0.97], 'C': [0.84], 'Q': [0.64], 'E': [0.53], 'G': [0.97], 'H': [0.75], 'I': [0.37], 'L': [0.53], 'K': [0.75], 'M': [0.64], 'F': [0.53], 'P': [0.97], 'S': [0.84], 'T': [0.75], 'W': [0.97], 'Y': [0.84], 'V': [0.37]} ,
           'bhar880101' :  {'A': [0.357], 'R': [0.529], 'N': [0.463], 'D': [0.511], 'C': [0.346], 'Q': [0.493], 'E': [0.497], 'G': [0.544], 'H': [0.323], 'I': [0.462], 'L': [0.365], 'K': [0.466], 'M': [0.295], 'F': [0.314], 'P': [0.509], 'S': [0.507], 'T': [0.444], 'W': [0.305], 'Y': [0.42], 'V': [0.386]} ,
           'bigc670101' :  {'A': [52.6], 'R': [109.1], 'N': [75.7], 'D': [68.4], 'C': [68.3], 'Q': [89.7], 'E': [84.7], 'G': [36.3], 'H': [91.9], 'I': [102.0], 'L': [102.0], 'K': [105.1], 'M': [97.7], 'F': [113.9], 'P': [73.6], 'S': [54.9], 'T': [71.2], 'W': [135.4], 'Y': [116.2], 'V': [85.1]} ,
           'biov880101' :  {'A': [16.0], 'R': [-70.0], 'N': [-74.0], 'D': [-78.0], 'C': [168.0], 'Q': [-73.0], 'E': [-106.0], 'G': [-13.0], 'H': [50.0], 'I': [151.0], 'L': [145.0], 'K': [-141.0], 'M': [124.0], 'F': [189.0], 'P': [-20.0], 'S': [-70.0], 'T': [-38.0], 'W': [145.0], 'Y': [53.0], 'V': [123.0]} ,
           'biov880102' :  {'A': [44.0], 'R': [-68.0], 'N': [-72.0], 'D': [-91.0], 'C': [90.0], 'Q': [-117.0], 'E': [-139.0], 'G': [-8.0], 'H': [47.0], 'I': [100.0], 'L': [108.0], 'K': [-188.0], 'M': [121.0], 'F': [148.0], 'P': [-36.0], 'S': [-60.0], 'T': [-54.0], 'W': [163.0], 'Y': [22.0], 'V': [117.0]} ,
           'broc820101' :  {'A': [7.3], 'R': [-3.6], 'N': [-5.7], 'D': [-2.9], 'C': [-9.2], 'Q': [-0.3], 'E': [-7.1], 'G': [-1.2], 'H': [-2.1], 'I': [6.6], 'L': [20.0], 'K': [-3.7], 'M': [5.6], 'F': [19.2], 'P': [5.1], 'S': [-4.1], 'T': [0.8], 'W': [16.3], 'Y': [5.9], 'V': [3.5]} ,
           'broc820102' :  {'A': [3.9], 'R': [3.2], 'N': [-2.8], 'D': [-2.8], 'C': [-14.3], 'Q': [1.8], 'E': [-7.5], 'G': [-2.3], 'H': [2.0], 'I': [11.0], 'L': [15.0], 'K': [-2.5], 'M': [4.1], 'F': [14.7], 'P': [5.6], 'S': [-3.5], 'T': [1.1], 'W': [17.8], 'Y': [3.8], 'V': [2.1]} ,
           'bulh740101' :  {'A': [-0.2], 'R': [-0.12], 'N': [0.08], 'D': [-0.2], 'C': [-0.45], 'Q': [0.16], 'E': [-0.3], 'G': [0.0], 'H': [-0.12], 'I': [-2.26], 'L': [-2.46], 'K': [-0.35], 'M': [-1.47], 'F': [-2.33], 'P': [-0.98], 'S': [-0.39], 'T': [-0.52], 'W': [-2.01], 'Y': [-2.24], 'V': [-1.56]} ,
           'bulh740102' :  {'A': [0.691], 'R': [0.728], 'N': [0.596], 'D': [0.558], 'C': [0.624], 'Q': [0.649], 'E': [0.632], 'G': [0.592], 'H': [0.646], 'I': [0.809], 'L': [0.842], 'K': [0.767], 'M': [0.709], 'F': [0.756], 'P': [0.73], 'S': [0.594], 'T': [0.655], 'W': [0.743], 'Y': [0.743], 'V': [0.777]} ,
           'buna790101' :  {'A': [8.249], 'R': [8.274], 'N': [8.747], 'D': [8.41], 'C': [8.312], 'Q': [8.411], 'E': [8.368], 'G': [8.391], 'H': [8.415], 'I': [8.195], 'L': [8.423], 'K': [8.408], 'M': [8.418], 'F': [8.228], 'P': [0.0], 'S': [8.38], 'T': [8.236], 'W': [8.094], 'Y': [8.183], 'V': [8.436]} ,
           'buna790102' :  {'A': [4.349], 'R': [4.396], 'N': [4.755], 'D': [4.765], 'C': [4.686], 'Q': [4.373], 'E': [4.295], 'G': [3.972], 'H': [4.63], 'I': [4.224], 'L': [4.385], 'K': [4.358], 'M': [4.513], 'F': [4.663], 'P': [4.471], 'S': [4.498], 'T': [4.346], 'W': [4.702], 'Y': [4.604], 'V': [4.184]} ,
           'buna790103' :  {'A': [6.5], 'R': [6.9], 'N': [7.5], 'D': [7.0], 'C': [7.7], 'Q': [6.0], 'E': [7.0], 'G': [5.6], 'H': [8.0], 'I': [7.0], 'L': [6.5], 'K': [6.5], 'M': [0.0], 'F': [9.4], 'P': [0.0], 'S': [6.5], 'T': [6.9], 'W': [0.0], 'Y': [6.8], 'V': [7.0]} ,
           'bura740101' :  {'A': [0.486], 'R': [0.262], 'N': [0.193], 'D': [0.288], 'C': [0.2], 'Q': [0.418], 'E': [0.538], 'G': [0.12], 'H': [0.4], 'I': [0.37], 'L': [0.42], 'K': [0.402], 'M': [0.417], 'F': [0.318], 'P': [0.208], 'S': [0.2], 'T': [0.272], 'W': [0.462], 'Y': [0.161], 'V': [0.379]} ,
           'bura740102' :  {'A': [0.288], 'R': [0.362], 'N': [0.229], 'D': [0.271], 'C': [0.533], 'Q': [0.327], 'E': [0.262], 'G': [0.312], 'H': [0.2], 'I': [0.411], 'L': [0.4], 'K': [0.265], 'M': [0.375], 'F': [0.318], 'P': [0.34], 'S': [0.354], 'T': [0.388], 'W': [0.231], 'Y': [0.429], 'V': [0.495]} ,
           'cham810101' :  {'A': [0.52], 'R': [0.68], 'N': [0.76], 'D': [0.76], 'C': [0.62], 'Q': [0.68], 'E': [0.68], 'G': [0.0], 'H': [0.7], 'I': [1.02], 'L': [0.98], 'K': [0.68], 'M': [0.78], 'F': [0.7], 'P': [0.36], 'S': [0.53], 'T': [0.5], 'W': [0.7], 'Y': [0.7], 'V': [0.76]} ,
           'cham820101' :  {'A': [0.046], 'R': [0.291], 'N': [0.134], 'D': [0.105], 'C': [0.128], 'Q': [0.18], 'E': [0.151], 'G': [0.0], 'H': [0.23], 'I': [0.186], 'L': [0.186], 'K': [0.219], 'M': [0.221], 'F': [0.29], 'P': [0.131], 'S': [0.062], 'T': [0.108], 'W': [0.409], 'Y': [0.298], 'V': [0.14]} ,
           'cham820102' :  {'A': [-0.368], 'R': [-1.03], 'N': [0.0], 'D': [2.06], 'C': [4.53], 'Q': [0.731], 'E': [1.77], 'G': [-0.525], 'H': [0.0], 'I': [0.791], 'L': [1.07], 'K': [0.0], 'M': [0.656], 'F': [1.06], 'P': [-2.24], 'S': [-0.524], 'T': [0.0], 'W': [1.6], 'Y': [4.91], 'V': [0.401]} ,
           'cham830101' :  {'A': [0.71], 'R': [1.06], 'N': [1.37], 'D': [1.21], 'C': [1.19], 'Q': [0.87], 'E': [0.84], 'G': [1.52], 'H': [1.07], 'I': [0.66], 'L': [0.69], 'K': [0.99], 'M': [0.59], 'F': [0.71], 'P': [1.61], 'S': [1.34], 'T': [1.08], 'W': [0.76], 'Y': [1.07], 'V': [0.63]} ,
           'cham830102' :  {'A': [-0.118], 'R': [0.124], 'N': [0.289], 'D': [0.048], 'C': [0.083], 'Q': [-0.105], 'E': [-0.245], 'G': [0.104], 'H': [0.138], 'I': [0.23], 'L': [-0.052], 'K': [0.032], 'M': [-0.258], 'F': [0.015], 'P': [0.0], 'S': [0.225], 'T': [0.166], 'W': [0.158], 'Y': [0.094], 'V': [0.513]} ,
           'cham830103' :  {'A': [0.0], 'R': [1.0], 'N': [1.0], 'D': [1.0], 'C': [1.0], 'Q': [1.0], 'E': [1.0], 'G': [0.0], 'H': [1.0], 'I': [2.0], 'L': [1.0], 'K': [1.0], 'M': [1.0], 'F': [1.0], 'P': [0.0], 'S': [1.0], 'T': [2.0], 'W': [1.0], 'Y': [1.0], 'V': [2.0]} ,
           'cham830104' :  {'A': [0.0], 'R': [1.0], 'N': [1.0], 'D': [1.0], 'C': [0.0], 'Q': [1.0], 'E': [1.0], 'G': [0.0], 'H': [1.0], 'I': [1.0], 'L': [2.0], 'K': [1.0], 'M': [1.0], 'F': [1.0], 'P': [0.0], 'S': [0.0], 'T': [0.0], 'W': [1.0], 'Y': [1.0], 'V': [0.0]} ,
           'cham830105' :  {'A': [0.0], 'R': [1.0], 'N': [0.0], 'D': [0.0], 'C': [0.0], 'Q': [1.0], 'E': [1.0], 'G': [0.0], 'H': [1.0], 'I': [0.0], 'L': [0.0], 'K': [1.0], 'M': [1.0], 'F': [1.0], 'P': [0.0], 'S': [0.0], 'T': [0.0], 'W': [1.5], 'Y': [1.0], 'V': [0.0]} ,
           'cham830106' :  {'A': [0.0], 'R': [5.0], 'N': [2.0], 'D': [2.0], 'C': [1.0], 'Q': [3.0], 'E': [3.0], 'G': [0.0], 'H': [3.0], 'I': [2.0], 'L': [2.0], 'K': [4.0], 'M': [3.0], 'F': [4.0], 'P': [0.0], 'S': [1.0], 'T': [1.0], 'W': [5.0], 'Y': [5.0], 'V': [1.0]} ,
           'cham830107' :  {'A': [0.0], 'R': [0.0], 'N': [1.0], 'D': [1.0], 'C': [0.0], 'Q': [0.0], 'E': [1.0], 'G': [1.0], 'H': [0.0], 'I': [0.0], 'L': [0.0], 'K': [0.0], 'M': [0.0], 'F': [0.0], 'P': [0.0], 'S': [0.0], 'T': [0.0], 'W': [0.0], 'Y': [0.0], 'V': [0.0]} ,
           'cham830108' :  {'A': [0.0], 'R': [1.0], 'N': [1.0], 'D': [0.0], 'C': [1.0], 'Q': [1.0], 'E': [0.0], 'G': [0.0], 'H': [1.0], 'I': [0.0], 'L': [0.0], 'K': [1.0], 'M': [1.0], 'F': [1.0], 'P': [0.0], 'S': [0.0], 'T': [0.0], 'W': [1.0], 'Y': [1.0], 'V': [0.0]} ,
           'choc750101' :  {'A': [91.5], 'R': [202.0], 'N': [135.2], 'D': [124.5], 'C': [117.7], 'Q': [161.1], 'E': [155.1], 'G': [66.4], 'H': [167.3], 'I': [168.8], 'L': [167.9], 'K': [171.3], 'M': [170.8], 'F': [203.4], 'P': [129.3], 'S': [99.1], 'T': [122.1], 'W': [237.6], 'Y': [203.6], 'V': [141.7]} ,
           'choc760101' :  {'A': [115.0], 'R': [225.0], 'N': [160.0], 'D': [150.0], 'C': [135.0], 'Q': [180.0], 'E': [190.0], 'G': [75.0], 'H': [195.0], 'I': [175.0], 'L': [170.0], 'K': [200.0], 'M': [185.0], 'F': [210.0], 'P': [145.0], 'S': [115.0], 'T': [140.0], 'W': [255.0], 'Y': [230.0], 'V': [155.0]} ,
           'choc760102' :  {'A': [25.0], 'R': [90.0], 'N': [63.0], 'D': [50.0], 'C': [19.0], 'Q': [71.0], 'E': [49.0], 'G': [23.0], 'H': [43.0], 'I': [18.0], 'L': [23.0], 'K': [97.0], 'M': [31.0], 'F': [24.0], 'P': [50.0], 'S': [44.0], 'T': [47.0], 'W': [32.0], 'Y': [60.0], 'V': [18.0]} ,
           'choc760103' :  {'A': [0.38], 'R': [0.01], 'N': [0.12], 'D': [0.15], 'C': [0.45], 'Q': [0.07], 'E': [0.18], 'G': [0.36], 'H': [0.17], 'I': [0.6], 'L': [0.45], 'K': [0.03], 'M': [0.4], 'F': [0.5], 'P': [0.18], 'S': [0.22], 'T': [0.23], 'W': [0.27], 'Y': [0.15], 'V': [0.54]} ,
           'choc760104' :  {'A': [0.2], 'R': [0.0], 'N': [0.03], 'D': [0.04], 'C': [0.22], 'Q': [0.01], 'E': [0.03], 'G': [0.18], 'H': [0.02], 'I': [0.19], 'L': [0.16], 'K': [0.0], 'M': [0.11], 'F': [0.14], 'P': [0.04], 'S': [0.08], 'T': [0.08], 'W': [0.04], 'Y': [0.03], 'V': [0.18]} ,
           'chop780101' :  {'A': [0.66], 'R': [0.95], 'N': [1.56], 'D': [1.46], 'C': [1.19], 'Q': [0.98], 'E': [0.74], 'G': [1.56], 'H': [0.95], 'I': [0.47], 'L': [0.59], 'K': [1.01], 'M': [0.6], 'F': [0.6], 'P': [1.52], 'S': [1.43], 'T': [0.96], 'W': [0.96], 'Y': [1.14], 'V': [0.5]} ,
           'chop780201' :  {'A': [1.42], 'R': [0.98], 'N': [0.67], 'D': [1.01], 'C': [0.7], 'Q': [1.11], 'E': [1.51], 'G': [0.57], 'H': [1.0], 'I': [1.08], 'L': [1.21], 'K': [1.16], 'M': [1.45], 'F': [1.13], 'P': [0.57], 'S': [0.77], 'T': [0.83], 'W': [1.08], 'Y': [0.69], 'V': [1.06]} ,
           'chop780202' :  {'A': [0.83], 'R': [0.93], 'N': [0.89], 'D': [0.54], 'C': [1.19], 'Q': [1.1], 'E': [0.37], 'G': [0.75], 'H': [0.87], 'I': [1.6], 'L': [1.3], 'K': [0.74], 'M': [1.05], 'F': [1.38], 'P': [0.55], 'S': [0.75], 'T': [1.19], 'W': [1.37], 'Y': [1.47], 'V': [1.7]} ,
           'chop780203' :  {'A': [0.74], 'R': [1.01], 'N': [1.46], 'D': [1.52], 'C': [0.96], 'Q': [0.96], 'E': [0.95], 'G': [1.56], 'H': [0.95], 'I': [0.47], 'L': [0.5], 'K': [1.19], 'M': [0.6], 'F': [0.66], 'P': [1.56], 'S': [1.43], 'T': [0.98], 'W': [0.6], 'Y': [1.14], 'V': [0.59]} ,
           'chop780204' :  {'A': [1.29], 'R': [0.44], 'N': [0.81], 'D': [2.02], 'C': [0.66], 'Q': [1.22], 'E': [2.44], 'G': [0.76], 'H': [0.73], 'I': [0.67], 'L': [0.58], 'K': [0.66], 'M': [0.71], 'F': [0.61], 'P': [2.01], 'S': [0.74], 'T': [1.08], 'W': [1.47], 'Y': [0.68], 'V': [0.61]} ,
           'chop780205' :  {'A': [1.2], 'R': [1.25], 'N': [0.59], 'D': [0.61], 'C': [1.11], 'Q': [1.22], 'E': [1.24], 'G': [0.42], 'H': [1.77], 'I': [0.98], 'L': [1.13], 'K': [1.83], 'M': [1.57], 'F': [1.1], 'P': [0.0], 'S': [0.96], 'T': [0.75], 'W': [0.4], 'Y': [0.73], 'V': [1.25]} ,
           'chop780206' :  {'A': [0.7], 'R': [0.34], 'N': [1.42], 'D': [0.98], 'C': [0.65], 'Q': [0.75], 'E': [1.04], 'G': [1.41], 'H': [1.22], 'I': [0.78], 'L': [0.85], 'K': [1.01], 'M': [0.83], 'F': [0.93], 'P': [1.1], 'S': [1.55], 'T': [1.09], 'W': [0.62], 'Y': [0.99], 'V': [0.75]} ,
           'chop780207' :  {'A': [0.52], 'R': [1.24], 'N': [1.64], 'D': [1.06], 'C': [0.94], 'Q': [0.7], 'E': [0.59], 'G': [1.64], 'H': [1.86], 'I': [0.87], 'L': [0.84], 'K': [1.49], 'M': [0.52], 'F': [1.04], 'P': [1.58], 'S': [0.93], 'T': [0.86], 'W': [0.16], 'Y': [0.96], 'V': [0.32]} ,
           'chop780208' :  {'A': [0.86], 'R': [0.9], 'N': [0.66], 'D': [0.38], 'C': [0.87], 'Q': [1.65], 'E': [0.35], 'G': [0.63], 'H': [0.54], 'I': [1.94], 'L': [1.3], 'K': [1.0], 'M': [1.43], 'F': [1.5], 'P': [0.66], 'S': [0.63], 'T': [1.17], 'W': [1.49], 'Y': [1.07], 'V': [1.69]} ,
           'chop780209' :  {'A': [0.75], 'R': [0.9], 'N': [1.21], 'D': [0.85], 'C': [1.11], 'Q': [0.65], 'E': [0.55], 'G': [0.74], 'H': [0.9], 'I': [1.35], 'L': [1.27], 'K': [0.74], 'M': [0.95], 'F': [1.5], 'P': [0.4], 'S': [0.79], 'T': [0.75], 'W': [1.19], 'Y': [1.96], 'V': [1.79]} ,
           'chop780210' :  {'A': [0.67], 'R': [0.89], 'N': [1.86], 'D': [1.39], 'C': [1.34], 'Q': [1.09], 'E': [0.92], 'G': [1.46], 'H': [0.78], 'I': [0.59], 'L': [0.46], 'K': [1.09], 'M': [0.52], 'F': [0.3], 'P': [1.58], 'S': [1.41], 'T': [1.09], 'W': [0.48], 'Y': [1.23], 'V': [0.42]} ,
           'chop780211' :  {'A': [0.74], 'R': [1.05], 'N': [1.13], 'D': [1.32], 'C': [0.53], 'Q': [0.77], 'E': [0.85], 'G': [1.68], 'H': [0.96], 'I': [0.53], 'L': [0.59], 'K': [0.82], 'M': [0.85], 'F': [0.44], 'P': [1.69], 'S': [1.49], 'T': [1.16], 'W': [1.59], 'Y': [1.01], 'V': [0.59]} ,
           'chop780212' :  {'A': [0.06], 'R': [0.07], 'N': [0.161], 'D': [0.147], 'C': [0.149], 'Q': [0.074], 'E': [0.056], 'G': [0.102], 'H': [0.14], 'I': [0.043], 'L': [0.061], 'K': [0.055], 'M': [0.068], 'F': [0.059], 'P': [0.102], 'S': [0.12], 'T': [0.086], 'W': [0.077], 'Y': [0.082], 'V': [0.062]} ,
           'chop780213' :  {'A': [0.076], 'R': [0.106], 'N': [0.083], 'D': [0.11], 'C': [0.053], 'Q': [0.098], 'E': [0.06], 'G': [0.085], 'H': [0.047], 'I': [0.034], 'L': [0.025], 'K': [0.115], 'M': [0.082], 'F': [0.041], 'P': [0.301], 'S': [0.139], 'T': [0.108], 'W': [0.013], 'Y': [0.065], 'V': [0.048]} ,
           'chop780214' :  {'A': [0.035], 'R': [0.099], 'N': [0.191], 'D': [0.179], 'C': [0.117], 'Q': [0.037], 'E': [0.077], 'G': [0.19], 'H': [0.093], 'I': [0.013], 'L': [0.036], 'K': [0.072], 'M': [0.014], 'F': [0.065], 'P': [0.034], 'S': [0.125], 'T': [0.065], 'W': [0.064], 'Y': [0.114], 'V': [0.028]} ,
           'chop780215' :  {'A': [0.058], 'R': [0.085], 'N': [0.091], 'D': [0.081], 'C': [0.128], 'Q': [0.098], 'E': [0.064], 'G': [0.152], 'H': [0.054], 'I': [0.056], 'L': [0.07], 'K': [0.095], 'M': [0.055], 'F': [0.065], 'P': [0.068], 'S': [0.106], 'T': [0.079], 'W': [0.167], 'Y': [0.125], 'V': [0.053]} ,
           'chop780216' :  {'A': [0.64], 'R': [1.05], 'N': [1.56], 'D': [1.61], 'C': [0.92], 'Q': [0.84], 'E': [0.8], 'G': [1.63], 'H': [0.77], 'I': [0.29], 'L': [0.36], 'K': [1.13], 'M': [0.51], 'F': [0.62], 'P': [2.04], 'S': [1.52], 'T': [0.98], 'W': [0.48], 'Y': [1.08], 'V': [0.43]} ,
           'cidh920101' :  {'A': [-0.45], 'R': [-0.24], 'N': [-0.2], 'D': [-1.52], 'C': [0.79], 'Q': [-0.99], 'E': [-0.8], 'G': [-1.0], 'H': [1.07], 'I': [0.76], 'L': [1.29], 'K': [-0.36], 'M': [1.37], 'F': [1.48], 'P': [-0.12], 'S': [-0.98], 'T': [-0.7], 'W': [1.38], 'Y': [1.49], 'V': [1.26]} ,
           'cidh920102' :  {'A': [-0.08], 'R': [-0.09], 'N': [-0.7], 'D': [-0.71], 'C': [0.76], 'Q': [-0.4], 'E': [-1.31], 'G': [-0.84], 'H': [0.43], 'I': [1.39], 'L': [1.24], 'K': [-0.09], 'M': [1.27], 'F': [1.53], 'P': [-0.01], 'S': [-0.93], 'T': [-0.59], 'W': [2.25], 'Y': [1.53], 'V': [1.09]} ,
           'cidh920103' :  {'A': [0.36], 'R': [-0.52], 'N': [-0.9], 'D': [-1.09], 'C': [0.7], 'Q': [-1.05], 'E': [-0.83], 'G': [-0.82], 'H': [0.16], 'I': [2.17], 'L': [1.18], 'K': [-0.56], 'M': [1.21], 'F': [1.01], 'P': [-0.06], 'S': [-0.6], 'T': [-1.2], 'W': [1.31], 'Y': [1.05], 'V': [1.21]} ,
           'cidh920104' :  {'A': [0.17], 'R': [-0.7], 'N': [-0.9], 'D': [-1.05], 'C': [1.24], 'Q': [-1.2], 'E': [-1.19], 'G': [-0.57], 'H': [-0.25], 'I': [2.06], 'L': [0.96], 'K': [-0.62], 'M': [0.6], 'F': [1.29], 'P': [-0.21], 'S': [-0.83], 'T': [-0.62], 'W': [1.51], 'Y': [0.66], 'V': [1.21]} ,
           'cidh920105' :  {'A': [0.02], 'R': [-0.42], 'N': [-0.77], 'D': [-1.04], 'C': [0.77], 'Q': [-1.1], 'E': [-1.14], 'G': [-0.8], 'H': [0.26], 'I': [1.81], 'L': [1.14], 'K': [-0.41], 'M': [1.0], 'F': [1.35], 'P': [-0.09], 'S': [-0.97], 'T': [-0.77], 'W': [1.71], 'Y': [1.11], 'V': [1.13]} ,
           'cohe430101' :  {'A': [0.75], 'R': [0.7], 'N': [0.61], 'D': [0.6], 'C': [0.61], 'Q': [0.67], 'E': [0.66], 'G': [0.64], 'H': [0.67], 'I': [0.9], 'L': [0.9], 'K': [0.82], 'M': [0.75], 'F': [0.77], 'P': [0.76], 'S': [0.68], 'T': [0.7], 'W': [0.74], 'Y': [0.71], 'V': [0.86]} ,
           'craj730101' :  {'A': [1.33], 'R': [0.79], 'N': [0.72], 'D': [0.97], 'C': [0.93], 'Q': [1.42], 'E': [1.66], 'G': [0.58], 'H': [1.49], 'I': [0.99], 'L': [1.29], 'K': [1.03], 'M': [1.4], 'F': [1.15], 'P': [0.49], 'S': [0.83], 'T': [0.94], 'W': [1.33], 'Y': [0.49], 'V': [0.96]} ,
           'craj730102' :  {'A': [1.0], 'R': [0.74], 'N': [0.75], 'D': [0.89], 'C': [0.99], 'Q': [0.87], 'E': [0.37], 'G': [0.56], 'H': [0.36], 'I': [1.75], 'L': [1.53], 'K': [1.18], 'M': [1.4], 'F': [1.26], 'P': [0.36], 'S': [0.65], 'T': [1.15], 'W': [0.84], 'Y': [1.41], 'V': [1.61]} ,
           'craj730103' :  {'A': [0.6], 'R': [0.79], 'N': [1.42], 'D': [1.24], 'C': [1.29], 'Q': [0.92], 'E': [0.64], 'G': [1.38], 'H': [0.95], 'I': [0.67], 'L': [0.7], 'K': [1.1], 'M': [0.67], 'F': [1.05], 'P': [1.47], 'S': [1.26], 'T': [1.05], 'W': [1.23], 'Y': [1.35], 'V': [0.48]} ,
           'dawd720101' :  {'A': [2.5], 'R': [7.5], 'N': [5.0], 'D': [2.5], 'C': [3.0], 'Q': [6.0], 'E': [5.0], 'G': [0.5], 'H': [6.0], 'I': [5.5], 'L': [5.5], 'K': [7.0], 'M': [6.0], 'F': [6.5], 'P': [5.5], 'S': [3.0], 'T': [5.0], 'W': [7.0], 'Y': [7.0], 'V': [5.0]} ,
           'daym780101' :  {'A': [8.6], 'R': [4.9], 'N': [4.3], 'D': [5.5], 'C': [2.9], 'Q': [3.9], 'E': [6.0], 'G': [8.4], 'H': [2.0], 'I': [4.5], 'L': [7.4], 'K': [6.6], 'M': [1.7], 'F': [3.6], 'P': [5.2], 'S': [7.0], 'T': [6.1], 'W': [1.3], 'Y': [3.4], 'V': [6.6]} ,
           'daym780201' :  {'A': [100.0], 'R': [65.0], 'N': [134.0], 'D': [106.0], 'C': [20.0], 'Q': [93.0], 'E': [102.0], 'G': [49.0], 'H': [66.0], 'I': [96.0], 'L': [40.0], 'K': [56.0], 'M': [94.0], 'F': [41.0], 'P': [56.0], 'S': [120.0], 'T': [97.0], 'W': [18.0], 'Y': [41.0], 'V': [74.0]} ,
           'desm900101' :  {'A': [1.56], 'R': [0.59], 'N': [0.51], 'D': [0.23], 'C': [1.8], 'Q': [0.39], 'E': [0.19], 'G': [1.03], 'H': [1.0], 'I': [1.27], 'L': [1.38], 'K': [0.15], 'M': [1.93], 'F': [1.42], 'P': [0.27], 'S': [0.96], 'T': [1.11], 'W': [0.91], 'Y': [1.1], 'V': [1.58]} ,
           'desm900102' :  {'A': [1.26], 'R': [0.38], 'N': [0.59], 'D': [0.27], 'C': [1.6], 'Q': [0.39], 'E': [0.23], 'G': [1.08], 'H': [1.0], 'I': [1.44], 'L': [1.36], 'K': [0.33], 'M': [1.52], 'F': [1.46], 'P': [0.54], 'S': [0.98], 'T': [1.01], 'W': [1.06], 'Y': [0.89], 'V': [1.33]} ,
           'eisd840101' :  {'A': [0.25], 'R': [-1.76], 'N': [-0.64], 'D': [-0.72], 'C': [0.04], 'Q': [-0.69], 'E': [-0.62], 'G': [0.16], 'H': [-0.4], 'I': [0.73], 'L': [0.53], 'K': [-1.1], 'M': [0.26], 'F': [0.61], 'P': [-0.07], 'S': [-0.26], 'T': [-0.18], 'W': [0.37], 'Y': [0.02], 'V': [0.54]} ,
           'eisd860101' :  {'A': [0.67], 'R': [-2.1], 'N': [-0.6], 'D': [-1.2], 'C': [0.38], 'Q': [-0.22], 'E': [-0.76], 'G': [0.0], 'H': [0.64], 'I': [1.9], 'L': [1.9], 'K': [-0.57], 'M': [2.4], 'F': [2.3], 'P': [1.2], 'S': [0.01], 'T': [0.52], 'W': [2.6], 'Y': [1.6], 'V': [1.5]} ,
           'eisd860102' :  {'A': [0.0], 'R': [10.0], 'N': [1.3], 'D': [1.9], 'C': [0.17], 'Q': [1.9], 'E': [3.0], 'G': [0.0], 'H': [0.99], 'I': [1.2], 'L': [1.0], 'K': [5.7], 'M': [1.9], 'F': [1.1], 'P': [0.18], 'S': [0.73], 'T': [1.5], 'W': [1.6], 'Y': [1.8], 'V': [0.48]} ,
           'eisd860103' :  {'A': [0.0], 'R': [-0.96], 'N': [-0.86], 'D': [-0.98], 'C': [0.76], 'Q': [-1.0], 'E': [-0.89], 'G': [0.0], 'H': [-0.75], 'I': [0.99], 'L': [0.89], 'K': [-0.99], 'M': [0.94], 'F': [0.92], 'P': [0.22], 'S': [-0.67], 'T': [0.09], 'W': [0.67], 'Y': [-0.93], 'V': [0.84]} ,
           'fasg760101' :  {'A': [89.09], 'R': [174.2], 'N': [132.12], 'D': [133.1], 'C': [121.15], 'Q': [146.15], 'E': [147.13], 'G': [75.07], 'H': [155.16], 'I': [131.17], 'L': [131.17], 'K': [146.19], 'M': [149.21], 'F': [165.19], 'P': [115.13], 'S': [105.09], 'T': [119.12], 'W': [204.24], 'Y': [181.19], 'V': [117.15]} ,
           'fasg760102' :  {'A': [297.0], 'R': [238.0], 'N': [236.0], 'D': [270.0], 'C': [178.0], 'Q': [185.0], 'E': [249.0], 'G': [290.0], 'H': [277.0], 'I': [284.0], 'L': [337.0], 'K': [224.0], 'M': [283.0], 'F': [284.0], 'P': [222.0], 'S': [228.0], 'T': [253.0], 'W': [282.0], 'Y': [344.0], 'V': [293.0]} ,
           'fasg760103' :  {'A': [1.8], 'R': [12.5], 'N': [-5.6], 'D': [5.05], 'C': [-16.5], 'Q': [6.3], 'E': [12.0], 'G': [0.0], 'H': [-38.5], 'I': [12.4], 'L': [-11.0], 'K': [14.6], 'M': [-10.0], 'F': [-34.5], 'P': [-86.2], 'S': [-7.5], 'T': [-28.0], 'W': [-33.7], 'Y': [-10.0], 'V': [5.63]} ,
           'fasg760104' :  {'A': [9.69], 'R': [8.99], 'N': [8.8], 'D': [9.6], 'C': [8.35], 'Q': [9.13], 'E': [9.67], 'G': [9.78], 'H': [9.17], 'I': [9.68], 'L': [9.6], 'K': [9.18], 'M': [9.21], 'F': [9.18], 'P': [10.64], 'S': [9.21], 'T': [9.1], 'W': [9.44], 'Y': [9.11], 'V': [9.62]} ,
           'fasg760105' :  {'A': [2.34], 'R': [1.82], 'N': [2.02], 'D': [1.88], 'C': [1.92], 'Q': [2.17], 'E': [2.1], 'G': [2.35], 'H': [1.82], 'I': [2.36], 'L': [2.36], 'K': [2.16], 'M': [2.28], 'F': [2.16], 'P': [1.95], 'S': [2.19], 'T': [2.09], 'W': [2.43], 'Y': [2.2], 'V': [2.32]} ,
           'fauj830101' :  {'A': [0.31], 'R': [-1.01], 'N': [-0.6], 'D': [-0.77], 'C': [1.54], 'Q': [-0.22], 'E': [-0.64], 'G': [0.0], 'H': [0.13], 'I': [1.8], 'L': [1.7], 'K': [-0.99], 'M': [1.23], 'F': [1.79], 'P': [0.72], 'S': [-0.04], 'T': [0.26], 'W': [2.25], 'Y': [0.96], 'V': [1.22]} ,
           'fauj880101' :  {'A': [1.28], 'R': [2.34], 'N': [1.6], 'D': [1.6], 'C': [1.77], 'Q': [1.56], 'E': [1.56], 'G': [0.0], 'H': [2.99], 'I': [4.19], 'L': [2.59], 'K': [1.89], 'M': [2.35], 'F': [2.94], 'P': [2.67], 'S': [1.31], 'T': [3.03], 'W': [3.21], 'Y': [2.94], 'V': [3.67]} ,
           'fauj880102' :  {'A': [0.53], 'R': [0.69], 'N': [0.58], 'D': [0.59], 'C': [0.66], 'Q': [0.71], 'E': [0.72], 'G': [0.0], 'H': [0.64], 'I': [0.96], 'L': [0.92], 'K': [0.78], 'M': [0.77], 'F': [0.71], 'P': [0.0], 'S': [0.55], 'T': [0.63], 'W': [0.84], 'Y': [0.71], 'V': [0.89]} ,
           'fauj880103' :  {'A': [1.0], 'R': [6.13], 'N': [2.95], 'D': [2.78], 'C': [2.43], 'Q': [3.95], 'E': [3.78], 'G': [0.0], 'H': [4.66], 'I': [4.0], 'L': [4.0], 'K': [4.77], 'M': [4.43], 'F': [5.89], 'P': [2.72], 'S': [1.6], 'T': [2.6], 'W': [8.08], 'Y': [6.47], 'V': [3.0]} ,
           'fauj880104' :  {'A': [2.87], 'R': [7.82], 'N': [4.58], 'D': [4.74], 'C': [4.47], 'Q': [6.11], 'E': [5.97], 'G': [2.06], 'H': [5.23], 'I': [4.92], 'L': [4.92], 'K': [6.89], 'M': [6.36], 'F': [4.62], 'P': [4.11], 'S': [3.97], 'T': [4.11], 'W': [7.68], 'Y': [4.73], 'V': [4.11]} ,
           'fauj880105' :  {'A': [1.52], 'R': [1.52], 'N': [1.52], 'D': [1.52], 'C': [1.52], 'Q': [1.52], 'E': [1.52], 'G': [1.0], 'H': [1.52], 'I': [1.9], 'L': [1.52], 'K': [1.52], 'M': [1.52], 'F': [1.52], 'P': [1.52], 'S': [1.52], 'T': [1.73], 'W': [1.52], 'Y': [1.52], 'V': [1.9]} ,
           'fauj880106' :  {'A': [2.04], 'R': [6.24], 'N': [4.37], 'D': [3.78], 'C': [3.41], 'Q': [3.53], 'E': [3.31], 'G': [1.0], 'H': [5.66], 'I': [3.49], 'L': [4.45], 'K': [4.87], 'M': [4.8], 'F': [6.02], 'P': [4.31], 'S': [2.7], 'T': [3.17], 'W': [5.9], 'Y': [6.72], 'V': [3.17]} ,
           'fauj880107' :  {'A': [7.3], 'R': [11.1], 'N': [8.0], 'D': [9.2], 'C': [14.4], 'Q': [10.6], 'E': [11.4], 'G': [0.0], 'H': [10.2], 'I': [16.1], 'L': [10.1], 'K': [10.9], 'M': [10.4], 'F': [13.9], 'P': [17.8], 'S': [13.1], 'T': [16.7], 'W': [13.2], 'Y': [13.9], 'V': [17.2]} ,
           'fauj880108' :  {'A': [-0.01], 'R': [0.04], 'N': [0.06], 'D': [0.15], 'C': [0.12], 'Q': [0.05], 'E': [0.07], 'G': [0.0], 'H': [0.08], 'I': [-0.01], 'L': [-0.01], 'K': [0.0], 'M': [0.04], 'F': [0.03], 'P': [0.0], 'S': [0.11], 'T': [0.04], 'W': [0.0], 'Y': [0.03], 'V': [0.01]} ,
           'fauj880109' :  {'A': [0.0], 'R': [4.0], 'N': [2.0], 'D': [1.0], 'C': [0.0], 'Q': [2.0], 'E': [1.0], 'G': [0.0], 'H': [1.0], 'I': [0.0], 'L': [0.0], 'K': [2.0], 'M': [0.0], 'F': [0.0], 'P': [0.0], 'S': [1.0], 'T': [1.0], 'W': [1.0], 'Y': [1.0], 'V': [0.0]} ,
           'fauj880110' :  {'A': [0.0], 'R': [3.0], 'N': [3.0], 'D': [4.0], 'C': [0.0], 'Q': [3.0], 'E': [4.0], 'G': [0.0], 'H': [1.0], 'I': [0.0], 'L': [0.0], 'K': [1.0], 'M': [0.0], 'F': [0.0], 'P': [0.0], 'S': [2.0], 'T': [2.0], 'W': [0.0], 'Y': [2.0], 'V': [0.0]} ,
           'fauj880111' :  {'A': [0.0], 'R': [1.0], 'N': [0.0], 'D': [0.0], 'C': [0.0], 'Q': [0.0], 'E': [0.0], 'G': [0.0], 'H': [1.0], 'I': [0.0], 'L': [0.0], 'K': [1.0], 'M': [0.0], 'F': [0.0], 'P': [0.0], 'S': [0.0], 'T': [0.0], 'W': [0.0], 'Y': [0.0], 'V': [0.0]} ,
           'fauj880112' :  {'A': [0.0], 'R': [0.0], 'N': [0.0], 'D': [1.0], 'C': [0.0], 'Q': [0.0], 'E': [1.0], 'G': [0.0], 'H': [0.0], 'I': [0.0], 'L': [0.0], 'K': [0.0], 'M': [0.0], 'F': [0.0], 'P': [0.0], 'S': [0.0], 'T': [0.0], 'W': [0.0], 'Y': [0.0], 'V': [0.0]} ,
           'fauj880113' :  {'A': [4.76], 'R': [4.3], 'N': [3.64], 'D': [5.69], 'C': [3.67], 'Q': [4.54], 'E': [5.48], 'G': [3.77], 'H': [2.84], 'I': [4.81], 'L': [4.79], 'K': [4.27], 'M': [4.25], 'F': [4.31], 'P': [0.0], 'S': [3.83], 'T': [3.87], 'W': [4.75], 'Y': [4.3], 'V': [4.86]} ,
           'fina770101' :  {'A': [1.08], 'R': [1.05], 'N': [0.85], 'D': [0.85], 'C': [0.95], 'Q': [0.95], 'E': [1.15], 'G': [0.55], 'H': [1.0], 'I': [1.05], 'L': [1.25], 'K': [1.15], 'M': [1.15], 'F': [1.1], 'P': [0.71], 'S': [0.75], 'T': [0.75], 'W': [1.1], 'Y': [1.1], 'V': [0.95]} ,
           'fina910101' :  {'A': [1.0], 'R': [0.7], 'N': [1.7], 'D': [3.2], 'C': [1.0], 'Q': [1.0], 'E': [1.7], 'G': [1.0], 'H': [1.0], 'I': [0.6], 'L': [1.0], 'K': [0.7], 'M': [1.0], 'F': [1.0], 'P': [1.0], 'S': [1.7], 'T': [1.7], 'W': [1.0], 'Y': [1.0], 'V': [0.6]} ,
           'fina910102' :  {'A': [1.0], 'R': [0.7], 'N': [1.0], 'D': [1.7], 'C': [1.0], 'Q': [1.0], 'E': [1.7], 'G': [1.3], 'H': [1.0], 'I': [1.0], 'L': [1.0], 'K': [0.7], 'M': [1.0], 'F': [1.0], 'P': [13.0], 'S': [1.0], 'T': [1.0], 'W': [1.0], 'Y': [1.0], 'V': [1.0]} ,
           'fina910103' :  {'A': [1.2], 'R': [1.7], 'N': [1.2], 'D': [0.7], 'C': [1.0], 'Q': [1.0], 'E': [0.7], 'G': [0.8], 'H': [1.2], 'I': [0.8], 'L': [1.0], 'K': [1.7], 'M': [1.0], 'F': [1.0], 'P': [1.0], 'S': [1.5], 'T': [1.0], 'W': [1.0], 'Y': [1.0], 'V': [0.8]} ,
           'fina910104' :  {'A': [1.0], 'R': [1.7], 'N': [1.0], 'D': [0.7], 'C': [1.0], 'Q': [1.0], 'E': [0.7], 'G': [1.5], 'H': [1.0], 'I': [1.0], 'L': [1.0], 'K': [1.7], 'M': [1.0], 'F': [1.0], 'P': [0.1], 'S': [1.0], 'T': [1.0], 'W': [1.0], 'Y': [1.0], 'V': [1.0]} ,
           'garj730101' :  {'A': [0.28], 'R': [0.1], 'N': [0.25], 'D': [0.21], 'C': [0.28], 'Q': [0.35], 'E': [0.33], 'G': [0.17], 'H': [0.21], 'I': [0.82], 'L': [1.0], 'K': [0.09], 'M': [0.74], 'F': [2.18], 'P': [0.39], 'S': [0.12], 'T': [0.21], 'W': [5.7], 'Y': [1.26], 'V': [0.6]} ,
           'geim800101' :  {'A': [1.29], 'R': [1.0], 'N': [0.81], 'D': [1.1], 'C': [0.79], 'Q': [1.07], 'E': [1.49], 'G': [0.63], 'H': [1.33], 'I': [1.05], 'L': [1.31], 'K': [1.33], 'M': [1.54], 'F': [1.13], 'P': [0.63], 'S': [0.78], 'T': [0.77], 'W': [1.18], 'Y': [0.71], 'V': [0.81]} ,
           'geim800102' :  {'A': [1.13], 'R': [1.09], 'N': [1.06], 'D': [0.94], 'C': [1.32], 'Q': [0.93], 'E': [1.2], 'G': [0.83], 'H': [1.09], 'I': [1.05], 'L': [1.13], 'K': [1.08], 'M': [1.23], 'F': [1.01], 'P': [0.82], 'S': [1.01], 'T': [1.17], 'W': [1.32], 'Y': [0.88], 'V': [1.13]} ,
           'geim800103' :  {'A': [1.55], 'R': [0.2], 'N': [1.2], 'D': [1.55], 'C': [1.44], 'Q': [1.13], 'E': [1.67], 'G': [0.59], 'H': [1.21], 'I': [1.27], 'L': [1.25], 'K': [1.2], 'M': [1.37], 'F': [0.4], 'P': [0.21], 'S': [1.01], 'T': [0.55], 'W': [1.86], 'Y': [1.08], 'V': [0.64]} ,
           'geim800104' :  {'A': [1.19], 'R': [1.0], 'N': [0.94], 'D': [1.07], 'C': [0.95], 'Q': [1.32], 'E': [1.64], 'G': [0.6], 'H': [1.03], 'I': [1.12], 'L': [1.18], 'K': [1.27], 'M': [1.49], 'F': [1.02], 'P': [0.68], 'S': [0.81], 'T': [0.85], 'W': [1.18], 'Y': [0.77], 'V': [0.74]} ,
           'geim800105' :  {'A': [0.84], 'R': [1.04], 'N': [0.66], 'D': [0.59], 'C': [1.27], 'Q': [1.02], 'E': [0.57], 'G': [0.94], 'H': [0.81], 'I': [1.29], 'L': [1.1], 'K': [0.86], 'M': [0.88], 'F': [1.15], 'P': [0.8], 'S': [1.05], 'T': [1.2], 'W': [1.15], 'Y': [1.39], 'V': [1.56]} ,
           'geim800106' :  {'A': [0.86], 'R': [1.15], 'N': [0.6], 'D': [0.66], 'C': [0.91], 'Q': [1.11], 'E': [0.37], 'G': [0.86], 'H': [1.07], 'I': [1.17], 'L': [1.28], 'K': [1.01], 'M': [1.15], 'F': [1.34], 'P': [0.61], 'S': [0.91], 'T': [1.14], 'W': [1.13], 'Y': [1.37], 'V': [1.31]} ,
           'geim800107' :  {'A': [0.91], 'R': [0.99], 'N': [0.72], 'D': [0.74], 'C': [1.12], 'Q': [0.9], 'E': [0.41], 'G': [0.91], 'H': [1.01], 'I': [1.29], 'L': [1.23], 'K': [0.86], 'M': [0.96], 'F': [1.26], 'P': [0.65], 'S': [0.93], 'T': [1.05], 'W': [1.15], 'Y': [1.21], 'V': [1.58]} ,
           'geim800108' :  {'A': [0.91], 'R': [1.0], 'N': [1.64], 'D': [1.4], 'C': [0.93], 'Q': [0.94], 'E': [0.97], 'G': [1.51], 'H': [0.9], 'I': [0.65], 'L': [0.59], 'K': [0.82], 'M': [0.58], 'F': [0.72], 'P': [1.66], 'S': [1.23], 'T': [1.04], 'W': [0.67], 'Y': [0.92], 'V': [0.6]} ,
           'geim800109' :  {'A': [0.8], 'R': [0.96], 'N': [1.1], 'D': [1.6], 'C': [0.0], 'Q': [1.6], 'E': [0.4], 'G': [2.0], 'H': [0.96], 'I': [0.85], 'L': [0.8], 'K': [0.94], 'M': [0.39], 'F': [1.2], 'P': [2.1], 'S': [1.3], 'T': [0.6], 'W': [0.0], 'Y': [1.8], 'V': [0.8]} ,
           'geim800110' :  {'A': [1.1], 'R': [0.93], 'N': [1.57], 'D': [1.41], 'C': [1.05], 'Q': [0.81], 'E': [1.4], 'G': [1.3], 'H': [0.85], 'I': [0.67], 'L': [0.52], 'K': [0.94], 'M': [0.69], 'F': [0.6], 'P': [1.77], 'S': [1.13], 'T': [0.88], 'W': [0.62], 'Y': [0.41], 'V': [0.58]} ,
           'geim800111' :  {'A': [0.93], 'R': [1.01], 'N': [1.36], 'D': [1.22], 'C': [0.92], 'Q': [0.83], 'E': [1.05], 'G': [1.45], 'H': [0.96], 'I': [0.58], 'L': [0.59], 'K': [0.91], 'M': [0.6], 'F': [0.71], 'P': [1.67], 'S': [1.25], 'T': [1.08], 'W': [0.68], 'Y': [0.98], 'V': [0.62]} ,
           'gold730101' :  {'A': [0.75], 'R': [0.75], 'N': [0.69], 'D': [0.0], 'C': [1.0], 'Q': [0.59], 'E': [0.0], 'G': [0.0], 'H': [0.0], 'I': [2.95], 'L': [2.4], 'K': [1.5], 'M': [1.3], 'F': [2.65], 'P': [2.6], 'S': [0.0], 'T': [0.45], 'W': [3.0], 'Y': [2.85], 'V': [1.7]} ,
           'gold730102' :  {'A': [88.3], 'R': [181.2], 'N': [125.1], 'D': [110.8], 'C': [112.4], 'Q': [148.7], 'E': [140.5], 'G': [60.0], 'H': [152.6], 'I': [168.5], 'L': [168.5], 'K': [175.6], 'M': [162.2], 'F': [189.0], 'P': [122.2], 'S': [88.7], 'T': [118.2], 'W': [227.0], 'Y': [193.0], 'V': [141.4]} ,
           'grar740101' :  {'A': [0.0], 'R': [0.65], 'N': [1.33], 'D': [1.38], 'C': [2.75], 'Q': [0.89], 'E': [0.92], 'G': [0.74], 'H': [0.58], 'I': [0.0], 'L': [0.0], 'K': [0.33], 'M': [0.0], 'F': [0.0], 'P': [0.39], 'S': [1.42], 'T': [0.71], 'W': [0.13], 'Y': [0.2], 'V': [0.0]} ,
           'grar740102' :  {'A': [8.1], 'R': [10.5], 'N': [11.6], 'D': [13.0], 'C': [5.5], 'Q': [10.5], 'E': [12.3], 'G': [9.0], 'H': [10.4], 'I': [5.2], 'L': [4.9], 'K': [11.3], 'M': [5.7], 'F': [5.2], 'P': [8.0], 'S': [9.2], 'T': [8.6], 'W': [5.4], 'Y': [6.2], 'V': [5.9]} ,
           'grar740103' :  {'A': [31.0], 'R': [124.0], 'N': [56.0], 'D': [54.0], 'C': [55.0], 'Q': [85.0], 'E': [83.0], 'G': [3.0], 'H': [96.0], 'I': [111.0], 'L': [111.0], 'K': [119.0], 'M': [105.0], 'F': [132.0], 'P': [32.5], 'S': [32.0], 'T': [61.0], 'W': [170.0], 'Y': [136.0], 'V': [84.0]} ,
           'guyh850101' :  {'A': [0.1], 'R': [1.91], 'N': [0.48], 'D': [0.78], 'C': [-1.42], 'Q': [0.95], 'E': [0.83], 'G': [0.33], 'H': [-0.5], 'I': [-1.13], 'L': [-1.18], 'K': [1.4], 'M': [-1.59], 'F': [-2.12], 'P': [0.73], 'S': [0.52], 'T': [0.07], 'W': [-0.51], 'Y': [-0.21], 'V': [-1.27]} ,
           'hopa770101' :  {'A': [1.0], 'R': [2.3], 'N': [2.2], 'D': [6.5], 'C': [0.1], 'Q': [2.1], 'E': [6.2], 'G': [1.1], 'H': [2.8], 'I': [0.8], 'L': [0.8], 'K': [5.3], 'M': [0.7], 'F': [1.4], 'P': [0.9], 'S': [1.7], 'T': [1.5], 'W': [1.9], 'Y': [2.1], 'V': [0.9]} ,
           'hopt810101' :  {'A': [-0.5], 'R': [3.0], 'N': [0.2], 'D': [3.0], 'C': [-1.0], 'Q': [0.2], 'E': [3.0], 'G': [0.0], 'H': [-0.5], 'I': [-1.8], 'L': [-1.8], 'K': [3.0], 'M': [-1.3], 'F': [-2.5], 'P': [0.0], 'S': [0.3], 'T': [-0.4], 'W': [-3.4], 'Y': [-2.3], 'V': [-1.5]} ,
           'hutj700101' :  {'A': [29.22], 'R': [26.37], 'N': [38.3], 'D': [37.09], 'C': [50.7], 'Q': [44.02], 'E': [41.84], 'G': [23.71], 'H': [59.64], 'I': [45.0], 'L': [48.03], 'K': [57.1], 'M': [69.32], 'F': [48.52], 'P': [36.13], 'S': [32.4], 'T': [35.2], 'W': [56.92], 'Y': [51.73], 'V': [40.35]} ,
           'hutj700102' :  {'A': [30.88], 'R': [68.43], 'N': [41.7], 'D': [40.66], 'C': [53.83], 'Q': [46.62], 'E': [44.98], 'G': [24.74], 'H': [65.99], 'I': [49.71], 'L': [50.62], 'K': [63.21], 'M': [55.32], 'F': [51.06], 'P': [39.21], 'S': [35.65], 'T': [36.5], 'W': [60.0], 'Y': [51.15], 'V': [42.75]} ,
           'hutj700103' :  {'A': [154.33], 'R': [341.01], 'N': [207.9], 'D': [194.91], 'C': [219.79], 'Q': [235.51], 'E': [223.16], 'G': [127.9], 'H': [242.54], 'I': [233.21], 'L': [232.3], 'K': [300.46], 'M': [202.65], 'F': [204.74], 'P': [179.93], 'S': [174.06], 'T': [205.8], 'W': [237.01], 'Y': [229.15], 'V': [207.6]} ,
           'isoy800101' :  {'A': [1.53], 'R': [1.17], 'N': [0.6], 'D': [1.0], 'C': [0.89], 'Q': [1.27], 'E': [1.63], 'G': [0.44], 'H': [1.03], 'I': [1.07], 'L': [1.32], 'K': [1.26], 'M': [1.66], 'F': [1.22], 'P': [0.25], 'S': [0.65], 'T': [0.86], 'W': [1.05], 'Y': [0.7], 'V': [0.93]} ,
           'isoy800102' :  {'A': [0.86], 'R': [0.98], 'N': [0.74], 'D': [0.69], 'C': [1.39], 'Q': [0.89], 'E': [0.66], 'G': [0.7], 'H': [1.06], 'I': [1.31], 'L': [1.01], 'K': [0.77], 'M': [1.06], 'F': [1.16], 'P': [1.16], 'S': [1.09], 'T': [1.24], 'W': [1.17], 'Y': [1.28], 'V': [1.4]} ,
           'isoy800103' :  {'A': [0.78], 'R': [1.06], 'N': [1.56], 'D': [1.5], 'C': [0.6], 'Q': [0.78], 'E': [0.97], 'G': [1.73], 'H': [0.83], 'I': [0.4], 'L': [0.57], 'K': [1.01], 'M': [0.3], 'F': [0.67], 'P': [1.55], 'S': [1.19], 'T': [1.09], 'W': [0.74], 'Y': [1.14], 'V': [0.44]} ,
           'isoy800104' :  {'A': [1.09], 'R': [0.97], 'N': [1.14], 'D': [0.77], 'C': [0.5], 'Q': [0.83], 'E': [0.92], 'G': [1.25], 'H': [0.67], 'I': [0.66], 'L': [0.44], 'K': [1.25], 'M': [0.45], 'F': [0.5], 'P': [2.96], 'S': [1.21], 'T': [1.33], 'W': [0.62], 'Y': [0.94], 'V': [0.56]} ,
           'isoy800105' :  {'A': [0.35], 'R': [0.75], 'N': [2.12], 'D': [2.16], 'C': [0.5], 'Q': [0.73], 'E': [0.65], 'G': [2.4], 'H': [1.19], 'I': [0.12], 'L': [0.58], 'K': [0.83], 'M': [0.22], 'F': [0.89], 'P': [0.43], 'S': [1.24], 'T': [0.85], 'W': [0.62], 'Y': [1.44], 'V': [0.43]} ,
           'isoy800106' :  {'A': [1.09], 'R': [1.07], 'N': [0.88], 'D': [1.24], 'C': [1.04], 'Q': [1.09], 'E': [1.14], 'G': [0.27], 'H': [1.07], 'I': [0.97], 'L': [1.3], 'K': [1.2], 'M': [0.55], 'F': [0.8], 'P': [1.78], 'S': [1.2], 'T': [0.99], 'W': [1.03], 'Y': [0.69], 'V': [0.77]} ,
           'isoy800107' :  {'A': [1.34], 'R': [2.78], 'N': [0.92], 'D': [1.77], 'C': [1.44], 'Q': [0.79], 'E': [2.54], 'G': [0.95], 'H': [0.0], 'I': [0.52], 'L': [1.05], 'K': [0.79], 'M': [0.0], 'F': [0.43], 'P': [0.37], 'S': [0.87], 'T': [1.14], 'W': [1.79], 'Y': [0.73], 'V': [0.0]} ,
           'isoy800108' :  {'A': [0.47], 'R': [0.52], 'N': [2.16], 'D': [1.15], 'C': [0.41], 'Q': [0.95], 'E': [0.64], 'G': [3.03], 'H': [0.89], 'I': [0.62], 'L': [0.53], 'K': [0.98], 'M': [0.68], 'F': [0.61], 'P': [0.63], 'S': [1.03], 'T': [0.39], 'W': [0.63], 'Y': [0.83], 'V': [0.76]} ,
           'janj780101' :  {'A': [27.8], 'R': [94.7], 'N': [60.1], 'D': [60.6], 'C': [15.5], 'Q': [68.7], 'E': [68.2], 'G': [24.5], 'H': [50.7], 'I': [22.8], 'L': [27.6], 'K': [103.0], 'M': [33.5], 'F': [25.5], 'P': [51.5], 'S': [42.0], 'T': [45.0], 'W': [34.7], 'Y': [55.2], 'V': [23.7]} ,
           'janj780102' :  {'A': [51.0], 'R': [5.0], 'N': [22.0], 'D': [19.0], 'C': [74.0], 'Q': [16.0], 'E': [16.0], 'G': [52.0], 'H': [34.0], 'I': [66.0], 'L': [60.0], 'K': [3.0], 'M': [52.0], 'F': [58.0], 'P': [25.0], 'S': [35.0], 'T': [30.0], 'W': [49.0], 'Y': [24.0], 'V': [64.0]} ,
           'janj780103' :  {'A': [15.0], 'R': [67.0], 'N': [49.0], 'D': [50.0], 'C': [5.0], 'Q': [56.0], 'E': [55.0], 'G': [10.0], 'H': [34.0], 'I': [13.0], 'L': [16.0], 'K': [85.0], 'M': [20.0], 'F': [10.0], 'P': [45.0], 'S': [32.0], 'T': [32.0], 'W': [17.0], 'Y': [41.0], 'V': [14.0]} ,
           'janj790101' :  {'A': [1.7], 'R': [0.1], 'N': [0.4], 'D': [0.4], 'C': [4.6], 'Q': [0.3], 'E': [0.3], 'G': [1.8], 'H': [0.8], 'I': [3.1], 'L': [2.4], 'K': [0.05], 'M': [1.9], 'F': [2.2], 'P': [0.6], 'S': [0.8], 'T': [0.7], 'W': [1.6], 'Y': [0.5], 'V': [2.9]} ,
           'janj790102' :  {'A': [0.3], 'R': [-1.4], 'N': [-0.5], 'D': [-0.6], 'C': [0.9], 'Q': [-0.7], 'E': [-0.7], 'G': [0.3], 'H': [-0.1], 'I': [0.7], 'L': [0.5], 'K': [-1.8], 'M': [0.4], 'F': [0.5], 'P': [-0.3], 'S': [-0.1], 'T': [-0.2], 'W': [0.3], 'Y': [-0.4], 'V': [0.6]} ,
           'jond750101' :  {'A': [0.87], 'R': [0.85], 'N': [0.09], 'D': [0.66], 'C': [1.52], 'Q': [0.0], 'E': [0.67], 'G': [0.1], 'H': [0.87], 'I': [3.15], 'L': [2.17], 'K': [1.64], 'M': [1.67], 'F': [2.87], 'P': [2.77], 'S': [0.07], 'T': [0.07], 'W': [3.77], 'Y': [2.67], 'V': [1.87]} ,
           'jond750102' :  {'A': [2.34], 'R': [1.18], 'N': [2.02], 'D': [2.01], 'C': [1.65], 'Q': [2.17], 'E': [2.19], 'G': [2.34], 'H': [1.82], 'I': [2.36], 'L': [2.36], 'K': [2.18], 'M': [2.28], 'F': [1.83], 'P': [1.99], 'S': [2.21], 'T': [2.1], 'W': [2.38], 'Y': [2.2], 'V': [2.32]} ,
           'jond920101' :  {'A': [0.077], 'R': [0.051], 'N': [0.043], 'D': [0.052], 'C': [0.02], 'Q': [0.041], 'E': [0.062], 'G': [0.074], 'H': [0.023], 'I': [0.053], 'L': [0.091], 'K': [0.059], 'M': [0.024], 'F': [0.04], 'P': [0.051], 'S': [0.069], 'T': [0.059], 'W': [0.014], 'Y': [0.032], 'V': [0.066]} ,
           'jond920102' :  {'A': [100.0], 'R': [83.0], 'N': [104.0], 'D': [86.0], 'C': [44.0], 'Q': [84.0], 'E': [77.0], 'G': [50.0], 'H': [91.0], 'I': [103.0], 'L': [54.0], 'K': [72.0], 'M': [93.0], 'F': [51.0], 'P': [58.0], 'S': [117.0], 'T': [107.0], 'W': [25.0], 'Y': [50.0], 'V': [98.0]} ,
           'jukt750101' :  {'A': [5.3], 'R': [2.6], 'N': [3.0], 'D': [3.6], 'C': [1.3], 'Q': [2.4], 'E': [3.3], 'G': [4.8], 'H': [1.4], 'I': [3.1], 'L': [4.7], 'K': [4.1], 'M': [1.1], 'F': [2.3], 'P': [2.5], 'S': [4.5], 'T': [3.7], 'W': [0.8], 'Y': [2.3], 'V': [4.2]} ,
           'junj780101' :  {'A': [685.0], 'R': [382.0], 'N': [397.0], 'D': [400.0], 'C': [241.0], 'Q': [313.0], 'E': [427.0], 'G': [707.0], 'H': [155.0], 'I': [394.0], 'L': [581.0], 'K': [575.0], 'M': [132.0], 'F': [303.0], 'P': [366.0], 'S': [593.0], 'T': [490.0], 'W': [99.0], 'Y': [292.0], 'V': [553.0]} ,
           'kanm800101' :  {'A': [1.36], 'R': [1.0], 'N': [0.89], 'D': [1.04], 'C': [0.82], 'Q': [1.14], 'E': [1.48], 'G': [0.63], 'H': [1.11], 'I': [1.08], 'L': [1.21], 'K': [1.22], 'M': [1.45], 'F': [1.05], 'P': [0.52], 'S': [0.74], 'T': [0.81], 'W': [0.97], 'Y': [0.79], 'V': [0.94]} ,
           'kanm800102' :  {'A': [0.81], 'R': [0.85], 'N': [0.62], 'D': [0.71], 'C': [1.17], 'Q': [0.98], 'E': [0.53], 'G': [0.88], 'H': [0.92], 'I': [1.48], 'L': [1.24], 'K': [0.77], 'M': [1.05], 'F': [1.2], 'P': [0.61], 'S': [0.92], 'T': [1.18], 'W': [1.18], 'Y': [1.23], 'V': [1.66]} ,
           'kanm800103' :  {'A': [1.45], 'R': [1.15], 'N': [0.64], 'D': [0.91], 'C': [0.7], 'Q': [1.14], 'E': [1.29], 'G': [0.53], 'H': [1.13], 'I': [1.23], 'L': [1.56], 'K': [1.27], 'M': [1.83], 'F': [1.2], 'P': [0.21], 'S': [0.48], 'T': [0.77], 'W': [1.17], 'Y': [0.74], 'V': [1.1]} ,
           'kanm800104' :  {'A': [0.75], 'R': [0.79], 'N': [0.33], 'D': [0.31], 'C': [1.46], 'Q': [0.75], 'E': [0.46], 'G': [0.83], 'H': [0.83], 'I': [1.87], 'L': [1.56], 'K': [0.66], 'M': [0.86], 'F': [1.37], 'P': [0.52], 'S': [0.82], 'T': [1.36], 'W': [0.79], 'Y': [1.08], 'V': [2.0]} ,
           'karp850101' :  {'A': [1.041], 'R': [1.038], 'N': [1.117], 'D': [1.033], 'C': [0.96], 'Q': [1.165], 'E': [1.094], 'G': [1.142], 'H': [0.982], 'I': [1.002], 'L': [0.967], 'K': [1.093], 'M': [0.947], 'F': [0.93], 'P': [1.055], 'S': [1.169], 'T': [1.073], 'W': [0.925], 'Y': [0.961], 'V': [0.982]} ,
           'karp850102' :  {'A': [0.946], 'R': [1.028], 'N': [1.006], 'D': [1.089], 'C': [0.878], 'Q': [1.025], 'E': [1.036], 'G': [1.042], 'H': [0.952], 'I': [0.892], 'L': [0.961], 'K': [1.082], 'M': [0.862], 'F': [0.912], 'P': [1.085], 'S': [1.048], 'T': [1.051], 'W': [0.917], 'Y': [0.93], 'V': [0.927]} ,
           'karp850103' :  {'A': [0.892], 'R': [0.901], 'N': [0.93], 'D': [0.932], 'C': [0.925], 'Q': [0.885], 'E': [0.933], 'G': [0.923], 'H': [0.894], 'I': [0.872], 'L': [0.921], 'K': [1.057], 'M': [0.804], 'F': [0.914], 'P': [0.932], 'S': [0.923], 'T': [0.934], 'W': [0.803], 'Y': [0.837], 'V': [0.913]} ,
           'khag800101' :  {'A': [49.1], 'R': [133.0], 'N': [-3.6], 'D': [0.0], 'C': [0.0], 'Q': [20.0], 'E': [0.0], 'G': [64.6], 'H': [75.7], 'I': [18.9], 'L': [15.6], 'K': [0.0], 'M': [6.8], 'F': [54.7], 'P': [43.8], 'S': [44.4], 'T': [31.0], 'W': [70.5], 'Y': [0.0], 'V': [29.5]} ,
           'klep840101' :  {'A': [0.0], 'R': [1.0], 'N': [0.0], 'D': [-1.0], 'C': [0.0], 'Q': [0.0], 'E': [-1.0], 'G': [0.0], 'H': [0.0], 'I': [0.0], 'L': [0.0], 'K': [1.0], 'M': [0.0], 'F': [0.0], 'P': [0.0], 'S': [0.0], 'T': [0.0], 'W': [0.0], 'Y': [0.0], 'V': [0.0]} ,
           'kriw710101' :  {'A': [4.6], 'R': [6.5], 'N': [5.9], 'D': [5.7], 'C': [-1.0], 'Q': [6.1], 'E': [5.6], 'G': [7.6], 'H': [4.5], 'I': [2.6], 'L': [3.25], 'K': [7.9], 'M': [1.4], 'F': [3.2], 'P': [7.0], 'S': [5.25], 'T': [4.8], 'W': [4.0], 'Y': [4.35], 'V': [3.4]} ,
           'kriw790101' :  {'A': [4.32], 'R': [6.55], 'N': [6.24], 'D': [6.04], 'C': [1.73], 'Q': [6.13], 'E': [6.17], 'G': [6.09], 'H': [5.66], 'I': [2.31], 'L': [3.93], 'K': [7.92], 'M': [2.44], 'F': [2.59], 'P': [7.19], 'S': [5.37], 'T': [5.16], 'W': [2.78], 'Y': [3.58], 'V': [3.31]} ,
           'kriw790102' :  {'A': [0.28], 'R': [0.34], 'N': [0.31], 'D': [0.33], 'C': [0.11], 'Q': [0.39], 'E': [0.37], 'G': [0.28], 'H': [0.23], 'I': [0.12], 'L': [0.16], 'K': [0.59], 'M': [0.08], 'F': [0.1], 'P': [0.46], 'S': [0.27], 'T': [0.26], 'W': [0.15], 'Y': [0.25], 'V': [0.22]} ,
           'kriw790103' :  {'A': [27.5], 'R': [105.0], 'N': [58.7], 'D': [40.0], 'C': [44.6], 'Q': [80.7], 'E': [62.0], 'G': [0.0], 'H': [79.0], 'I': [93.5], 'L': [93.5], 'K': [100.0], 'M': [94.1], 'F': [115.5], 'P': [41.9], 'S': [29.3], 'T': [51.3], 'W': [145.5], 'Y': [117.3], 'V': [71.5]} ,
           'kytj820101' :  {'A': [1.8], 'R': [-4.5], 'N': [-3.5], 'D': [-3.5], 'C': [2.5], 'Q': [-3.5], 'E': [-3.5], 'G': [-0.4], 'H': [-3.2], 'I': [4.5], 'L': [3.8], 'K': [-3.9], 'M': [1.9], 'F': [2.8], 'P': [-1.6], 'S': [-0.8], 'T': [-0.7], 'W': [-0.9], 'Y': [-1.3], 'V': [4.2]} ,
           'lawe840101' :  {'A': [-0.48], 'R': [-0.06], 'N': [-0.87], 'D': [-0.75], 'C': [-0.32], 'Q': [-0.32], 'E': [-0.71], 'G': [0.0], 'H': [-0.51], 'I': [0.81], 'L': [1.02], 'K': [-0.09], 'M': [0.81], 'F': [1.03], 'P': [2.03], 'S': [0.05], 'T': [-0.35], 'W': [0.66], 'Y': [1.24], 'V': [0.56]} ,
           'levm760101' :  {'A': [-0.5], 'R': [3.0], 'N': [0.2], 'D': [2.5], 'C': [-1.0], 'Q': [0.2], 'E': [2.5], 'G': [0.0], 'H': [-0.5], 'I': [-1.8], 'L': [-1.8], 'K': [3.0], 'M': [-1.3], 'F': [-2.5], 'P': [-1.4], 'S': [0.3], 'T': [-0.4], 'W': [-3.4], 'Y': [-2.3], 'V': [-1.5]} ,
           'levm760102' :  {'A': [0.77], 'R': [3.72], 'N': [1.98], 'D': [1.99], 'C': [1.38], 'Q': [2.58], 'E': [2.63], 'G': [0.0], 'H': [2.76], 'I': [1.83], 'L': [2.08], 'K': [2.94], 'M': [2.34], 'F': [2.97], 'P': [1.42], 'S': [1.28], 'T': [1.43], 'W': [3.58], 'Y': [3.36], 'V': [1.49]} ,
           'levm760103' :  {'A': [121.9], 'R': [121.4], 'N': [117.5], 'D': [121.2], 'C': [113.7], 'Q': [118.0], 'E': [118.2], 'G': [0.0], 'H': [118.2], 'I': [118.9], 'L': [118.1], 'K': [122.0], 'M': [113.1], 'F': [118.2], 'P': [81.9], 'S': [117.9], 'T': [117.1], 'W': [118.4], 'Y': [110.0], 'V': [121.7]} ,
           'levm760104' :  {'A': [243.2], 'R': [206.6], 'N': [207.1], 'D': [215.0], 'C': [209.4], 'Q': [205.4], 'E': [213.6], 'G': [300.0], 'H': [219.9], 'I': [217.9], 'L': [205.6], 'K': [210.9], 'M': [204.0], 'F': [203.7], 'P': [237.4], 'S': [232.0], 'T': [226.7], 'W': [203.7], 'Y': [195.6], 'V': [220.3]} ,
           'levm760105' :  {'A': [0.77], 'R': [2.38], 'N': [1.45], 'D': [1.43], 'C': [1.22], 'Q': [1.75], 'E': [1.77], 'G': [0.58], 'H': [1.78], 'I': [1.56], 'L': [1.54], 'K': [2.08], 'M': [1.8], 'F': [1.9], 'P': [1.25], 'S': [1.08], 'T': [1.24], 'W': [2.21], 'Y': [2.13], 'V': [1.29]} ,
           'levm760106' :  {'A': [5.2], 'R': [6.0], 'N': [5.0], 'D': [5.0], 'C': [6.1], 'Q': [6.0], 'E': [6.0], 'G': [4.2], 'H': [6.0], 'I': [7.0], 'L': [7.0], 'K': [6.0], 'M': [6.8], 'F': [7.1], 'P': [6.2], 'S': [4.9], 'T': [5.0], 'W': [7.6], 'Y': [7.1], 'V': [6.4]} ,
           'levm760107' :  {'A': [0.025], 'R': [0.2], 'N': [0.1], 'D': [0.1], 'C': [0.1], 'Q': [0.1], 'E': [0.1], 'G': [0.025], 'H': [0.1], 'I': [0.19], 'L': [0.19], 'K': [0.2], 'M': [0.19], 'F': [0.39], 'P': [0.17], 'S': [0.025], 'T': [0.1], 'W': [0.56], 'Y': [0.39], 'V': [0.15]} ,
           'levm780101' :  {'A': [1.29], 'R': [0.96], 'N': [0.9], 'D': [1.04], 'C': [1.11], 'Q': [1.27], 'E': [1.44], 'G': [0.56], 'H': [1.22], 'I': [0.97], 'L': [1.3], 'K': [1.23], 'M': [1.47], 'F': [1.07], 'P': [0.52], 'S': [0.82], 'T': [0.82], 'W': [0.99], 'Y': [0.72], 'V': [0.91]} ,
           'levm780102' :  {'A': [0.9], 'R': [0.99], 'N': [0.76], 'D': [0.72], 'C': [0.74], 'Q': [0.8], 'E': [0.75], 'G': [0.92], 'H': [1.08], 'I': [1.45], 'L': [1.02], 'K': [0.77], 'M': [0.97], 'F': [1.32], 'P': [0.64], 'S': [0.95], 'T': [1.21], 'W': [1.14], 'Y': [1.25], 'V': [1.49]} ,
           'levm780103' :  {'A': [0.77], 'R': [0.88], 'N': [1.28], 'D': [1.41], 'C': [0.81], 'Q': [0.98], 'E': [0.99], 'G': [1.64], 'H': [0.68], 'I': [0.51], 'L': [0.58], 'K': [0.96], 'M': [0.41], 'F': [0.59], 'P': [1.91], 'S': [1.32], 'T': [1.04], 'W': [0.76], 'Y': [1.05], 'V': [0.47]} ,
           'levm780104' :  {'A': [1.32], 'R': [0.98], 'N': [0.95], 'D': [1.03], 'C': [0.92], 'Q': [1.1], 'E': [1.44], 'G': [0.61], 'H': [1.31], 'I': [0.93], 'L': [1.31], 'K': [1.25], 'M': [1.39], 'F': [1.02], 'P': [0.58], 'S': [0.76], 'T': [0.79], 'W': [0.97], 'Y': [0.73], 'V': [0.93]} ,
           'levm780105' :  {'A': [0.86], 'R': [0.97], 'N': [0.73], 'D': [0.69], 'C': [1.04], 'Q': [1.0], 'E': [0.66], 'G': [0.89], 'H': [0.85], 'I': [1.47], 'L': [1.04], 'K': [0.77], 'M': [0.93], 'F': [1.21], 'P': [0.68], 'S': [1.02], 'T': [1.27], 'W': [1.26], 'Y': [1.31], 'V': [1.43]} ,
           'levm780106' :  {'A': [0.79], 'R': [0.9], 'N': [1.25], 'D': [1.47], 'C': [0.79], 'Q': [0.92], 'E': [1.02], 'G': [1.67], 'H': [0.81], 'I': [0.5], 'L': [0.57], 'K': [0.99], 'M': [0.51], 'F': [0.77], 'P': [1.78], 'S': [1.3], 'T': [0.97], 'W': [0.79], 'Y': [0.93], 'V': [0.46]} ,
           'lewp710101' :  {'A': [0.22], 'R': [0.28], 'N': [0.42], 'D': [0.73], 'C': [0.2], 'Q': [0.26], 'E': [0.08], 'G': [0.58], 'H': [0.14], 'I': [0.22], 'L': [0.19], 'K': [0.27], 'M': [0.38], 'F': [0.08], 'P': [0.46], 'S': [0.55], 'T': [0.49], 'W': [0.43], 'Y': [0.46], 'V': [0.08]} ,
           'lifs790101' :  {'A': [0.92], 'R': [0.93], 'N': [0.6], 'D': [0.48], 'C': [1.16], 'Q': [0.95], 'E': [0.61], 'G': [0.61], 'H': [0.93], 'I': [1.81], 'L': [1.3], 'K': [0.7], 'M': [1.19], 'F': [1.25], 'P': [0.4], 'S': [0.82], 'T': [1.12], 'W': [1.54], 'Y': [1.53], 'V': [1.81]} ,
           'lifs790102' :  {'A': [1.0], 'R': [0.68], 'N': [0.54], 'D': [0.5], 'C': [0.91], 'Q': [0.28], 'E': [0.59], 'G': [0.79], 'H': [0.38], 'I': [2.6], 'L': [1.42], 'K': [0.59], 'M': [1.49], 'F': [1.3], 'P': [0.35], 'S': [0.7], 'T': [0.59], 'W': [0.89], 'Y': [1.08], 'V': [2.63]} ,
           'lifs790103' :  {'A': [0.9], 'R': [1.02], 'N': [0.62], 'D': [0.47], 'C': [1.24], 'Q': [1.18], 'E': [0.62], 'G': [0.56], 'H': [1.12], 'I': [1.54], 'L': [1.26], 'K': [0.74], 'M': [1.09], 'F': [1.23], 'P': [0.42], 'S': [0.87], 'T': [1.3], 'W': [1.75], 'Y': [1.68], 'V': [1.53]} ,
           'manp780101' :  {'A': [12.97], 'R': [11.72], 'N': [11.42], 'D': [10.85], 'C': [14.63], 'Q': [11.76], 'E': [11.89], 'G': [12.43], 'H': [12.16], 'I': [15.67], 'L': [14.9], 'K': [11.36], 'M': [14.39], 'F': [14.0], 'P': [11.37], 'S': [11.23], 'T': [11.69], 'W': [13.93], 'Y': [13.42], 'V': [15.71]} ,
           'maxf760101' :  {'A': [1.43], 'R': [1.18], 'N': [0.64], 'D': [0.92], 'C': [0.94], 'Q': [1.22], 'E': [1.67], 'G': [0.46], 'H': [0.98], 'I': [1.04], 'L': [1.36], 'K': [1.27], 'M': [1.53], 'F': [1.19], 'P': [0.49], 'S': [0.7], 'T': [0.78], 'W': [1.01], 'Y': [0.69], 'V': [0.98]} ,
           'maxf760102' :  {'A': [0.86], 'R': [0.94], 'N': [0.74], 'D': [0.72], 'C': [1.17], 'Q': [0.89], 'E': [0.62], 'G': [0.97], 'H': [1.06], 'I': [1.24], 'L': [0.98], 'K': [0.79], 'M': [1.08], 'F': [1.16], 'P': [1.22], 'S': [1.04], 'T': [1.18], 'W': [1.07], 'Y': [1.25], 'V': [1.33]} ,
           'maxf760103' :  {'A': [0.64], 'R': [0.62], 'N': [3.14], 'D': [1.92], 'C': [0.32], 'Q': [0.8], 'E': [1.01], 'G': [0.63], 'H': [2.05], 'I': [0.92], 'L': [0.37], 'K': [0.89], 'M': [1.07], 'F': [0.86], 'P': [0.5], 'S': [1.01], 'T': [0.92], 'W': [1.0], 'Y': [1.31], 'V': [0.87]} ,
           'maxf760104' :  {'A': [0.17], 'R': [0.76], 'N': [2.62], 'D': [1.08], 'C': [0.95], 'Q': [0.91], 'E': [0.28], 'G': [5.02], 'H': [0.57], 'I': [0.26], 'L': [0.21], 'K': [1.17], 'M': [0.0], 'F': [0.28], 'P': [0.12], 'S': [0.57], 'T': [0.23], 'W': [0.0], 'Y': [0.97], 'V': [0.24]} ,
           'maxf760105' :  {'A': [1.13], 'R': [0.48], 'N': [1.11], 'D': [1.18], 'C': [0.38], 'Q': [0.41], 'E': [1.02], 'G': [3.84], 'H': [0.3], 'I': [0.4], 'L': [0.65], 'K': [1.13], 'M': [0.0], 'F': [0.45], 'P': [0.0], 'S': [0.81], 'T': [0.71], 'W': [0.93], 'Y': [0.38], 'V': [0.48]} ,
           'maxf760106' :  {'A': [1.0], 'R': [1.18], 'N': [0.87], 'D': [1.39], 'C': [1.09], 'Q': [1.13], 'E': [1.04], 'G': [0.46], 'H': [0.71], 'I': [0.68], 'L': [1.01], 'K': [1.05], 'M': [0.36], 'F': [0.65], 'P': [1.95], 'S': [1.56], 'T': [1.23], 'W': [1.1], 'Y': [0.87], 'V': [0.58]} ,
           'mcmt640101' :  {'A': [4.34], 'R': [26.66], 'N': [13.28], 'D': [12.0], 'C': [35.77], 'Q': [17.56], 'E': [17.26], 'G': [0.0], 'H': [21.81], 'I': [19.06], 'L': [18.78], 'K': [21.29], 'M': [21.64], 'F': [29.4], 'P': [10.93], 'S': [6.35], 'T': [11.01], 'W': [42.53], 'Y': [31.53], 'V': [13.92]} ,
           'meej800101' :  {'A': [0.5], 'R': [0.8], 'N': [0.8], 'D': [-8.2], 'C': [-6.8], 'Q': [-4.8], 'E': [-16.9], 'G': [0.0], 'H': [-3.5], 'I': [13.9], 'L': [8.8], 'K': [0.1], 'M': [4.8], 'F': [13.2], 'P': [6.1], 'S': [1.2], 'T': [2.7], 'W': [14.9], 'Y': [6.1], 'V': [2.7]} ,
           'meej800102' :  {'A': [-0.1], 'R': [-4.5], 'N': [-1.6], 'D': [-2.8], 'C': [-2.2], 'Q': [-2.5], 'E': [-7.5], 'G': [-0.5], 'H': [0.8], 'I': [11.8], 'L': [10.0], 'K': [-3.2], 'M': [7.1], 'F': [13.9], 'P': [8.0], 'S': [-3.7], 'T': [1.5], 'W': [18.1], 'Y': [8.2], 'V': [3.3]} ,
           'meej810101' :  {'A': [1.1], 'R': [-0.4], 'N': [-4.2], 'D': [-1.6], 'C': [7.1], 'Q': [-2.9], 'E': [0.7], 'G': [-0.2], 'H': [-0.7], 'I': [8.5], 'L': [11.0], 'K': [-1.9], 'M': [5.4], 'F': [13.4], 'P': [4.4], 'S': [-3.2], 'T': [-1.7], 'W': [17.1], 'Y': [7.4], 'V': [5.9]} ,
           'meej810102' :  {'A': [1.0], 'R': [-2.0], 'N': [-3.0], 'D': [-0.5], 'C': [4.6], 'Q': [-2.0], 'E': [1.1], 'G': [0.2], 'H': [-2.2], 'I': [7.0], 'L': [9.6], 'K': [-3.0], 'M': [4.0], 'F': [12.6], 'P': [3.1], 'S': [-2.9], 'T': [-0.6], 'W': [15.1], 'Y': [6.7], 'V': [4.6]} ,
           'meih800101' :  {'A': [0.93], 'R': [0.98], 'N': [0.98], 'D': [1.01], 'C': [0.88], 'Q': [1.02], 'E': [1.02], 'G': [1.01], 'H': [0.89], 'I': [0.79], 'L': [0.85], 'K': [1.05], 'M': [0.84], 'F': [0.78], 'P': [1.0], 'S': [1.02], 'T': [0.99], 'W': [0.83], 'Y': [0.93], 'V': [0.81]} ,
           'meih800102' :  {'A': [0.94], 'R': [1.09], 'N': [1.04], 'D': [1.08], 'C': [0.84], 'Q': [1.11], 'E': [1.12], 'G': [1.01], 'H': [0.92], 'I': [0.76], 'L': [0.82], 'K': [1.23], 'M': [0.83], 'F': [0.73], 'P': [1.04], 'S': [1.04], 'T': [1.02], 'W': [0.87], 'Y': [1.03], 'V': [0.81]} ,
           'meih800103' :  {'A': [87.0], 'R': [81.0], 'N': [70.0], 'D': [71.0], 'C': [104.0], 'Q': [66.0], 'E': [72.0], 'G': [90.0], 'H': [90.0], 'I': [105.0], 'L': [104.0], 'K': [65.0], 'M': [100.0], 'F': [108.0], 'P': [78.0], 'S': [83.0], 'T': [83.0], 'W': [94.0], 'Y': [83.0], 'V': [94.0]} ,
           'miys850101' :  {'A': [2.36], 'R': [1.92], 'N': [1.7], 'D': [1.67], 'C': [3.36], 'Q': [1.75], 'E': [1.74], 'G': [2.06], 'H': [2.41], 'I': [4.17], 'L': [3.93], 'K': [1.23], 'M': [4.22], 'F': [4.37], 'P': [1.89], 'S': [1.81], 'T': [2.04], 'W': [3.82], 'Y': [2.91], 'V': [3.49]} ,
           'nagk730101' :  {'A': [1.29], 'R': [0.83], 'N': [0.77], 'D': [1.0], 'C': [0.94], 'Q': [1.1], 'E': [1.54], 'G': [0.72], 'H': [1.29], 'I': [0.94], 'L': [1.23], 'K': [1.23], 'M': [1.23], 'F': [1.23], 'P': [0.7], 'S': [0.78], 'T': [0.87], 'W': [1.06], 'Y': [0.63], 'V': [0.97]} ,
           'nagk730102' :  {'A': [0.96], 'R': [0.67], 'N': [0.72], 'D': [0.9], 'C': [1.13], 'Q': [1.18], 'E': [0.33], 'G': [0.9], 'H': [0.87], 'I': [1.54], 'L': [1.26], 'K': [0.81], 'M': [1.29], 'F': [1.37], 'P': [0.75], 'S': [0.77], 'T': [1.23], 'W': [1.13], 'Y': [1.07], 'V': [1.41]} ,
           'nagk730103' :  {'A': [0.72], 'R': [1.33], 'N': [1.38], 'D': [1.04], 'C': [1.01], 'Q': [0.81], 'E': [0.75], 'G': [1.35], 'H': [0.76], 'I': [0.8], 'L': [0.63], 'K': [0.84], 'M': [0.62], 'F': [0.58], 'P': [1.43], 'S': [1.34], 'T': [1.03], 'W': [0.87], 'Y': [1.35], 'V': [0.83]} ,
           'nakh900101' :  {'A': [7.99], 'R': [5.86], 'N': [4.33], 'D': [5.14], 'C': [1.81], 'Q': [3.98], 'E': [6.1], 'G': [6.91], 'H': [2.17], 'I': [5.48], 'L': [9.16], 'K': [6.01], 'M': [2.5], 'F': [3.83], 'P': [4.95], 'S': [6.84], 'T': [5.77], 'W': [1.34], 'Y': [3.15], 'V': [6.65]} ,
           'nakh900102' :  {'A': [3.73], 'R': [3.34], 'N': [2.33], 'D': [2.23], 'C': [2.3], 'Q': [2.36], 'E': [3.0], 'G': [3.36], 'H': [1.55], 'I': [2.52], 'L': [3.4], 'K': [3.36], 'M': [1.37], 'F': [1.94], 'P': [3.18], 'S': [2.83], 'T': [2.63], 'W': [1.15], 'Y': [1.76], 'V': [2.53]} ,
           'nakh900103' :  {'A': [5.74], 'R': [1.92], 'N': [5.25], 'D': [2.11], 'C': [1.03], 'Q': [2.3], 'E': [2.63], 'G': [5.66], 'H': [2.3], 'I': [9.12], 'L': [15.36], 'K': [3.2], 'M': [5.3], 'F': [6.51], 'P': [4.79], 'S': [7.55], 'T': [7.51], 'W': [2.51], 'Y': [4.08], 'V': [5.12]} ,
           'nakh900104' :  {'A': [-0.6], 'R': [-1.18], 'N': [0.39], 'D': [-1.36], 'C': [-0.34], 'Q': [-0.71], 'E': [-1.16], 'G': [-0.37], 'H': [0.08], 'I': [1.44], 'L': [1.82], 'K': [-0.84], 'M': [2.04], 'F': [1.38], 'P': [-0.05], 'S': [0.25], 'T': [0.66], 'W': [1.02], 'Y': [0.53], 'V': [-0.6]} ,
           'nakh900105' :  {'A': [5.88], 'R': [1.54], 'N': [4.38], 'D': [1.7], 'C': [1.11], 'Q': [2.3], 'E': [2.6], 'G': [5.29], 'H': [2.33], 'I': [8.78], 'L': [16.52], 'K': [2.58], 'M': [6.0], 'F': [6.58], 'P': [5.29], 'S': [7.68], 'T': [8.38], 'W': [2.89], 'Y': [3.51], 'V': [4.66]} ,
           'nakh900106' :  {'A': [-0.57], 'R': [-1.29], 'N': [0.02], 'D': [-1.54], 'C': [-0.3], 'Q': [-0.71], 'E': [-1.17], 'G': [-0.48], 'H': [0.1], 'I': [1.31], 'L': [2.16], 'K': [-1.02], 'M': [2.55], 'F': [1.42], 'P': [0.11], 'S': [0.3], 'T': [0.99], 'W': [1.35], 'Y': [0.2], 'V': [-0.79]} ,
           'nakh900107' :  {'A': [5.39], 'R': [2.81], 'N': [7.31], 'D': [3.07], 'C': [0.86], 'Q': [2.31], 'E': [2.7], 'G': [6.52], 'H': [2.23], 'I': [9.94], 'L': [12.64], 'K': [4.67], 'M': [3.68], 'F': [6.34], 'P': [3.62], 'S': [7.24], 'T': [5.44], 'W': [1.64], 'Y': [5.42], 'V': [6.18]} ,
           'nakh900108' :  {'A': [-0.7], 'R': [-0.91], 'N': [1.28], 'D': [-0.93], 'C': [-0.41], 'Q': [-0.71], 'E': [-1.13], 'G': [-0.12], 'H': [0.04], 'I': [1.77], 'L': [1.02], 'K': [-0.4], 'M': [0.86], 'F': [1.29], 'P': [-0.42], 'S': [0.14], 'T': [-0.13], 'W': [0.26], 'Y': [1.29], 'V': [-0.19]} ,
           'nakh900109' :  {'A': [9.25], 'R': [3.96], 'N': [3.71], 'D': [3.89], 'C': [1.07], 'Q': [3.17], 'E': [4.8], 'G': [8.51], 'H': [1.88], 'I': [6.47], 'L': [10.94], 'K': [3.5], 'M': [3.14], 'F': [6.36], 'P': [4.36], 'S': [6.26], 'T': [5.66], 'W': [2.22], 'Y': [3.28], 'V': [7.55]} ,
           'nakh900110' :  {'A': [0.34], 'R': [-0.57], 'N': [-0.27], 'D': [-0.56], 'C': [-0.32], 'Q': [-0.34], 'E': [-0.43], 'G': [0.48], 'H': [-0.19], 'I': [0.39], 'L': [0.52], 'K': [-0.75], 'M': [0.47], 'F': [1.3], 'P': [-0.19], 'S': [-0.2], 'T': [-0.04], 'W': [0.77], 'Y': [0.07], 'V': [0.36]} ,
           'nakh900111' :  {'A': [10.17], 'R': [1.21], 'N': [1.36], 'D': [1.18], 'C': [1.48], 'Q': [1.57], 'E': [1.15], 'G': [8.87], 'H': [1.07], 'I': [10.91], 'L': [16.22], 'K': [1.04], 'M': [4.12], 'F': [9.6], 'P': [2.24], 'S': [5.38], 'T': [5.61], 'W': [2.67], 'Y': [2.68], 'V': [11.44]} ,
           'nakh900112' :  {'A': [6.61], 'R': [0.41], 'N': [1.84], 'D': [0.59], 'C': [0.83], 'Q': [1.2], 'E': [1.63], 'G': [4.88], 'H': [1.14], 'I': [12.91], 'L': [21.66], 'K': [1.15], 'M': [7.17], 'F': [7.76], 'P': [3.51], 'S': [6.84], 'T': [8.89], 'W': [2.11], 'Y': [2.57], 'V': [6.3]} ,
           'nakh900113' :  {'A': [1.61], 'R': [0.4], 'N': [0.73], 'D': [0.75], 'C': [0.37], 'Q': [0.61], 'E': [1.5], 'G': [3.12], 'H': [0.46], 'I': [1.61], 'L': [1.37], 'K': [0.62], 'M': [1.59], 'F': [1.24], 'P': [0.67], 'S': [0.68], 'T': [0.92], 'W': [1.63], 'Y': [0.67], 'V': [1.3]} ,
           'nakh920101' :  {'A': [8.63], 'R': [6.75], 'N': [4.18], 'D': [6.24], 'C': [1.03], 'Q': [4.76], 'E': [7.82], 'G': [6.8], 'H': [2.7], 'I': [3.48], 'L': [8.44], 'K': [6.25], 'M': [2.14], 'F': [2.73], 'P': [6.28], 'S': [8.53], 'T': [4.43], 'W': [0.8], 'Y': [2.54], 'V': [5.44]} ,
           'nakh920102' :  {'A': [10.88], 'R': [6.01], 'N': [5.75], 'D': [6.13], 'C': [0.69], 'Q': [4.68], 'E': [9.34], 'G': [7.72], 'H': [2.15], 'I': [1.8], 'L': [8.03], 'K': [6.11], 'M': [3.79], 'F': [2.93], 'P': [7.21], 'S': [7.25], 'T': [3.51], 'W': [0.47], 'Y': [1.01], 'V': [4.57]} ,
           'nakh920103' :  {'A': [5.15], 'R': [4.38], 'N': [4.81], 'D': [5.75], 'C': [3.24], 'Q': [4.45], 'E': [7.05], 'G': [6.38], 'H': [2.69], 'I': [4.4], 'L': [8.11], 'K': [5.25], 'M': [1.6], 'F': [3.52], 'P': [5.65], 'S': [8.04], 'T': [7.41], 'W': [1.68], 'Y': [3.42], 'V': [7.0]} ,
           'nakh920104' :  {'A': [5.04], 'R': [3.73], 'N': [5.94], 'D': [5.26], 'C': [2.2], 'Q': [4.5], 'E': [6.07], 'G': [7.09], 'H': [2.99], 'I': [4.32], 'L': [9.88], 'K': [6.31], 'M': [1.85], 'F': [3.72], 'P': [6.22], 'S': [8.05], 'T': [5.2], 'W': [2.1], 'Y': [3.32], 'V': [6.19]} ,
           'nakh920105' :  {'A': [9.9], 'R': [0.09], 'N': [0.94], 'D': [0.35], 'C': [2.55], 'Q': [0.87], 'E': [0.08], 'G': [8.14], 'H': [0.2], 'I': [15.25], 'L': [22.28], 'K': [0.16], 'M': [1.85], 'F': [6.47], 'P': [2.38], 'S': [4.17], 'T': [4.33], 'W': [2.21], 'Y': [3.42], 'V': [14.34]} ,
           'nakh920106' :  {'A': [6.69], 'R': [6.65], 'N': [4.49], 'D': [4.97], 'C': [1.7], 'Q': [5.39], 'E': [7.76], 'G': [6.32], 'H': [2.11], 'I': [4.51], 'L': [8.23], 'K': [8.36], 'M': [2.46], 'F': [3.59], 'P': [5.2], 'S': [7.4], 'T': [5.18], 'W': [1.06], 'Y': [2.75], 'V': [5.27]} ,
           'nakh920107' :  {'A': [5.08], 'R': [4.75], 'N': [5.75], 'D': [5.96], 'C': [2.95], 'Q': [4.24], 'E': [6.04], 'G': [8.2], 'H': [2.1], 'I': [4.95], 'L': [8.03], 'K': [4.93], 'M': [2.61], 'F': [4.36], 'P': [4.84], 'S': [6.41], 'T': [5.87], 'W': [2.31], 'Y': [4.55], 'V': [6.07]} ,
           'nakh920108' :  {'A': [9.36], 'R': [0.27], 'N': [2.31], 'D': [0.94], 'C': [2.56], 'Q': [1.14], 'E': [0.94], 'G': [6.17], 'H': [0.47], 'I': [13.73], 'L': [16.64], 'K': [0.58], 'M': [3.93], 'F': [10.99], 'P': [1.96], 'S': [5.58], 'T': [4.68], 'W': [2.2], 'Y': [3.13], 'V': [12.43]} ,
           'nisk800101' :  {'A': [0.23], 'R': [-0.26], 'N': [-0.94], 'D': [-1.13], 'C': [1.78], 'Q': [-0.57], 'E': [-0.75], 'G': [-0.07], 'H': [0.11], 'I': [1.19], 'L': [1.03], 'K': [-1.05], 'M': [0.66], 'F': [0.48], 'P': [-0.76], 'S': [-0.67], 'T': [-0.36], 'W': [0.9], 'Y': [0.59], 'V': [1.24]} ,
           'nisk860101' :  {'A': [-0.22], 'R': [-0.93], 'N': [-2.65], 'D': [-4.12], 'C': [4.66], 'Q': [-2.76], 'E': [-3.64], 'G': [-1.62], 'H': [1.28], 'I': [5.58], 'L': [5.01], 'K': [-4.18], 'M': [3.51], 'F': [5.27], 'P': [-3.03], 'S': [-2.84], 'T': [-1.2], 'W': [5.2], 'Y': [2.15], 'V': [4.45]} ,
           'nozy710101' :  {'A': [0.5], 'R': [0.0], 'N': [0.0], 'D': [0.0], 'C': [0.0], 'Q': [0.0], 'E': [0.0], 'G': [0.0], 'H': [0.5], 'I': [1.8], 'L': [1.8], 'K': [0.0], 'M': [1.3], 'F': [2.5], 'P': [0.0], 'S': [0.0], 'T': [0.4], 'W': [3.4], 'Y': [2.3], 'V': [1.5]} ,
           'oobm770101' :  {'A': [-1.895], 'R': [-1.475], 'N': [-1.56], 'D': [-1.518], 'C': [-2.035], 'Q': [-1.521], 'E': [-1.535], 'G': [-1.898], 'H': [-1.755], 'I': [-1.951], 'L': [-1.966], 'K': [-1.374], 'M': [-1.963], 'F': [-1.864], 'P': [-1.699], 'S': [-1.753], 'T': [-1.767], 'W': [-1.869], 'Y': [-1.686], 'V': [-1.981]} ,
           'oobm770102' :  {'A': [-1.404], 'R': [-0.921], 'N': [-1.178], 'D': [-1.162], 'C': [-1.365], 'Q': [-1.116], 'E': [-1.163], 'G': [-1.364], 'H': [-1.215], 'I': [-1.189], 'L': [-1.315], 'K': [-1.074], 'M': [-1.303], 'F': [-1.135], 'P': [-1.236], 'S': [-1.297], 'T': [-1.252], 'W': [-1.03], 'Y': [-1.03], 'V': [-1.254]} ,
           'oobm770103' :  {'A': [-0.491], 'R': [-0.554], 'N': [-0.382], 'D': [-0.356], 'C': [-0.67], 'Q': [-0.405], 'E': [-0.371], 'G': [-0.534], 'H': [-0.54], 'I': [-0.762], 'L': [-0.65], 'K': [-0.3], 'M': [-0.659], 'F': [-0.729], 'P': [-0.463], 'S': [-0.455], 'T': [-0.515], 'W': [-0.839], 'Y': [-0.656], 'V': [-0.728]} ,
           'oobm770104' :  {'A': [-9.475], 'R': [-16.225], 'N': [-12.48], 'D': [-12.144], 'C': [-12.21], 'Q': [-13.689], 'E': [-13.815], 'G': [-7.592], 'H': [-17.55], 'I': [-15.608], 'L': [-15.728], 'K': [-12.366], 'M': [-15.704], 'F': [-20.504], 'P': [-11.893], 'S': [-10.518], 'T': [-12.369], 'W': [-26.166], 'Y': [-20.232], 'V': [-13.867]} ,
           'oobm770105' :  {'A': [-7.02], 'R': [-10.131], 'N': [-9.424], 'D': [-9.296], 'C': [-8.19], 'Q': [-10.044], 'E': [-10.467], 'G': [-5.456], 'H': [-12.15], 'I': [-9.512], 'L': [-10.52], 'K': [-9.666], 'M': [-10.424], 'F': [-12.485], 'P': [-8.652], 'S': [-7.782], 'T': [-8.764], 'W': [-14.42], 'Y': [-12.36], 'V': [-8.778]} ,
           'oobm850101' :  {'A': [2.01], 'R': [0.84], 'N': [0.03], 'D': [-2.05], 'C': [1.98], 'Q': [1.02], 'E': [0.93], 'G': [0.12], 'H': [-0.14], 'I': [3.7], 'L': [2.73], 'K': [2.55], 'M': [1.75], 'F': [2.68], 'P': [0.41], 'S': [1.47], 'T': [2.39], 'W': [2.49], 'Y': [2.23], 'V': [3.5]} ,
           'oobm850102' :  {'A': [1.34], 'R': [0.95], 'N': [2.49], 'D': [3.32], 'C': [1.07], 'Q': [1.49], 'E': [2.2], 'G': [2.07], 'H': [1.27], 'I': [0.66], 'L': [0.54], 'K': [0.61], 'M': [0.7], 'F': [0.8], 'P': [2.12], 'S': [0.94], 'T': [1.09], 'W': [-4.65], 'Y': [-0.17], 'V': [1.32]} ,
           'oobm850103' :  {'A': [0.46], 'R': [-1.54], 'N': [1.31], 'D': [-0.33], 'C': [0.2], 'Q': [-1.12], 'E': [0.48], 'G': [0.64], 'H': [-1.31], 'I': [3.28], 'L': [0.43], 'K': [-1.71], 'M': [0.15], 'F': [0.52], 'P': [-0.58], 'S': [-0.83], 'T': [-1.52], 'W': [1.25], 'Y': [-2.21], 'V': [0.54]} ,
           'oobm850104' :  {'A': [-2.49], 'R': [2.55], 'N': [2.27], 'D': [8.86], 'C': [-3.13], 'Q': [1.79], 'E': [4.04], 'G': [-0.56], 'H': [4.22], 'I': [-10.87], 'L': [-7.16], 'K': [-9.97], 'M': [-4.96], 'F': [-6.64], 'P': [5.19], 'S': [-1.6], 'T': [-4.75], 'W': [-17.84], 'Y': [9.25], 'V': [-3.97]} ,
           'oobm850105' :  {'A': [4.55], 'R': [5.97], 'N': [5.56], 'D': [2.85], 'C': [-0.78], 'Q': [4.15], 'E': [5.16], 'G': [9.14], 'H': [4.48], 'I': [2.1], 'L': [3.24], 'K': [10.68], 'M': [2.18], 'F': [4.37], 'P': [5.14], 'S': [6.78], 'T': [8.6], 'W': [1.97], 'Y': [2.4], 'V': [3.81]} ,
           'palj810101' :  {'A': [1.3], 'R': [0.93], 'N': [0.9], 'D': [1.02], 'C': [0.92], 'Q': [1.04], 'E': [1.43], 'G': [0.63], 'H': [1.33], 'I': [0.87], 'L': [1.3], 'K': [1.23], 'M': [1.32], 'F': [1.09], 'P': [0.63], 'S': [0.78], 'T': [0.8], 'W': [1.03], 'Y': [0.71], 'V': [0.95]} ,
           'palj810102' :  {'A': [1.32], 'R': [1.04], 'N': [0.74], 'D': [0.97], 'C': [0.7], 'Q': [1.25], 'E': [1.48], 'G': [0.59], 'H': [1.06], 'I': [1.01], 'L': [1.22], 'K': [1.13], 'M': [1.47], 'F': [1.1], 'P': [0.57], 'S': [0.77], 'T': [0.86], 'W': [1.02], 'Y': [0.72], 'V': [1.05]} ,
           'palj810103' :  {'A': [0.81], 'R': [1.03], 'N': [0.81], 'D': [0.71], 'C': [1.12], 'Q': [1.03], 'E': [0.59], 'G': [0.94], 'H': [0.85], 'I': [1.47], 'L': [1.03], 'K': [0.77], 'M': [0.96], 'F': [1.13], 'P': [0.75], 'S': [1.02], 'T': [1.19], 'W': [1.24], 'Y': [1.35], 'V': [1.44]} ,
           'palj810104' :  {'A': [0.9], 'R': [0.75], 'N': [0.82], 'D': [0.75], 'C': [1.12], 'Q': [0.95], 'E': [0.44], 'G': [0.83], 'H': [0.86], 'I': [1.59], 'L': [1.24], 'K': [0.75], 'M': [0.94], 'F': [1.41], 'P': [0.46], 'S': [0.7], 'T': [1.2], 'W': [1.28], 'Y': [1.45], 'V': [1.73]} ,
           'palj810105' :  {'A': [0.84], 'R': [0.91], 'N': [1.48], 'D': [1.28], 'C': [0.69], 'Q': [1.0], 'E': [0.78], 'G': [1.76], 'H': [0.53], 'I': [0.55], 'L': [0.49], 'K': [0.95], 'M': [0.52], 'F': [0.88], 'P': [1.47], 'S': [1.29], 'T': [1.05], 'W': [0.88], 'Y': [1.28], 'V': [0.51]} ,
           'palj810106' :  {'A': [0.65], 'R': [0.93], 'N': [1.45], 'D': [1.47], 'C': [1.43], 'Q': [0.94], 'E': [0.75], 'G': [1.53], 'H': [0.96], 'I': [0.57], 'L': [0.56], 'K': [0.95], 'M': [0.71], 'F': [0.72], 'P': [1.51], 'S': [1.46], 'T': [0.96], 'W': [0.9], 'Y': [1.12], 'V': [0.55]} ,
           'palj810107' :  {'A': [1.08], 'R': [0.93], 'N': [1.05], 'D': [0.86], 'C': [1.22], 'Q': [0.95], 'E': [1.09], 'G': [0.85], 'H': [1.02], 'I': [0.98], 'L': [1.04], 'K': [1.01], 'M': [1.11], 'F': [0.96], 'P': [0.91], 'S': [0.95], 'T': [1.15], 'W': [1.17], 'Y': [0.8], 'V': [1.03]} ,
           'palj810108' :  {'A': [1.34], 'R': [0.91], 'N': [0.83], 'D': [1.06], 'C': [1.27], 'Q': [1.13], 'E': [1.69], 'G': [0.47], 'H': [1.11], 'I': [0.84], 'L': [1.39], 'K': [1.08], 'M': [0.9], 'F': [1.02], 'P': [0.48], 'S': [1.05], 'T': [0.74], 'W': [0.64], 'Y': [0.73], 'V': [1.18]} ,
           'palj810109' :  {'A': [1.15], 'R': [1.06], 'N': [0.87], 'D': [1.0], 'C': [1.03], 'Q': [1.43], 'E': [1.37], 'G': [0.64], 'H': [0.95], 'I': [0.99], 'L': [1.22], 'K': [1.2], 'M': [1.45], 'F': [0.92], 'P': [0.72], 'S': [0.84], 'T': [0.97], 'W': [1.11], 'Y': [0.72], 'V': [0.82]} ,
           'palj810110' :  {'A': [0.89], 'R': [1.06], 'N': [0.67], 'D': [0.71], 'C': [1.04], 'Q': [1.06], 'E': [0.72], 'G': [0.87], 'H': [1.04], 'I': [1.14], 'L': [1.02], 'K': [1.0], 'M': [1.41], 'F': [1.32], 'P': [0.69], 'S': [0.86], 'T': [1.15], 'W': [1.06], 'Y': [1.35], 'V': [1.66]} ,
           'palj810111' :  {'A': [0.82], 'R': [0.99], 'N': [1.27], 'D': [0.98], 'C': [0.71], 'Q': [1.01], 'E': [0.54], 'G': [0.94], 'H': [1.26], 'I': [1.67], 'L': [0.94], 'K': [0.73], 'M': [1.3], 'F': [1.56], 'P': [0.69], 'S': [0.65], 'T': [0.98], 'W': [1.25], 'Y': [1.26], 'V': [1.22]} ,
           'palj810112' :  {'A': [0.98], 'R': [1.03], 'N': [0.66], 'D': [0.74], 'C': [1.01], 'Q': [0.63], 'E': [0.59], 'G': [0.9], 'H': [1.17], 'I': [1.38], 'L': [1.05], 'K': [0.83], 'M': [0.82], 'F': [1.23], 'P': [0.73], 'S': [0.98], 'T': [1.2], 'W': [1.26], 'Y': [1.23], 'V': [1.62]} ,
           'palj810113' :  {'A': [0.69], 'R': [0.0], 'N': [1.52], 'D': [2.42], 'C': [0.0], 'Q': [1.44], 'E': [0.63], 'G': [2.64], 'H': [0.22], 'I': [0.43], 'L': [0.0], 'K': [1.18], 'M': [0.88], 'F': [2.2], 'P': [1.34], 'S': [1.43], 'T': [0.28], 'W': [0.0], 'Y': [1.53], 'V': [0.14]} ,
           'palj810114' :  {'A': [0.87], 'R': [1.3], 'N': [1.36], 'D': [1.24], 'C': [0.83], 'Q': [1.06], 'E': [0.91], 'G': [1.69], 'H': [0.91], 'I': [0.27], 'L': [0.67], 'K': [0.66], 'M': [0.0], 'F': [0.47], 'P': [1.54], 'S': [1.08], 'T': [1.12], 'W': [1.24], 'Y': [0.54], 'V': [0.69]} ,
           'palj810115' :  {'A': [0.91], 'R': [0.77], 'N': [1.32], 'D': [0.9], 'C': [0.5], 'Q': [1.06], 'E': [0.53], 'G': [1.61], 'H': [1.08], 'I': [0.36], 'L': [0.77], 'K': [1.27], 'M': [0.76], 'F': [0.37], 'P': [1.62], 'S': [1.34], 'T': [0.87], 'W': [1.1], 'Y': [1.24], 'V': [0.52]} ,
           'palj810116' :  {'A': [0.92], 'R': [0.9], 'N': [1.57], 'D': [1.22], 'C': [0.62], 'Q': [0.66], 'E': [0.92], 'G': [1.61], 'H': [0.39], 'I': [0.79], 'L': [0.5], 'K': [0.86], 'M': [0.5], 'F': [0.96], 'P': [1.3], 'S': [1.4], 'T': [1.11], 'W': [0.57], 'Y': [1.78], 'V': [0.5]} ,
           'parj860101' :  {'A': [2.1], 'R': [4.2], 'N': [7.0], 'D': [10.0], 'C': [1.4], 'Q': [6.0], 'E': [7.8], 'G': [5.7], 'H': [2.1], 'I': [-8.0], 'L': [-9.2], 'K': [5.7], 'M': [-4.2], 'F': [-9.2], 'P': [2.1], 'S': [6.5], 'T': [5.2], 'W': [-10.0], 'Y': [-1.9], 'V': [-3.7]} ,
           'pliv810101' :  {'A': [-2.89], 'R': [-3.3], 'N': [-3.41], 'D': [-3.38], 'C': [-2.49], 'Q': [-3.15], 'E': [-2.94], 'G': [-3.25], 'H': [-2.84], 'I': [-1.72], 'L': [-1.61], 'K': [-3.31], 'M': [-1.84], 'F': [-1.63], 'P': [-2.5], 'S': [-3.3], 'T': [-2.91], 'W': [-1.75], 'Y': [-2.42], 'V': [-2.08]} ,
           'ponp800101' :  {'A': [12.28], 'R': [11.49], 'N': [11.0], 'D': [10.97], 'C': [14.93], 'Q': [11.28], 'E': [11.19], 'G': [12.01], 'H': [12.84], 'I': [14.77], 'L': [14.1], 'K': [10.8], 'M': [14.33], 'F': [13.43], 'P': [11.19], 'S': [11.26], 'T': [11.65], 'W': [12.95], 'Y': [13.29], 'V': [15.07]} ,
           'ponp800102' :  {'A': [7.62], 'R': [6.81], 'N': [6.17], 'D': [6.18], 'C': [10.93], 'Q': [6.67], 'E': [6.38], 'G': [7.31], 'H': [7.85], 'I': [9.99], 'L': [9.37], 'K': [5.72], 'M': [9.83], 'F': [8.99], 'P': [6.64], 'S': [6.93], 'T': [7.08], 'W': [8.41], 'Y': [8.53], 'V': [10.38]} ,
           'ponp800103' :  {'A': [2.63], 'R': [2.45], 'N': [2.27], 'D': [2.29], 'C': [3.36], 'Q': [2.45], 'E': [2.31], 'G': [2.55], 'H': [2.57], 'I': [3.08], 'L': [2.98], 'K': [2.12], 'M': [3.18], 'F': [3.02], 'P': [2.46], 'S': [2.6], 'T': [2.55], 'W': [2.85], 'Y': [2.79], 'V': [3.21]} ,
           'ponp800104' :  {'A': [13.65], 'R': [11.28], 'N': [12.24], 'D': [10.98], 'C': [14.49], 'Q': [11.3], 'E': [12.55], 'G': [15.36], 'H': [11.59], 'I': [14.63], 'L': [14.01], 'K': [11.96], 'M': [13.4], 'F': [14.08], 'P': [11.51], 'S': [11.26], 'T': [13.0], 'W': [12.06], 'Y': [12.64], 'V': [12.88]} ,
           'ponp800105' :  {'A': [14.6], 'R': [13.24], 'N': [11.79], 'D': [13.78], 'C': [15.9], 'Q': [12.02], 'E': [13.59], 'G': [14.18], 'H': [15.35], 'I': [14.1], 'L': [16.49], 'K': [13.28], 'M': [16.23], 'F': [14.18], 'P': [14.1], 'S': [13.36], 'T': [14.5], 'W': [13.9], 'Y': [14.76], 'V': [16.3]} ,
           'ponp800106' :  {'A': [10.67], 'R': [11.05], 'N': [10.85], 'D': [10.21], 'C': [14.15], 'Q': [11.71], 'E': [11.71], 'G': [10.95], 'H': [12.07], 'I': [12.95], 'L': [13.07], 'K': [9.93], 'M': [15.0], 'F': [13.27], 'P': [10.62], 'S': [11.18], 'T': [10.53], 'W': [11.41], 'Y': [11.52], 'V': [13.86]} ,
           'ponp800107' :  {'A': [3.7], 'R': [2.53], 'N': [2.12], 'D': [2.6], 'C': [3.03], 'Q': [2.7], 'E': [3.3], 'G': [3.13], 'H': [3.57], 'I': [7.69], 'L': [5.88], 'K': [1.79], 'M': [5.21], 'F': [6.6], 'P': [2.12], 'S': [2.43], 'T': [2.6], 'W': [6.25], 'Y': [3.03], 'V': [7.14]} ,
           'ponp800108' :  {'A': [6.05], 'R': [5.7], 'N': [5.04], 'D': [4.95], 'C': [7.86], 'Q': [5.45], 'E': [5.1], 'G': [6.16], 'H': [5.8], 'I': [7.51], 'L': [7.37], 'K': [4.88], 'M': [6.39], 'F': [6.62], 'P': [5.65], 'S': [5.53], 'T': [5.81], 'W': [6.98], 'Y': [6.73], 'V': [7.62]} ,
           'pram820101' :  {'A': [0.305], 'R': [0.227], 'N': [0.322], 'D': [0.335], 'C': [0.339], 'Q': [0.306], 'E': [0.282], 'G': [0.352], 'H': [0.215], 'I': [0.278], 'L': [0.262], 'K': [0.391], 'M': [0.28], 'F': [0.195], 'P': [0.346], 'S': [0.326], 'T': [0.251], 'W': [0.291], 'Y': [0.293], 'V': [0.291]} ,
           'pram820102' :  {'A': [0.175], 'R': [0.083], 'N': [0.09], 'D': [0.14], 'C': [0.074], 'Q': [0.093], 'E': [0.135], 'G': [0.201], 'H': [0.125], 'I': [0.1], 'L': [0.104], 'K': [0.058], 'M': [0.054], 'F': [0.104], 'P': [0.136], 'S': [0.155], 'T': [0.152], 'W': [0.092], 'Y': [0.081], 'V': [0.096]} ,
           'pram820103' :  {'A': [0.687], 'R': [0.59], 'N': [0.489], 'D': [0.632], 'C': [0.263], 'Q': [0.527], 'E': [0.669], 'G': [0.67], 'H': [0.594], 'I': [0.564], 'L': [0.541], 'K': [0.407], 'M': [0.328], 'F': [0.577], 'P': [0.6], 'S': [0.692], 'T': [0.713], 'W': [0.632], 'Y': [0.495], 'V': [0.529]} ,
           'pram900101' :  {'A': [-6.7], 'R': [51.5], 'N': [20.1], 'D': [38.5], 'C': [-8.4], 'Q': [17.2], 'E': [34.3], 'G': [-4.2], 'H': [12.6], 'I': [-13.0], 'L': [-11.7], 'K': [36.8], 'M': [-14.2], 'F': [-15.5], 'P': [0.8], 'S': [-2.5], 'T': [-5.0], 'W': [-7.9], 'Y': [2.9], 'V': [-10.9]} ,
           'pram900102' :  {'A': [1.29], 'R': [0.96], 'N': [0.9], 'D': [1.04], 'C': [1.11], 'Q': [1.27], 'E': [1.44], 'G': [0.56], 'H': [1.22], 'I': [0.97], 'L': [1.3], 'K': [1.23], 'M': [1.47], 'F': [1.07], 'P': [0.52], 'S': [0.82], 'T': [0.82], 'W': [0.99], 'Y': [0.72], 'V': [0.91]} ,
           'pram900103' :  {'A': [0.9], 'R': [0.99], 'N': [0.76], 'D': [0.72], 'C': [0.74], 'Q': [0.8], 'E': [0.75], 'G': [0.92], 'H': [1.08], 'I': [1.45], 'L': [1.02], 'K': [0.77], 'M': [0.97], 'F': [1.32], 'P': [0.64], 'S': [0.95], 'T': [1.21], 'W': [1.14], 'Y': [1.25], 'V': [1.49]} ,
           'pram900104' :  {'A': [0.78], 'R': [0.88], 'N': [1.28], 'D': [1.41], 'C': [0.8], 'Q': [0.97], 'E': [1.0], 'G': [1.64], 'H': [0.69], 'I': [0.51], 'L': [0.59], 'K': [0.96], 'M': [0.39], 'F': [0.58], 'P': [1.91], 'S': [1.33], 'T': [1.03], 'W': [0.75], 'Y': [1.05], 'V': [0.47]} ,
           'ptio830101' :  {'A': [1.1], 'R': [0.95], 'N': [0.8], 'D': [0.65], 'C': [0.95], 'Q': [1.0], 'E': [1.0], 'G': [0.6], 'H': [0.85], 'I': [1.1], 'L': [1.25], 'K': [1.0], 'M': [1.15], 'F': [1.1], 'P': [0.1], 'S': [0.75], 'T': [0.75], 'W': [1.1], 'Y': [1.1], 'V': [0.95]} ,
           'ptio830102' :  {'A': [1.0], 'R': [0.7], 'N': [0.6], 'D': [0.5], 'C': [1.9], 'Q': [1.0], 'E': [0.7], 'G': [0.3], 'H': [0.8], 'I': [4.0], 'L': [2.0], 'K': [0.7], 'M': [1.9], 'F': [3.1], 'P': [0.2], 'S': [0.9], 'T': [1.7], 'W': [2.2], 'Y': [2.8], 'V': [4.0]} ,
           'qian880101' :  {'A': [0.12], 'R': [0.04], 'N': [-0.1], 'D': [0.01], 'C': [-0.25], 'Q': [-0.03], 'E': [-0.02], 'G': [-0.02], 'H': [-0.06], 'I': [-0.07], 'L': [0.05], 'K': [0.26], 'M': [0.0], 'F': [0.05], 'P': [-0.19], 'S': [-0.19], 'T': [-0.04], 'W': [-0.06], 'Y': [-0.14], 'V': [-0.03]} ,
           'qian880102' :  {'A': [0.26], 'R': [-0.14], 'N': [-0.03], 'D': [0.15], 'C': [-0.15], 'Q': [-0.13], 'E': [0.21], 'G': [-0.37], 'H': [0.1], 'I': [-0.03], 'L': [-0.02], 'K': [0.12], 'M': [0.0], 'F': [0.12], 'P': [-0.08], 'S': [0.01], 'T': [-0.34], 'W': [-0.01], 'Y': [-0.29], 'V': [0.02]} ,
           'qian880103' :  {'A': [0.64], 'R': [-0.1], 'N': [0.09], 'D': [0.33], 'C': [0.03], 'Q': [-0.23], 'E': [0.51], 'G': [-0.09], 'H': [-0.23], 'I': [-0.22], 'L': [0.41], 'K': [-0.17], 'M': [0.13], 'F': [-0.03], 'P': [-0.43], 'S': [-0.1], 'T': [-0.07], 'W': [-0.02], 'Y': [-0.38], 'V': [-0.01]} ,
           'qian880104' :  {'A': [0.29], 'R': [-0.03], 'N': [-0.04], 'D': [0.11], 'C': [-0.05], 'Q': [0.26], 'E': [0.28], 'G': [-0.67], 'H': [-0.26], 'I': [0.0], 'L': [0.47], 'K': [-0.19], 'M': [0.27], 'F': [0.24], 'P': [-0.34], 'S': [-0.17], 'T': [-0.2], 'W': [0.25], 'Y': [-0.3], 'V': [-0.01]} ,
           'qian880105' :  {'A': [0.68], 'R': [-0.22], 'N': [-0.09], 'D': [-0.02], 'C': [-0.15], 'Q': [-0.15], 'E': [0.44], 'G': [-0.73], 'H': [-0.14], 'I': [-0.08], 'L': [0.61], 'K': [0.03], 'M': [0.39], 'F': [0.06], 'P': [-0.76], 'S': [-0.26], 'T': [-0.1], 'W': [0.2], 'Y': [-0.04], 'V': [0.12]} ,
           'qian880106' :  {'A': [0.34], 'R': [0.22], 'N': [-0.33], 'D': [0.06], 'C': [-0.18], 'Q': [0.01], 'E': [0.2], 'G': [-0.88], 'H': [-0.09], 'I': [-0.03], 'L': [0.2], 'K': [-0.11], 'M': [0.43], 'F': [0.15], 'P': [-0.81], 'S': [-0.35], 'T': [-0.37], 'W': [0.07], 'Y': [-0.31], 'V': [0.13]} ,
           'qian880107' :  {'A': [0.57], 'R': [0.23], 'N': [-0.36], 'D': [-0.46], 'C': [-0.15], 'Q': [0.15], 'E': [0.26], 'G': [-0.71], 'H': [-0.05], 'I': [0.0], 'L': [0.48], 'K': [0.16], 'M': [0.41], 'F': [0.03], 'P': [-1.12], 'S': [-0.47], 'T': [-0.54], 'W': [-0.1], 'Y': [-0.35], 'V': [0.31]} ,
           'qian880108' :  {'A': [0.33], 'R': [0.1], 'N': [-0.19], 'D': [-0.44], 'C': [-0.03], 'Q': [0.19], 'E': [0.21], 'G': [-0.46], 'H': [0.27], 'I': [-0.33], 'L': [0.57], 'K': [0.23], 'M': [0.79], 'F': [0.48], 'P': [-1.86], 'S': [-0.23], 'T': [-0.33], 'W': [0.15], 'Y': [-0.19], 'V': [0.24]} ,
           'qian880109' :  {'A': [0.13], 'R': [0.08], 'N': [-0.07], 'D': [-0.71], 'C': [-0.09], 'Q': [0.12], 'E': [0.13], 'G': [-0.39], 'H': [0.32], 'I': [0.0], 'L': [0.5], 'K': [0.37], 'M': [0.63], 'F': [0.15], 'P': [-1.4], 'S': [-0.28], 'T': [-0.21], 'W': [0.02], 'Y': [-0.1], 'V': [0.17]} ,
           'qian880110' :  {'A': [0.31], 'R': [0.18], 'N': [-0.1], 'D': [-0.81], 'C': [-0.26], 'Q': [0.41], 'E': [-0.06], 'G': [-0.42], 'H': [0.51], 'I': [-0.15], 'L': [0.56], 'K': [0.47], 'M': [0.58], 'F': [0.1], 'P': [-1.33], 'S': [-0.49], 'T': [-0.44], 'W': [0.14], 'Y': [-0.08], 'V': [-0.01]} ,
           'qian880111' :  {'A': [0.21], 'R': [0.07], 'N': [-0.04], 'D': [-0.58], 'C': [-0.12], 'Q': [0.13], 'E': [-0.23], 'G': [-0.15], 'H': [0.37], 'I': [0.31], 'L': [0.7], 'K': [0.28], 'M': [0.61], 'F': [-0.06], 'P': [-1.03], 'S': [-0.28], 'T': [-0.25], 'W': [0.21], 'Y': [0.16], 'V': [0.0]} ,
           'qian880112' :  {'A': [0.18], 'R': [0.21], 'N': [-0.03], 'D': [-0.32], 'C': [-0.29], 'Q': [-0.27], 'E': [-0.25], 'G': [-0.4], 'H': [0.28], 'I': [-0.03], 'L': [0.62], 'K': [0.41], 'M': [0.21], 'F': [0.05], 'P': [-0.84], 'S': [-0.05], 'T': [-0.16], 'W': [0.32], 'Y': [0.11], 'V': [0.06]} ,
           'qian880113' :  {'A': [-0.08], 'R': [0.05], 'N': [-0.08], 'D': [-0.24], 'C': [-0.25], 'Q': [-0.28], 'E': [-0.19], 'G': [-0.1], 'H': [0.29], 'I': [-0.01], 'L': [0.28], 'K': [0.45], 'M': [0.11], 'F': [0.0], 'P': [-0.42], 'S': [0.07], 'T': [-0.33], 'W': [0.36], 'Y': [0.0], 'V': [-0.13]} ,
           'qian880114' :  {'A': [-0.18], 'R': [-0.13], 'N': [0.28], 'D': [0.05], 'C': [-0.26], 'Q': [0.21], 'E': [-0.06], 'G': [0.23], 'H': [0.24], 'I': [-0.42], 'L': [-0.23], 'K': [0.03], 'M': [-0.42], 'F': [-0.18], 'P': [-0.13], 'S': [0.41], 'T': [0.33], 'W': [-0.1], 'Y': [-0.1], 'V': [-0.07]} ,
           'qian880115' :  {'A': [-0.01], 'R': [0.02], 'N': [0.41], 'D': [-0.09], 'C': [-0.27], 'Q': [0.01], 'E': [0.09], 'G': [0.13], 'H': [0.22], 'I': [-0.27], 'L': [-0.25], 'K': [0.08], 'M': [-0.57], 'F': [-0.12], 'P': [0.26], 'S': [0.44], 'T': [0.35], 'W': [-0.15], 'Y': [0.15], 'V': [-0.09]} ,
           'qian880116' :  {'A': [-0.19], 'R': [0.03], 'N': [0.02], 'D': [-0.06], 'C': [-0.29], 'Q': [0.02], 'E': [-0.1], 'G': [0.19], 'H': [-0.16], 'I': [-0.08], 'L': [-0.42], 'K': [-0.09], 'M': [-0.38], 'F': [-0.32], 'P': [0.05], 'S': [0.25], 'T': [0.22], 'W': [-0.19], 'Y': [0.05], 'V': [-0.15]} ,
           'qian880117' :  {'A': [-0.14], 'R': [0.14], 'N': [-0.27], 'D': [-0.1], 'C': [-0.64], 'Q': [-0.11], 'E': [-0.39], 'G': [0.46], 'H': [-0.04], 'I': [0.16], 'L': [-0.57], 'K': [0.04], 'M': [0.24], 'F': [0.08], 'P': [0.02], 'S': [-0.12], 'T': [0.0], 'W': [-0.1], 'Y': [0.18], 'V': [0.29]} ,
           'qian880118' :  {'A': [-0.31], 'R': [0.25], 'N': [-0.53], 'D': [-0.54], 'C': [-0.06], 'Q': [0.07], 'E': [-0.52], 'G': [0.37], 'H': [-0.32], 'I': [0.57], 'L': [0.09], 'K': [-0.29], 'M': [0.29], 'F': [0.24], 'P': [-0.31], 'S': [0.11], 'T': [0.03], 'W': [0.15], 'Y': [0.29], 'V': [0.48]} ,
           'qian880119' :  {'A': [-0.1], 'R': [0.19], 'N': [-0.89], 'D': [-0.89], 'C': [0.13], 'Q': [-0.04], 'E': [-0.34], 'G': [-0.45], 'H': [-0.34], 'I': [0.95], 'L': [0.32], 'K': [-0.46], 'M': [0.43], 'F': [0.36], 'P': [-0.91], 'S': [-0.12], 'T': [0.49], 'W': [0.34], 'Y': [0.42], 'V': [0.76]} ,
           'qian880120' :  {'A': [-0.25], 'R': [-0.02], 'N': [-0.77], 'D': [-1.01], 'C': [0.13], 'Q': [-0.12], 'E': [-0.62], 'G': [-0.72], 'H': [-0.16], 'I': [1.1], 'L': [0.23], 'K': [-0.59], 'M': [0.32], 'F': [0.48], 'P': [-1.24], 'S': [-0.31], 'T': [0.17], 'W': [0.45], 'Y': [0.77], 'V': [0.69]} ,
           'qian880121' :  {'A': [-0.26], 'R': [-0.09], 'N': [-0.34], 'D': [-0.55], 'C': [0.47], 'Q': [-0.33], 'E': [-0.75], 'G': [-0.56], 'H': [-0.04], 'I': [0.94], 'L': [0.25], 'K': [-0.55], 'M': [-0.05], 'F': [0.2], 'P': [-1.28], 'S': [-0.28], 'T': [0.08], 'W': [0.22], 'Y': [0.53], 'V': [0.67]} ,
           'qian880122' :  {'A': [0.05], 'R': [-0.11], 'N': [-0.4], 'D': [-0.11], 'C': [0.36], 'Q': [-0.67], 'E': [-0.35], 'G': [0.14], 'H': [0.02], 'I': [0.47], 'L': [0.32], 'K': [-0.51], 'M': [-0.1], 'F': [0.2], 'P': [-0.79], 'S': [0.03], 'T': [-0.15], 'W': [0.09], 'Y': [0.34], 'V': [0.58]} ,
           'qian880123' :  {'A': [-0.44], 'R': [-0.13], 'N': [0.05], 'D': [-0.2], 'C': [0.13], 'Q': [-0.58], 'E': [-0.28], 'G': [0.08], 'H': [0.09], 'I': [-0.04], 'L': [-0.12], 'K': [-0.33], 'M': [-0.21], 'F': [-0.13], 'P': [-0.48], 'S': [0.27], 'T': [0.47], 'W': [-0.22], 'Y': [-0.11], 'V': [0.06]} ,
           'qian880124' :  {'A': [-0.31], 'R': [-0.1], 'N': [0.06], 'D': [0.13], 'C': [-0.11], 'Q': [-0.47], 'E': [-0.05], 'G': [0.45], 'H': [-0.06], 'I': [-0.25], 'L': [-0.44], 'K': [-0.44], 'M': [-0.28], 'F': [-0.04], 'P': [-0.29], 'S': [0.34], 'T': [0.27], 'W': [-0.08], 'Y': [0.06], 'V': [0.11]} ,
           'qian880125' :  {'A': [-0.02], 'R': [0.04], 'N': [0.03], 'D': [0.11], 'C': [-0.02], 'Q': [-0.17], 'E': [0.1], 'G': [0.38], 'H': [-0.09], 'I': [-0.48], 'L': [-0.26], 'K': [-0.39], 'M': [-0.14], 'F': [-0.03], 'P': [-0.04], 'S': [0.41], 'T': [0.36], 'W': [-0.01], 'Y': [-0.08], 'V': [-0.18]} ,
           'qian880126' :  {'A': [-0.06], 'R': [0.02], 'N': [0.1], 'D': [0.24], 'C': [-0.19], 'Q': [-0.04], 'E': [-0.04], 'G': [0.17], 'H': [0.19], 'I': [-0.2], 'L': [-0.46], 'K': [-0.43], 'M': [-0.52], 'F': [-0.33], 'P': [0.37], 'S': [0.43], 'T': [0.5], 'W': [-0.32], 'Y': [0.35], 'V': [0.0]} ,
           'qian880127' :  {'A': [-0.05], 'R': [0.06], 'N': [0.0], 'D': [0.15], 'C': [0.3], 'Q': [-0.08], 'E': [-0.02], 'G': [-0.14], 'H': [-0.07], 'I': [0.26], 'L': [0.04], 'K': [-0.42], 'M': [0.25], 'F': [0.09], 'P': [0.31], 'S': [-0.11], 'T': [-0.06], 'W': [0.19], 'Y': [0.33], 'V': [0.04]} ,
           'qian880128' :  {'A': [-0.19], 'R': [0.17], 'N': [-0.38], 'D': [0.09], 'C': [0.41], 'Q': [0.04], 'E': [-0.2], 'G': [0.28], 'H': [-0.19], 'I': [-0.06], 'L': [0.34], 'K': [-0.2], 'M': [0.45], 'F': [0.07], 'P': [0.04], 'S': [-0.23], 'T': [-0.02], 'W': [0.16], 'Y': [0.22], 'V': [0.05]} ,
           'qian880129' :  {'A': [-0.43], 'R': [0.06], 'N': [0.0], 'D': [-0.31], 'C': [0.19], 'Q': [0.14], 'E': [-0.41], 'G': [-0.21], 'H': [0.21], 'I': [0.29], 'L': [-0.1], 'K': [0.33], 'M': [-0.01], 'F': [0.25], 'P': [0.28], 'S': [-0.23], 'T': [-0.26], 'W': [0.15], 'Y': [0.09], 'V': [-0.1]} ,
           'qian880130' :  {'A': [-0.19], 'R': [-0.07], 'N': [0.17], 'D': [-0.27], 'C': [0.42], 'Q': [-0.29], 'E': [-0.22], 'G': [0.17], 'H': [0.17], 'I': [-0.34], 'L': [-0.22], 'K': [0.0], 'M': [-0.53], 'F': [-0.31], 'P': [0.14], 'S': [0.22], 'T': [0.1], 'W': [-0.15], 'Y': [-0.02], 'V': [-0.33]} ,
           'qian880131' :  {'A': [-0.25], 'R': [0.12], 'N': [0.61], 'D': [0.6], 'C': [0.18], 'Q': [0.09], 'E': [-0.12], 'G': [0.09], 'H': [0.42], 'I': [-0.54], 'L': [-0.55], 'K': [0.14], 'M': [-0.47], 'F': [-0.29], 'P': [0.89], 'S': [0.24], 'T': [0.16], 'W': [-0.44], 'Y': [-0.19], 'V': [-0.45]} ,
           'qian880132' :  {'A': [-0.27], 'R': [-0.4], 'N': [0.71], 'D': [0.54], 'C': [0.0], 'Q': [-0.08], 'E': [-0.12], 'G': [1.14], 'H': [0.18], 'I': [-0.74], 'L': [-0.54], 'K': [0.45], 'M': [-0.76], 'F': [-0.47], 'P': [1.4], 'S': [0.4], 'T': [-0.1], 'W': [-0.46], 'Y': [-0.05], 'V': [-0.86]} ,
           'qian880133' :  {'A': [-0.42], 'R': [-0.23], 'N': [0.81], 'D': [0.95], 'C': [-0.18], 'Q': [-0.01], 'E': [-0.09], 'G': [1.24], 'H': [0.05], 'I': [-1.17], 'L': [-0.69], 'K': [0.09], 'M': [-0.86], 'F': [-0.39], 'P': [1.77], 'S': [0.63], 'T': [0.29], 'W': [-0.37], 'Y': [-0.41], 'V': [-1.32]} ,
           'qian880134' :  {'A': [-0.24], 'R': [-0.04], 'N': [0.45], 'D': [0.65], 'C': [-0.38], 'Q': [0.01], 'E': [0.07], 'G': [0.85], 'H': [-0.21], 'I': [-0.65], 'L': [-0.8], 'K': [0.17], 'M': [-0.71], 'F': [-0.61], 'P': [2.27], 'S': [0.33], 'T': [0.13], 'W': [-0.44], 'Y': [-0.49], 'V': [-0.99]} ,
           'qian880135' :  {'A': [-0.14], 'R': [0.21], 'N': [0.35], 'D': [0.66], 'C': [-0.09], 'Q': [0.11], 'E': [0.06], 'G': [0.36], 'H': [-0.31], 'I': [-0.51], 'L': [-0.8], 'K': [-0.14], 'M': [-0.56], 'F': [-0.25], 'P': [1.59], 'S': [0.32], 'T': [0.21], 'W': [-0.17], 'Y': [-0.35], 'V': [-0.7]} ,
           'qian880136' :  {'A': [0.01], 'R': [-0.13], 'N': [-0.11], 'D': [0.78], 'C': [-0.31], 'Q': [-0.13], 'E': [0.09], 'G': [0.14], 'H': [-0.56], 'I': [-0.09], 'L': [-0.81], 'K': [-0.43], 'M': [-0.49], 'F': [-0.2], 'P': [1.14], 'S': [0.13], 'T': [-0.02], 'W': [-0.2], 'Y': [0.1], 'V': [-0.11]} ,
           'qian880137' :  {'A': [-0.3], 'R': [-0.09], 'N': [-0.12], 'D': [0.44], 'C': [0.03], 'Q': [0.24], 'E': [0.18], 'G': [-0.12], 'H': [-0.2], 'I': [-0.07], 'L': [-0.18], 'K': [0.06], 'M': [-0.44], 'F': [0.11], 'P': [0.77], 'S': [-0.09], 'T': [-0.27], 'W': [-0.09], 'Y': [-0.25], 'V': [-0.06]} ,
           'qian880138' :  {'A': [-0.23], 'R': [-0.2], 'N': [0.06], 'D': [0.34], 'C': [0.19], 'Q': [0.47], 'E': [0.28], 'G': [0.14], 'H': [-0.22], 'I': [0.42], 'L': [-0.36], 'K': [-0.15], 'M': [-0.19], 'F': [-0.02], 'P': [0.78], 'S': [-0.29], 'T': [-0.3], 'W': [-0.18], 'Y': [0.07], 'V': [0.29]} ,
           'qian880139' :  {'A': [0.08], 'R': [-0.01], 'N': [-0.06], 'D': [0.04], 'C': [0.37], 'Q': [0.48], 'E': [0.36], 'G': [-0.02], 'H': [-0.45], 'I': [0.09], 'L': [0.24], 'K': [-0.27], 'M': [0.16], 'F': [0.34], 'P': [0.16], 'S': [-0.35], 'T': [-0.04], 'W': [-0.06], 'Y': [-0.2], 'V': [0.18]} ,
           'racs770101' :  {'A': [0.934], 'R': [0.962], 'N': [0.986], 'D': [0.994], 'C': [0.9], 'Q': [1.047], 'E': [0.986], 'G': [1.015], 'H': [0.882], 'I': [0.766], 'L': [0.825], 'K': [1.04], 'M': [0.804], 'F': [0.773], 'P': [1.047], 'S': [1.056], 'T': [1.008], 'W': [0.848], 'Y': [0.931], 'V': [0.825]} ,
           'racs770102' :  {'A': [0.941], 'R': [1.112], 'N': [1.038], 'D': [1.071], 'C': [0.866], 'Q': [1.15], 'E': [1.1], 'G': [1.055], 'H': [0.911], 'I': [0.742], 'L': [0.798], 'K': [1.232], 'M': [0.781], 'F': [0.723], 'P': [1.093], 'S': [1.082], 'T': [1.043], 'W': [0.867], 'Y': [1.05], 'V': [0.817]} ,
           'racs770103' :  {'A': [1.16], 'R': [1.72], 'N': [1.97], 'D': [2.66], 'C': [0.5], 'Q': [3.87], 'E': [2.4], 'G': [1.63], 'H': [0.86], 'I': [0.57], 'L': [0.51], 'K': [3.9], 'M': [0.4], 'F': [0.43], 'P': [2.04], 'S': [1.61], 'T': [1.48], 'W': [0.75], 'Y': [1.72], 'V': [0.59]} ,
           'racs820101' :  {'A': [0.85], 'R': [2.02], 'N': [0.88], 'D': [1.5], 'C': [0.9], 'Q': [1.71], 'E': [1.79], 'G': [1.54], 'H': [1.59], 'I': [0.67], 'L': [1.03], 'K': [0.88], 'M': [1.17], 'F': [0.85], 'P': [1.47], 'S': [1.5], 'T': [1.96], 'W': [0.83], 'Y': [1.34], 'V': [0.89]} ,
           'racs820102' :  {'A': [1.58], 'R': [1.14], 'N': [0.77], 'D': [0.98], 'C': [1.04], 'Q': [1.24], 'E': [1.49], 'G': [0.66], 'H': [0.99], 'I': [1.09], 'L': [1.21], 'K': [1.27], 'M': [1.41], 'F': [1.0], 'P': [1.46], 'S': [1.05], 'T': [0.87], 'W': [1.23], 'Y': [0.68], 'V': [0.88]} ,
           'racs820103' :  {'A': [0.82], 'R': [2.6], 'N': [2.07], 'D': [2.64], 'C': [0.0], 'Q': [0.0], 'E': [2.62], 'G': [1.63], 'H': [0.0], 'I': [2.32], 'L': [0.0], 'K': [2.86], 'M': [0.0], 'F': [0.0], 'P': [0.0], 'S': [1.23], 'T': [2.48], 'W': [0.0], 'Y': [1.9], 'V': [1.62]} ,
           'racs820104' :  {'A': [0.78], 'R': [1.75], 'N': [1.32], 'D': [1.25], 'C': [3.14], 'Q': [0.93], 'E': [0.94], 'G': [1.13], 'H': [1.03], 'I': [1.26], 'L': [0.91], 'K': [0.85], 'M': [0.41], 'F': [1.07], 'P': [1.73], 'S': [1.31], 'T': [1.57], 'W': [0.98], 'Y': [1.31], 'V': [1.11]} ,
           'racs820105' :  {'A': [0.88], 'R': [0.99], 'N': [1.02], 'D': [1.16], 'C': [1.14], 'Q': [0.93], 'E': [1.01], 'G': [0.7], 'H': [1.87], 'I': [1.61], 'L': [1.09], 'K': [0.83], 'M': [1.71], 'F': [1.52], 'P': [0.87], 'S': [1.14], 'T': [0.96], 'W': [1.96], 'Y': [1.68], 'V': [1.56]} ,
           'racs820106' :  {'A': [0.3], 'R': [0.9], 'N': [2.73], 'D': [1.26], 'C': [0.72], 'Q': [0.97], 'E': [1.33], 'G': [3.09], 'H': [1.33], 'I': [0.45], 'L': [0.96], 'K': [0.71], 'M': [1.89], 'F': [1.2], 'P': [0.83], 'S': [1.16], 'T': [0.97], 'W': [1.58], 'Y': [0.86], 'V': [0.64]} ,
           'racs820107' :  {'A': [0.4], 'R': [1.2], 'N': [1.24], 'D': [1.59], 'C': [2.98], 'Q': [0.5], 'E': [1.26], 'G': [1.89], 'H': [2.71], 'I': [1.31], 'L': [0.57], 'K': [0.87], 'M': [0.0], 'F': [1.27], 'P': [0.38], 'S': [0.92], 'T': [1.38], 'W': [1.53], 'Y': [1.79], 'V': [0.95]} ,
           'racs820108' :  {'A': [1.48], 'R': [1.02], 'N': [0.99], 'D': [1.19], 'C': [0.86], 'Q': [1.42], 'E': [1.43], 'G': [0.46], 'H': [1.27], 'I': [1.12], 'L': [1.33], 'K': [1.36], 'M': [1.41], 'F': [1.3], 'P': [0.25], 'S': [0.89], 'T': [0.81], 'W': [1.27], 'Y': [0.91], 'V': [0.93]} ,
           'racs820109' :  {'A': [0.0], 'R': [0.0], 'N': [4.14], 'D': [2.15], 'C': [0.0], 'Q': [0.0], 'E': [0.0], 'G': [6.49], 'H': [0.0], 'I': [0.0], 'L': [0.0], 'K': [0.0], 'M': [0.0], 'F': [2.11], 'P': [1.99], 'S': [0.0], 'T': [1.24], 'W': [0.0], 'Y': [1.9], 'V': [0.0]} ,
           'racs820110' :  {'A': [1.02], 'R': [1.0], 'N': [1.31], 'D': [1.76], 'C': [1.05], 'Q': [1.05], 'E': [0.83], 'G': [2.39], 'H': [0.4], 'I': [0.83], 'L': [1.06], 'K': [0.94], 'M': [1.33], 'F': [0.41], 'P': [2.73], 'S': [1.18], 'T': [0.77], 'W': [1.22], 'Y': [1.09], 'V': [0.88]} ,
           'racs820111' :  {'A': [0.93], 'R': [1.52], 'N': [0.92], 'D': [0.6], 'C': [1.08], 'Q': [0.94], 'E': [0.73], 'G': [0.78], 'H': [1.08], 'I': [1.74], 'L': [1.03], 'K': [1.0], 'M': [1.31], 'F': [1.51], 'P': [1.37], 'S': [0.97], 'T': [1.38], 'W': [1.12], 'Y': [1.65], 'V': [1.7]} ,
           'racs820112' :  {'A': [0.99], 'R': [1.19], 'N': [1.15], 'D': [1.18], 'C': [2.32], 'Q': [1.52], 'E': [1.36], 'G': [1.4], 'H': [1.06], 'I': [0.81], 'L': [1.26], 'K': [0.91], 'M': [1.0], 'F': [1.25], 'P': [0.0], 'S': [1.5], 'T': [1.18], 'W': [1.33], 'Y': [1.09], 'V': [1.01]} ,
           'racs820113' :  {'A': [17.05], 'R': [21.25], 'N': [34.81], 'D': [19.27], 'C': [28.84], 'Q': [15.42], 'E': [20.12], 'G': [38.14], 'H': [23.07], 'I': [16.66], 'L': [10.89], 'K': [16.46], 'M': [20.61], 'F': [16.26], 'P': [23.94], 'S': [19.95], 'T': [18.92], 'W': [23.36], 'Y': [26.49], 'V': [17.06]} ,
           'racs820114' :  {'A': [14.53], 'R': [17.82], 'N': [13.59], 'D': [19.78], 'C': [30.57], 'Q': [22.18], 'E': [18.19], 'G': [37.16], 'H': [22.63], 'I': [20.28], 'L': [14.3], 'K': [14.07], 'M': [20.61], 'F': [19.61], 'P': [52.63], 'S': [18.56], 'T': [21.09], 'W': [19.78], 'Y': [26.36], 'V': [21.87]} ,
           'rada880101' :  {'A': [1.81], 'R': [-14.92], 'N': [-6.64], 'D': [-8.72], 'C': [1.28], 'Q': [-5.54], 'E': [-6.81], 'G': [0.94], 'H': [-4.66], 'I': [4.92], 'L': [4.92], 'K': [-5.55], 'M': [2.35], 'F': [2.98], 'P': [0.0], 'S': [-3.4], 'T': [-2.57], 'W': [2.33], 'Y': [-0.14], 'V': [4.04]} ,
           'rada880102' :  {'A': [0.52], 'R': [-1.32], 'N': [-0.01], 'D': [0.0], 'C': [0.0], 'Q': [-0.07], 'E': [-0.79], 'G': [0.0], 'H': [0.95], 'I': [2.04], 'L': [1.76], 'K': [0.08], 'M': [1.32], 'F': [2.09], 'P': [0.0], 'S': [0.04], 'T': [0.27], 'W': [2.51], 'Y': [1.63], 'V': [1.18]} ,
           'rada880103' :  {'A': [0.13], 'R': [-5.0], 'N': [-3.04], 'D': [-2.23], 'C': [-2.52], 'Q': [-3.84], 'E': [-3.43], 'G': [1.45], 'H': [-5.61], 'I': [-2.77], 'L': [-2.64], 'K': [-3.97], 'M': [-3.83], 'F': [-3.74], 'P': [0.0], 'S': [-1.66], 'T': [-2.31], 'W': [-8.21], 'Y': [-5.97], 'V': [-2.05]} ,
           'rada880104' :  {'A': [1.29], 'R': [-13.6], 'N': [-6.63], 'D': [0.0], 'C': [0.0], 'Q': [-5.47], 'E': [-6.02], 'G': [0.94], 'H': [-5.61], 'I': [2.88], 'L': [3.16], 'K': [-5.63], 'M': [1.03], 'F': [0.89], 'P': [0.0], 'S': [-3.44], 'T': [-2.84], 'W': [-0.18], 'Y': [-1.77], 'V': [2.86]} ,
           'rada880105' :  {'A': [1.42], 'R': [-18.6], 'N': [-9.67], 'D': [0.0], 'C': [0.0], 'Q': [-9.31], 'E': [-9.45], 'G': [2.39], 'H': [-11.22], 'I': [0.11], 'L': [0.52], 'K': [-9.6], 'M': [-2.8], 'F': [-2.85], 'P': [0.0], 'S': [-5.1], 'T': [-5.15], 'W': [-8.39], 'Y': [-7.74], 'V': [0.81]} ,
           'rada880106' :  {'A': [93.7], 'R': [250.4], 'N': [146.3], 'D': [142.6], 'C': [135.2], 'Q': [177.7], 'E': [182.9], 'G': [52.6], 'H': [188.1], 'I': [182.2], 'L': [173.7], 'K': [215.2], 'M': [197.6], 'F': [228.6], 'P': [0.0], 'S': [109.5], 'T': [142.1], 'W': [271.6], 'Y': [239.9], 'V': [157.2]} ,
           'rada880107' :  {'A': [-0.29], 'R': [-2.71], 'N': [-1.18], 'D': [-1.02], 'C': [0.0], 'Q': [-1.53], 'E': [-0.9], 'G': [-0.34], 'H': [-0.94], 'I': [0.24], 'L': [-0.12], 'K': [-2.05], 'M': [-0.24], 'F': [0.0], 'P': [0.0], 'S': [-0.75], 'T': [-0.71], 'W': [-0.59], 'Y': [-1.02], 'V': [0.09]} ,
           'rada880108' :  {'A': [-0.06], 'R': [-0.84], 'N': [-0.48], 'D': [-0.8], 'C': [1.36], 'Q': [-0.73], 'E': [-0.77], 'G': [-0.41], 'H': [0.49], 'I': [1.31], 'L': [1.21], 'K': [-1.18], 'M': [1.27], 'F': [1.27], 'P': [0.0], 'S': [-0.5], 'T': [-0.27], 'W': [0.88], 'Y': [0.33], 'V': [1.09]} ,
           'ricj880101' :  {'A': [0.7], 'R': [0.4], 'N': [1.2], 'D': [1.4], 'C': [0.6], 'Q': [1.0], 'E': [1.0], 'G': [1.6], 'H': [1.2], 'I': [0.9], 'L': [0.9], 'K': [1.0], 'M': [0.3], 'F': [1.2], 'P': [0.7], 'S': [1.6], 'T': [0.3], 'W': [1.1], 'Y': [1.9], 'V': [0.7]} ,
           'ricj880102' :  {'A': [0.7], 'R': [0.4], 'N': [1.2], 'D': [1.4], 'C': [0.6], 'Q': [1.0], 'E': [1.0], 'G': [1.6], 'H': [1.2], 'I': [0.9], 'L': [0.9], 'K': [1.0], 'M': [0.3], 'F': [1.2], 'P': [0.7], 'S': [1.6], 'T': [0.3], 'W': [1.1], 'Y': [1.9], 'V': [0.7]} ,
           'ricj880103' :  {'A': [0.5], 'R': [0.4], 'N': [3.5], 'D': [2.1], 'C': [0.6], 'Q': [0.4], 'E': [0.4], 'G': [1.8], 'H': [1.1], 'I': [0.2], 'L': [0.2], 'K': [0.7], 'M': [0.8], 'F': [0.2], 'P': [0.8], 'S': [2.3], 'T': [1.6], 'W': [0.3], 'Y': [0.8], 'V': [0.1]} ,
           'ricj880104' :  {'A': [1.2], 'R': [0.7], 'N': [0.7], 'D': [0.8], 'C': [0.8], 'Q': [0.7], 'E': [2.2], 'G': [0.3], 'H': [0.7], 'I': [0.9], 'L': [0.9], 'K': [0.6], 'M': [0.3], 'F': [0.5], 'P': [2.6], 'S': [0.7], 'T': [0.8], 'W': [2.1], 'Y': [1.8], 'V': [1.1]} ,
           'ricj880105' :  {'A': [1.6], 'R': [0.9], 'N': [0.7], 'D': [2.6], 'C': [1.2], 'Q': [0.8], 'E': [2.0], 'G': [0.9], 'H': [0.7], 'I': [0.7], 'L': [0.3], 'K': [1.0], 'M': [1.0], 'F': [0.9], 'P': [0.5], 'S': [0.8], 'T': [0.7], 'W': [1.7], 'Y': [0.4], 'V': [0.6]} ,
           'ricj880106' :  {'A': [1.0], 'R': [0.4], 'N': [0.7], 'D': [2.2], 'C': [0.6], 'Q': [1.5], 'E': [3.3], 'G': [0.6], 'H': [0.7], 'I': [0.4], 'L': [0.6], 'K': [0.8], 'M': [1.0], 'F': [0.6], 'P': [0.4], 'S': [0.4], 'T': [1.0], 'W': [1.4], 'Y': [1.2], 'V': [1.1]} ,
           'ricj880107' :  {'A': [1.1], 'R': [1.5], 'N': [0.0], 'D': [0.3], 'C': [1.1], 'Q': [1.3], 'E': [0.5], 'G': [0.4], 'H': [1.5], 'I': [1.1], 'L': [2.6], 'K': [0.8], 'M': [1.7], 'F': [1.9], 'P': [0.1], 'S': [0.4], 'T': [0.5], 'W': [3.1], 'Y': [0.6], 'V': [1.5]} ,
           'ricj880108' :  {'A': [1.4], 'R': [1.2], 'N': [1.2], 'D': [0.6], 'C': [1.6], 'Q': [1.4], 'E': [0.9], 'G': [0.6], 'H': [0.9], 'I': [0.9], 'L': [1.1], 'K': [1.9], 'M': [1.7], 'F': [1.0], 'P': [0.3], 'S': [1.1], 'T': [0.6], 'W': [1.4], 'Y': [0.2], 'V': [0.8]} ,
           'ricj880109' :  {'A': [1.8], 'R': [1.3], 'N': [0.9], 'D': [1.0], 'C': [0.7], 'Q': [1.3], 'E': [0.8], 'G': [0.5], 'H': [1.0], 'I': [1.2], 'L': [1.2], 'K': [1.1], 'M': [1.5], 'F': [1.3], 'P': [0.3], 'S': [0.6], 'T': [1.0], 'W': [1.5], 'Y': [0.8], 'V': [1.2]} ,
           'ricj880110' :  {'A': [1.8], 'R': [1.0], 'N': [0.6], 'D': [0.7], 'C': [0.0], 'Q': [1.0], 'E': [1.1], 'G': [0.5], 'H': [2.4], 'I': [1.3], 'L': [1.2], 'K': [1.4], 'M': [2.7], 'F': [1.9], 'P': [0.3], 'S': [0.5], 'T': [0.5], 'W': [1.1], 'Y': [1.3], 'V': [0.4]} ,
           'ricj880111' :  {'A': [1.3], 'R': [0.8], 'N': [0.6], 'D': [0.5], 'C': [0.7], 'Q': [0.2], 'E': [0.7], 'G': [0.5], 'H': [1.9], 'I': [1.6], 'L': [1.4], 'K': [1.0], 'M': [2.8], 'F': [2.9], 'P': [0.0], 'S': [0.5], 'T': [0.6], 'W': [2.1], 'Y': [0.8], 'V': [1.4]} ,
           'ricj880112' :  {'A': [0.7], 'R': [0.8], 'N': [0.8], 'D': [0.6], 'C': [0.2], 'Q': [1.3], 'E': [1.6], 'G': [0.1], 'H': [1.1], 'I': [1.4], 'L': [1.9], 'K': [2.2], 'M': [1.0], 'F': [1.8], 'P': [0.0], 'S': [0.6], 'T': [0.7], 'W': [0.4], 'Y': [1.1], 'V': [1.3]} ,
           'ricj880113' :  {'A': [1.4], 'R': [2.1], 'N': [0.9], 'D': [0.7], 'C': [1.2], 'Q': [1.6], 'E': [1.7], 'G': [0.2], 'H': [1.8], 'I': [0.4], 'L': [0.8], 'K': [1.9], 'M': [1.3], 'F': [0.3], 'P': [0.2], 'S': [1.6], 'T': [0.9], 'W': [0.4], 'Y': [0.3], 'V': [0.7]} ,
           'ricj880114' :  {'A': [1.1], 'R': [1.0], 'N': [1.2], 'D': [0.4], 'C': [1.6], 'Q': [2.1], 'E': [0.8], 'G': [0.2], 'H': [3.4], 'I': [0.7], 'L': [0.7], 'K': [2.0], 'M': [1.0], 'F': [0.7], 'P': [0.0], 'S': [1.7], 'T': [1.0], 'W': [0.0], 'Y': [1.2], 'V': [0.7]} ,
           'ricj880115' :  {'A': [0.8], 'R': [0.9], 'N': [1.6], 'D': [0.7], 'C': [0.4], 'Q': [0.9], 'E': [0.3], 'G': [3.9], 'H': [1.3], 'I': [0.7], 'L': [0.7], 'K': [1.3], 'M': [0.8], 'F': [0.5], 'P': [0.7], 'S': [0.8], 'T': [0.3], 'W': [0.0], 'Y': [0.8], 'V': [0.2]} ,
           'ricj880116' :  {'A': [1.0], 'R': [1.4], 'N': [0.9], 'D': [1.4], 'C': [0.8], 'Q': [1.4], 'E': [0.8], 'G': [1.2], 'H': [1.2], 'I': [1.1], 'L': [0.9], 'K': [1.2], 'M': [0.8], 'F': [0.1], 'P': [1.9], 'S': [0.7], 'T': [0.8], 'W': [0.4], 'Y': [0.9], 'V': [0.6]} ,
           'ricj880117' :  {'A': [0.7], 'R': [1.1], 'N': [1.5], 'D': [1.4], 'C': [0.4], 'Q': [1.1], 'E': [0.7], 'G': [0.6], 'H': [1.0], 'I': [0.7], 'L': [0.5], 'K': [1.3], 'M': [0.0], 'F': [1.2], 'P': [1.5], 'S': [0.9], 'T': [2.1], 'W': [2.7], 'Y': [0.5], 'V': [1.0]} ,
           'robb760101' :  {'A': [6.5], 'R': [-0.9], 'N': [-5.1], 'D': [0.5], 'C': [-1.3], 'Q': [1.0], 'E': [7.8], 'G': [-8.6], 'H': [1.2], 'I': [0.6], 'L': [3.2], 'K': [2.3], 'M': [5.3], 'F': [1.6], 'P': [-7.7], 'S': [-3.9], 'T': [-2.6], 'W': [1.2], 'Y': [-4.5], 'V': [1.4]} ,
           'robb760102' :  {'A': [2.3], 'R': [-5.2], 'N': [0.3], 'D': [7.4], 'C': [0.8], 'Q': [-0.7], 'E': [10.3], 'G': [-5.2], 'H': [-2.8], 'I': [-4.0], 'L': [-2.1], 'K': [-4.1], 'M': [-3.5], 'F': [-1.1], 'P': [8.1], 'S': [-3.5], 'T': [2.3], 'W': [-0.9], 'Y': [-3.7], 'V': [-4.4]} ,
           'robb760103' :  {'A': [6.7], 'R': [0.3], 'N': [-6.1], 'D': [-3.1], 'C': [-4.9], 'Q': [0.6], 'E': [2.2], 'G': [-6.8], 'H': [-1.0], 'I': [3.2], 'L': [5.5], 'K': [0.5], 'M': [7.2], 'F': [2.8], 'P': [-22.8], 'S': [-3.0], 'T': [-4.0], 'W': [4.0], 'Y': [-4.6], 'V': [2.5]} ,
           'robb760104' :  {'A': [2.3], 'R': [1.4], 'N': [-3.3], 'D': [-4.4], 'C': [6.1], 'Q': [2.7], 'E': [2.5], 'G': [-8.3], 'H': [5.9], 'I': [-0.5], 'L': [0.1], 'K': [7.3], 'M': [3.5], 'F': [1.6], 'P': [-24.4], 'S': [-1.9], 'T': [-3.7], 'W': [-0.9], 'Y': [-0.6], 'V': [2.3]} ,
           'robb760105' :  {'A': [-2.3], 'R': [0.4], 'N': [-4.1], 'D': [-4.4], 'C': [4.4], 'Q': [1.2], 'E': [-5.0], 'G': [-4.2], 'H': [-2.5], 'I': [6.7], 'L': [2.3], 'K': [-3.3], 'M': [2.3], 'F': [2.6], 'P': [-1.8], 'S': [-1.7], 'T': [1.3], 'W': [-1.0], 'Y': [4.0], 'V': [6.8]} ,
           'robb760106' :  {'A': [-2.7], 'R': [0.4], 'N': [-4.2], 'D': [-4.4], 'C': [3.7], 'Q': [0.8], 'E': [-8.1], 'G': [-3.9], 'H': [-3.0], 'I': [7.7], 'L': [3.7], 'K': [-2.9], 'M': [3.7], 'F': [3.0], 'P': [-6.6], 'S': [-2.4], 'T': [1.7], 'W': [0.3], 'Y': [3.3], 'V': [7.1]} ,
           'robb760107' :  {'A': [0.0], 'R': [1.1], 'N': [-2.0], 'D': [-2.6], 'C': [5.4], 'Q': [2.4], 'E': [3.1], 'G': [-3.4], 'H': [0.8], 'I': [-0.1], 'L': [-3.7], 'K': [-3.1], 'M': [-2.1], 'F': [0.7], 'P': [7.4], 'S': [1.3], 'T': [0.0], 'W': [-3.4], 'Y': [4.8], 'V': [2.7]} ,
           'robb760108' :  {'A': [-5.0], 'R': [2.1], 'N': [4.2], 'D': [3.1], 'C': [4.4], 'Q': [0.4], 'E': [-4.7], 'G': [5.7], 'H': [-0.3], 'I': [-4.6], 'L': [-5.6], 'K': [1.0], 'M': [-4.8], 'F': [-1.8], 'P': [2.6], 'S': [2.6], 'T': [0.3], 'W': [3.4], 'Y': [2.9], 'V': [-6.0]} ,
           'robb760109' :  {'A': [-3.3], 'R': [0.0], 'N': [5.4], 'D': [3.9], 'C': [-0.3], 'Q': [-0.4], 'E': [-1.8], 'G': [-1.2], 'H': [3.0], 'I': [-0.5], 'L': [-2.3], 'K': [-1.2], 'M': [-4.3], 'F': [0.8], 'P': [6.5], 'S': [1.8], 'T': [-0.7], 'W': [-0.8], 'Y': [3.1], 'V': [-3.5]} ,
           'robb760110' :  {'A': [-4.7], 'R': [2.0], 'N': [3.9], 'D': [1.9], 'C': [6.2], 'Q': [-2.0], 'E': [-4.2], 'G': [5.7], 'H': [-2.6], 'I': [-7.0], 'L': [-6.2], 'K': [2.8], 'M': [-4.8], 'F': [-3.7], 'P': [3.6], 'S': [2.1], 'T': [0.6], 'W': [3.3], 'Y': [3.8], 'V': [-6.2]} ,
           'robb760111' :  {'A': [-3.7], 'R': [1.0], 'N': [-0.6], 'D': [-0.6], 'C': [4.0], 'Q': [3.4], 'E': [-4.3], 'G': [5.9], 'H': [-0.8], 'I': [-0.5], 'L': [-2.8], 'K': [1.3], 'M': [-1.6], 'F': [1.6], 'P': [-6.0], 'S': [1.5], 'T': [1.2], 'W': [6.5], 'Y': [1.3], 'V': [-4.6]} ,
           'robb760112' :  {'A': [-2.5], 'R': [-1.2], 'N': [4.6], 'D': [0.0], 'C': [-4.7], 'Q': [-0.5], 'E': [-4.4], 'G': [4.9], 'H': [1.6], 'I': [-3.3], 'L': [-2.0], 'K': [-0.8], 'M': [-4.1], 'F': [-4.1], 'P': [5.8], 'S': [2.5], 'T': [1.7], 'W': [1.2], 'Y': [-0.6], 'V': [-3.5]} ,
           'robb760113' :  {'A': [-5.1], 'R': [2.6], 'N': [4.7], 'D': [3.1], 'C': [3.8], 'Q': [0.2], 'E': [-5.2], 'G': [5.6], 'H': [-0.9], 'I': [-4.5], 'L': [-5.4], 'K': [1.0], 'M': [-5.3], 'F': [-2.4], 'P': [3.5], 'S': [3.2], 'T': [0.0], 'W': [2.9], 'Y': [3.2], 'V': [-6.3]} ,
           'robb790101' :  {'A': [-1.0], 'R': [0.3], 'N': [-0.7], 'D': [-1.2], 'C': [2.1], 'Q': [-0.1], 'E': [-0.7], 'G': [0.3], 'H': [1.1], 'I': [4.0], 'L': [2.0], 'K': [-0.9], 'M': [1.8], 'F': [2.8], 'P': [0.4], 'S': [-1.2], 'T': [-0.5], 'W': [3.0], 'Y': [2.1], 'V': [1.4]} ,
           'rosg850101' :  {'A': [86.6], 'R': [162.2], 'N': [103.3], 'D': [97.8], 'C': [132.3], 'Q': [119.2], 'E': [113.9], 'G': [62.9], 'H': [155.8], 'I': [158.0], 'L': [164.1], 'K': [115.5], 'M': [172.9], 'F': [194.1], 'P': [92.9], 'S': [85.6], 'T': [106.5], 'W': [224.6], 'Y': [177.7], 'V': [141.0]} ,
           'rosg850102' :  {'A': [0.74], 'R': [0.64], 'N': [0.63], 'D': [0.62], 'C': [0.91], 'Q': [0.62], 'E': [0.62], 'G': [0.72], 'H': [0.78], 'I': [0.88], 'L': [0.85], 'K': [0.52], 'M': [0.85], 'F': [0.88], 'P': [0.64], 'S': [0.66], 'T': [0.7], 'W': [0.85], 'Y': [0.76], 'V': [0.86]} ,
           'rosm880101' :  {'A': [-0.67], 'R': [12.1], 'N': [7.23], 'D': [8.72], 'C': [-0.34], 'Q': [6.39], 'E': [7.35], 'G': [0.0], 'H': [3.82], 'I': [-3.02], 'L': [-3.02], 'K': [6.13], 'M': [-1.3], 'F': [-3.24], 'P': [-1.75], 'S': [4.35], 'T': [3.86], 'W': [-2.86], 'Y': [0.98], 'V': [-2.18]} ,
           'rosm880102' :  {'A': [-0.67], 'R': [3.89], 'N': [2.27], 'D': [1.57], 'C': [-2.0], 'Q': [2.12], 'E': [1.78], 'G': [0.0], 'H': [1.09], 'I': [-3.02], 'L': [-3.02], 'K': [2.46], 'M': [-1.67], 'F': [-3.24], 'P': [-1.75], 'S': [0.1], 'T': [-0.42], 'W': [-2.86], 'Y': [0.98], 'V': [-2.18]} ,
           'rosm880103' :  {'A': [0.4], 'R': [0.3], 'N': [0.9], 'D': [0.8], 'C': [0.5], 'Q': [0.7], 'E': [1.3], 'G': [0.0], 'H': [1.0], 'I': [0.4], 'L': [0.6], 'K': [0.4], 'M': [0.3], 'F': [0.7], 'P': [0.9], 'S': [0.4], 'T': [0.4], 'W': [0.6], 'Y': [1.2], 'V': [0.4]} ,
           'simz760101' :  {'A': [0.73], 'R': [0.73], 'N': [-0.01], 'D': [0.54], 'C': [0.7], 'Q': [-0.1], 'E': [0.55], 'G': [0.0], 'H': [1.1], 'I': [2.97], 'L': [2.49], 'K': [1.5], 'M': [1.3], 'F': [2.65], 'P': [2.6], 'S': [0.04], 'T': [0.44], 'W': [3.0], 'Y': [2.97], 'V': [1.69]} ,
           'snep660101' :  {'A': [0.239], 'R': [0.211], 'N': [0.249], 'D': [0.171], 'C': [0.22], 'Q': [0.26], 'E': [0.187], 'G': [0.16], 'H': [0.205], 'I': [0.273], 'L': [0.281], 'K': [0.228], 'M': [0.253], 'F': [0.234], 'P': [0.165], 'S': [0.236], 'T': [0.213], 'W': [0.183], 'Y': [0.193], 'V': [0.255]} ,
           'snep660102' :  {'A': [0.33], 'R': [-0.176], 'N': [-0.233], 'D': [-0.371], 'C': [0.074], 'Q': [-0.254], 'E': [-0.409], 'G': [0.37], 'H': [-0.078], 'I': [0.149], 'L': [0.129], 'K': [-0.075], 'M': [-0.092], 'F': [-0.011], 'P': [0.37], 'S': [0.022], 'T': [0.136], 'W': [-0.011], 'Y': [-0.138], 'V': [0.245]} ,
           'snep660103' :  {'A': [-0.11], 'R': [0.079], 'N': [-0.136], 'D': [-0.285], 'C': [-0.184], 'Q': [-0.067], 'E': [-0.246], 'G': [-0.073], 'H': [0.32], 'I': [0.001], 'L': [-0.008], 'K': [0.049], 'M': [-0.041], 'F': [0.438], 'P': [-0.016], 'S': [-0.153], 'T': [-0.208], 'W': [0.493], 'Y': [0.381], 'V': [-0.155]} ,
           'snep660104' :  {'A': [-0.062], 'R': [-0.167], 'N': [0.166], 'D': [-0.079], 'C': [0.38], 'Q': [-0.025], 'E': [-0.184], 'G': [-0.017], 'H': [0.056], 'I': [-0.309], 'L': [-0.264], 'K': [-0.371], 'M': [0.077], 'F': [0.074], 'P': [-0.036], 'S': [0.47], 'T': [0.348], 'W': [0.05], 'Y': [0.22], 'V': [-0.212]} ,
           'suem840101' :  {'A': [1.071], 'R': [1.033], 'N': [0.784], 'D': [0.68], 'C': [0.922], 'Q': [0.977], 'E': [0.97], 'G': [0.591], 'H': [0.85], 'I': [1.14], 'L': [1.14], 'K': [0.939], 'M': [1.2], 'F': [1.086], 'P': [0.659], 'S': [0.76], 'T': [0.817], 'W': [1.107], 'Y': [1.02], 'V': [0.95]} ,
           'suem840102' :  {'A': [8.0], 'R': [0.1], 'N': [0.1], 'D': [70.0], 'C': [26.0], 'Q': [33.0], 'E': [6.0], 'G': [0.1], 'H': [0.1], 'I': [55.0], 'L': [33.0], 'K': [1.0], 'M': [54.0], 'F': [18.0], 'P': [42.0], 'S': [0.1], 'T': [0.1], 'W': [77.0], 'Y': [66.0], 'V': [0.1]} ,
           'swer830101' :  {'A': [-0.4], 'R': [-0.59], 'N': [-0.92], 'D': [-1.31], 'C': [0.17], 'Q': [-0.91], 'E': [-1.22], 'G': [-0.67], 'H': [-0.64], 'I': [1.25], 'L': [1.22], 'K': [-0.67], 'M': [1.02], 'F': [1.92], 'P': [-0.49], 'S': [-0.55], 'T': [-0.28], 'W': [0.5], 'Y': [1.67], 'V': [0.91]} ,
           'tans770101' :  {'A': [1.42], 'R': [1.06], 'N': [0.71], 'D': [1.01], 'C': [0.73], 'Q': [1.02], 'E': [1.63], 'G': [0.5], 'H': [1.2], 'I': [1.12], 'L': [1.29], 'K': [1.24], 'M': [1.21], 'F': [1.16], 'P': [0.65], 'S': [0.71], 'T': [0.78], 'W': [1.05], 'Y': [0.67], 'V': [0.99]} ,
           'tans770102' :  {'A': [0.946], 'R': [1.128], 'N': [0.432], 'D': [1.311], 'C': [0.481], 'Q': [1.615], 'E': [0.698], 'G': [0.36], 'H': [2.168], 'I': [1.283], 'L': [1.192], 'K': [1.203], 'M': [0.0], 'F': [0.963], 'P': [2.093], 'S': [0.523], 'T': [1.961], 'W': [1.925], 'Y': [0.802], 'V': [0.409]} ,
           'tans770103' :  {'A': [0.79], 'R': [1.087], 'N': [0.832], 'D': [0.53], 'C': [1.268], 'Q': [1.038], 'E': [0.643], 'G': [0.725], 'H': [0.864], 'I': [1.361], 'L': [1.111], 'K': [0.735], 'M': [1.092], 'F': [1.052], 'P': [1.249], 'S': [1.093], 'T': [1.214], 'W': [1.114], 'Y': [1.34], 'V': [1.428]} ,
           'tans770104' :  {'A': [1.194], 'R': [0.795], 'N': [0.659], 'D': [1.056], 'C': [0.678], 'Q': [1.29], 'E': [0.928], 'G': [1.015], 'H': [0.611], 'I': [0.603], 'L': [0.595], 'K': [1.06], 'M': [0.831], 'F': [0.377], 'P': [3.159], 'S': [1.444], 'T': [1.172], 'W': [0.452], 'Y': [0.816], 'V': [0.64]} ,
           'tans770105' :  {'A': [0.497], 'R': [0.677], 'N': [2.072], 'D': [1.498], 'C': [1.348], 'Q': [0.711], 'E': [0.651], 'G': [1.848], 'H': [1.474], 'I': [0.471], 'L': [0.656], 'K': [0.932], 'M': [0.425], 'F': [1.348], 'P': [0.179], 'S': [1.151], 'T': [0.749], 'W': [1.283], 'Y': [1.283], 'V': [0.654]} ,
           'tans770106' :  {'A': [0.937], 'R': [1.725], 'N': [1.08], 'D': [1.64], 'C': [1.004], 'Q': [1.078], 'E': [0.679], 'G': [0.901], 'H': [1.085], 'I': [0.178], 'L': [0.808], 'K': [1.254], 'M': [0.886], 'F': [0.803], 'P': [0.748], 'S': [1.145], 'T': [1.487], 'W': [0.803], 'Y': [1.227], 'V': [0.625]} ,
           'tans770107' :  {'A': [0.289], 'R': [1.38], 'N': [3.169], 'D': [0.917], 'C': [1.767], 'Q': [2.372], 'E': [0.285], 'G': [4.259], 'H': [1.061], 'I': [0.262], 'L': [0.0], 'K': [1.288], 'M': [0.0], 'F': [0.393], 'P': [0.0], 'S': [0.16], 'T': [0.218], 'W': [0.0], 'Y': [0.654], 'V': [0.167]} ,
           'tans770108' :  {'A': [0.328], 'R': [2.088], 'N': [1.498], 'D': [3.379], 'C': [0.0], 'Q': [0.0], 'E': [0.0], 'G': [0.5], 'H': [1.204], 'I': [2.078], 'L': [0.414], 'K': [0.835], 'M': [0.982], 'F': [1.336], 'P': [0.415], 'S': [1.089], 'T': [1.732], 'W': [1.781], 'Y': [0.0], 'V': [0.946]} ,
           'tans770109' :  {'A': [0.945], 'R': [0.364], 'N': [1.202], 'D': [1.315], 'C': [0.932], 'Q': [0.704], 'E': [1.014], 'G': [2.355], 'H': [0.525], 'I': [0.673], 'L': [0.758], 'K': [0.947], 'M': [1.028], 'F': [0.622], 'P': [0.579], 'S': [1.14], 'T': [0.863], 'W': [0.777], 'Y': [0.907], 'V': [0.561]} ,
           'tans770110' :  {'A': [0.842], 'R': [0.936], 'N': [1.352], 'D': [1.366], 'C': [1.032], 'Q': [0.998], 'E': [0.758], 'G': [1.349], 'H': [1.079], 'I': [0.459], 'L': [0.665], 'K': [1.045], 'M': [0.668], 'F': [0.881], 'P': [1.385], 'S': [1.257], 'T': [1.055], 'W': [0.881], 'Y': [1.101], 'V': [0.643]} ,
           'vasm830101' :  {'A': [0.135], 'R': [0.296], 'N': [0.196], 'D': [0.289], 'C': [0.159], 'Q': [0.236], 'E': [0.184], 'G': [0.051], 'H': [0.223], 'I': [0.173], 'L': [0.215], 'K': [0.17], 'M': [0.239], 'F': [0.087], 'P': [0.151], 'S': [0.01], 'T': [0.1], 'W': [0.166], 'Y': [0.066], 'V': [0.285]} ,
           'vasm830102' :  {'A': [0.507], 'R': [0.459], 'N': [0.287], 'D': [0.223], 'C': [0.592], 'Q': [0.383], 'E': [0.445], 'G': [0.39], 'H': [0.31], 'I': [0.111], 'L': [0.619], 'K': [0.559], 'M': [0.431], 'F': [0.077], 'P': [0.739], 'S': [0.689], 'T': [0.785], 'W': [0.16], 'Y': [0.06], 'V': [0.356]} ,
           'vasm830103' :  {'A': [0.159], 'R': [0.194], 'N': [0.385], 'D': [0.283], 'C': [0.187], 'Q': [0.236], 'E': [0.206], 'G': [0.049], 'H': [0.233], 'I': [0.581], 'L': [0.083], 'K': [0.159], 'M': [0.198], 'F': [0.682], 'P': [0.366], 'S': [0.15], 'T': [0.074], 'W': [0.463], 'Y': [0.737], 'V': [0.301]} ,
           'velv850101' :  {'A': [0.03731], 'R': [0.09593], 'N': [0.00359], 'D': [0.1263], 'C': [0.08292], 'Q': [0.07606], 'E': [0.0058], 'G': [0.00499], 'H': [0.02415], 'I': [0.0], 'L': [0.0], 'K': [0.0371], 'M': [0.08226], 'F': [0.0946], 'P': [0.01979], 'S': [0.08292], 'T': [0.09408], 'W': [0.05481], 'Y': [0.05159], 'V': [0.00569]} ,
           'vent840101' :  {'A': [0.0], 'R': [0.0], 'N': [0.0], 'D': [0.0], 'C': [0.0], 'Q': [0.0], 'E': [0.0], 'G': [0.0], 'H': [0.0], 'I': [1.0], 'L': [1.0], 'K': [0.0], 'M': [0.0], 'F': [1.0], 'P': [0.0], 'S': [0.0], 'T': [0.0], 'W': [1.0], 'Y': [1.0], 'V': [1.0]} ,
           'vheg790101' :  {'A': [-12.04], 'R': [39.23], 'N': [4.25], 'D': [23.22], 'C': [3.95], 'Q': [2.16], 'E': [16.81], 'G': [-7.85], 'H': [6.28], 'I': [-18.32], 'L': [-17.79], 'K': [9.71], 'M': [-8.86], 'F': [-21.98], 'P': [5.82], 'S': [-1.54], 'T': [-4.15], 'W': [-16.19], 'Y': [-1.51], 'V': [-16.22]} ,
           'warp780101' :  {'A': [10.04], 'R': [6.18], 'N': [5.63], 'D': [5.76], 'C': [8.89], 'Q': [5.41], 'E': [5.37], 'G': [7.99], 'H': [7.49], 'I': [8.72], 'L': [8.79], 'K': [4.4], 'M': [9.15], 'F': [7.98], 'P': [7.79], 'S': [7.08], 'T': [7.0], 'W': [8.07], 'Y': [6.9], 'V': [8.88]} ,
           'weba780101' :  {'A': [0.89], 'R': [0.88], 'N': [0.89], 'D': [0.87], 'C': [0.85], 'Q': [0.82], 'E': [0.84], 'G': [0.92], 'H': [0.83], 'I': [0.76], 'L': [0.73], 'K': [0.97], 'M': [0.74], 'F': [0.52], 'P': [0.82], 'S': [0.96], 'T': [0.92], 'W': [0.2], 'Y': [0.49], 'V': [0.85]} ,
           'werd780101' :  {'A': [0.52], 'R': [0.49], 'N': [0.42], 'D': [0.37], 'C': [0.83], 'Q': [0.35], 'E': [0.38], 'G': [0.41], 'H': [0.7], 'I': [0.79], 'L': [0.77], 'K': [0.31], 'M': [0.76], 'F': [0.87], 'P': [0.35], 'S': [0.49], 'T': [0.38], 'W': [0.86], 'Y': [0.64], 'V': [0.72]} ,
           'werd780102' :  {'A': [0.16], 'R': [-0.2], 'N': [1.03], 'D': [-0.24], 'C': [-0.12], 'Q': [-0.55], 'E': [-0.45], 'G': [-0.16], 'H': [-0.18], 'I': [-0.19], 'L': [-0.44], 'K': [-0.12], 'M': [-0.79], 'F': [-0.25], 'P': [-0.59], 'S': [-0.01], 'T': [0.05], 'W': [-0.33], 'Y': [-0.42], 'V': [-0.46]} ,
           'werd780103' :  {'A': [0.15], 'R': [-0.37], 'N': [0.69], 'D': [-0.22], 'C': [-0.19], 'Q': [-0.06], 'E': [0.14], 'G': [0.36], 'H': [-0.25], 'I': [0.02], 'L': [0.06], 'K': [-0.16], 'M': [0.11], 'F': [1.18], 'P': [0.11], 'S': [0.13], 'T': [0.28], 'W': [-0.12], 'Y': [0.19], 'V': [-0.08]} ,
           'werd780104' :  {'A': [-0.07], 'R': [-0.4], 'N': [-0.57], 'D': [-0.8], 'C': [0.17], 'Q': [-0.26], 'E': [-0.63], 'G': [0.27], 'H': [-0.49], 'I': [0.06], 'L': [-0.17], 'K': [-0.45], 'M': [0.03], 'F': [0.4], 'P': [-0.47], 'S': [-0.11], 'T': [0.09], 'W': [-0.61], 'Y': [-0.61], 'V': [-0.11]} ,
           'woec730101' :  {'A': [7.0], 'R': [9.1], 'N': [10.0], 'D': [13.0], 'C': [5.5], 'Q': [8.6], 'E': [12.5], 'G': [7.9], 'H': [8.4], 'I': [4.9], 'L': [4.9], 'K': [10.1], 'M': [5.3], 'F': [5.0], 'P': [6.6], 'S': [7.5], 'T': [6.6], 'W': [5.3], 'Y': [5.7], 'V': [5.6]} ,
           'wolr810101' :  {'A': [1.94], 'R': [-19.92], 'N': [-9.68], 'D': [-10.95], 'C': [-1.24], 'Q': [-9.38], 'E': [-10.2], 'G': [2.39], 'H': [-10.27], 'I': [2.15], 'L': [2.28], 'K': [-9.52], 'M': [-1.48], 'F': [-0.76], 'P': [-3.68], 'S': [-5.06], 'T': [-4.88], 'W': [-5.88], 'Y': [-6.11], 'V': [1.99]} ,
           'wols870101' :  {'A': [0.07], 'R': [2.88], 'N': [3.22], 'D': [3.64], 'C': [0.71], 'Q': [2.18], 'E': [3.08], 'G': [2.23], 'H': [2.41], 'I': [-4.44], 'L': [-4.19], 'K': [2.84], 'M': [-2.49], 'F': [-4.92], 'P': [-1.22], 'S': [1.96], 'T': [0.92], 'W': [-4.75], 'Y': [-1.39], 'V': [-2.69]} ,
           'wols870102' :  {'A': [-1.73], 'R': [2.52], 'N': [1.45], 'D': [1.13], 'C': [-0.97], 'Q': [0.53], 'E': [0.39], 'G': [-5.36], 'H': [1.74], 'I': [-1.68], 'L': [-1.03], 'K': [1.41], 'M': [-0.27], 'F': [1.3], 'P': [0.88], 'S': [-1.63], 'T': [-2.09], 'W': [3.65], 'Y': [2.32], 'V': [-2.53]} ,
           'wols870103' :  {'A': [0.09], 'R': [-3.44], 'N': [0.84], 'D': [2.36], 'C': [4.13], 'Q': [-1.14], 'E': [-0.07], 'G': [0.3], 'H': [1.11], 'I': [-1.03], 'L': [-0.98], 'K': [-3.14], 'M': [-0.41], 'F': [0.45], 'P': [2.23], 'S': [0.57], 'T': [-1.4], 'W': [0.85], 'Y': [0.01], 'V': [-1.29]} ,
           'yutk870101' :  {'A': [8.5], 'R': [0.0], 'N': [8.2], 'D': [8.5], 'C': [11.0], 'Q': [6.3], 'E': [8.8], 'G': [7.1], 'H': [10.1], 'I': [16.8], 'L': [15.0], 'K': [7.9], 'M': [13.3], 'F': [11.2], 'P': [8.2], 'S': [7.4], 'T': [8.8], 'W': [9.9], 'Y': [8.8], 'V': [12.0]} ,
           'yutk870102' :  {'A': [6.8], 'R': [0.0], 'N': [6.2], 'D': [7.0], 'C': [8.3], 'Q': [8.5], 'E': [4.9], 'G': [6.4], 'H': [9.2], 'I': [10.0], 'L': [12.2], 'K': [7.5], 'M': [8.4], 'F': [8.3], 'P': [6.9], 'S': [8.0], 'T': [7.0], 'W': [5.7], 'Y': [6.8], 'V': [9.4]} ,
           'yutk870103' :  {'A': [18.08], 'R': [0.0], 'N': [17.47], 'D': [17.36], 'C': [18.17], 'Q': [17.93], 'E': [18.16], 'G': [18.24], 'H': [18.49], 'I': [18.62], 'L': [18.6], 'K': [17.96], 'M': [18.11], 'F': [17.3], 'P': [18.16], 'S': [17.57], 'T': [17.54], 'W': [17.19], 'Y': [17.99], 'V': [18.3]} ,
           'yutk870104' :  {'A': [18.56], 'R': [0.0], 'N': [18.24], 'D': [17.94], 'C': [17.84], 'Q': [18.51], 'E': [17.97], 'G': [18.57], 'H': [18.64], 'I': [19.21], 'L': [19.01], 'K': [18.36], 'M': [18.49], 'F': [17.95], 'P': [18.77], 'S': [18.06], 'T': [17.71], 'W': [16.87], 'Y': [18.23], 'V': [18.98]} ,
           'zasb820101' :  {'A': [-0.152], 'R': [-0.089], 'N': [-0.203], 'D': [-0.355], 'C': [0.0], 'Q': [-0.181], 'E': [-0.411], 'G': [-0.19], 'H': [0.0], 'I': [-0.086], 'L': [-0.102], 'K': [-0.062], 'M': [-0.107], 'F': [0.001], 'P': [-0.181], 'S': [-0.203], 'T': [-0.17], 'W': [0.275], 'Y': [0.0], 'V': [-0.125]} ,
           'zimj680101' :  {'A': [0.83], 'R': [0.83], 'N': [0.09], 'D': [0.64], 'C': [1.48], 'Q': [0.0], 'E': [0.65], 'G': [0.1], 'H': [1.1], 'I': [3.07], 'L': [2.52], 'K': [1.6], 'M': [1.4], 'F': [2.75], 'P': [2.7], 'S': [0.14], 'T': [0.54], 'W': [0.31], 'Y': [2.97], 'V': [1.79]} ,
           'zimj680102' :  {'A': [11.5], 'R': [14.28], 'N': [12.82], 'D': [11.68], 'C': [13.46], 'Q': [14.45], 'E': [13.57], 'G': [3.4], 'H': [13.69], 'I': [21.4], 'L': [21.4], 'K': [15.71], 'M': [16.25], 'F': [19.8], 'P': [17.43], 'S': [9.47], 'T': [15.77], 'W': [21.67], 'Y': [18.03], 'V': [21.57]} ,
           'zimj680103' :  {'A': [0.0], 'R': [52.0], 'N': [3.38], 'D': [49.7], 'C': [1.48], 'Q': [3.53], 'E': [49.9], 'G': [0.0], 'H': [51.6], 'I': [0.13], 'L': [0.13], 'K': [49.5], 'M': [1.43], 'F': [0.35], 'P': [1.58], 'S': [1.67], 'T': [1.66], 'W': [2.1], 'Y': [1.61], 'V': [0.13]} ,
           'zimj680104' :  {'A': [6.0], 'R': [10.76], 'N': [5.41], 'D': [2.77], 'C': [5.05], 'Q': [5.65], 'E': [3.22], 'G': [5.97], 'H': [7.59], 'I': [6.02], 'L': [5.98], 'K': [9.74], 'M': [5.74], 'F': [5.48], 'P': [6.3], 'S': [5.68], 'T': [5.66], 'W': [5.89], 'Y': [5.66], 'V': [5.96]} ,
           'zimj680105' :  {'A': [9.9], 'R': [4.6], 'N': [5.4], 'D': [2.8], 'C': [2.8], 'Q': [9.0], 'E': [3.2], 'G': [5.6], 'H': [8.2], 'I': [17.1], 'L': [17.6], 'K': [3.5], 'M': [14.9], 'F': [18.8], 'P': [14.8], 'S': [6.9], 'T': [9.5], 'W': [17.1], 'Y': [15.0], 'V': [14.3]} ,
           'aurr980101' :  {'A': [0.94], 'R': [1.15], 'N': [0.79], 'D': [1.19], 'C': [0.6], 'Q': [0.94], 'E': [1.41], 'G': [1.18], 'H': [1.15], 'I': [1.07], 'L': [0.95], 'K': [1.03], 'M': [0.88], 'F': [1.06], 'P': [1.18], 'S': [0.69], 'T': [0.87], 'W': [0.91], 'Y': [1.04], 'V': [0.9]} ,
           'aurr980102' :  {'A': [0.98], 'R': [1.14], 'N': [1.05], 'D': [1.05], 'C': [0.41], 'Q': [0.9], 'E': [1.04], 'G': [1.25], 'H': [1.01], 'I': [0.88], 'L': [0.8], 'K': [1.06], 'M': [1.12], 'F': [1.12], 'P': [1.31], 'S': [1.02], 'T': [0.8], 'W': [0.9], 'Y': [1.12], 'V': [0.87]} ,
           'aurr980103' :  {'A': [1.05], 'R': [0.81], 'N': [0.91], 'D': [1.39], 'C': [0.6], 'Q': [0.87], 'E': [1.11], 'G': [1.26], 'H': [1.43], 'I': [0.95], 'L': [0.96], 'K': [0.97], 'M': [0.99], 'F': [0.95], 'P': [1.05], 'S': [0.96], 'T': [1.03], 'W': [1.06], 'Y': [0.94], 'V': [0.62]} ,
           'aurr980104' :  {'A': [0.75], 'R': [0.9], 'N': [1.24], 'D': [1.72], 'C': [0.66], 'Q': [1.08], 'E': [1.1], 'G': [1.14], 'H': [0.96], 'I': [0.8], 'L': [1.01], 'K': [0.66], 'M': [1.02], 'F': [0.88], 'P': [1.33], 'S': [1.2], 'T': [1.13], 'W': [0.68], 'Y': [0.8], 'V': [0.58]} ,
           'aurr980105' :  {'A': [0.67], 'R': [0.76], 'N': [1.28], 'D': [1.58], 'C': [0.37], 'Q': [1.05], 'E': [0.94], 'G': [0.98], 'H': [0.83], 'I': [0.78], 'L': [0.79], 'K': [0.84], 'M': [0.98], 'F': [0.96], 'P': [1.12], 'S': [1.25], 'T': [1.41], 'W': [0.94], 'Y': [0.82], 'V': [0.67]} ,
           'aurr980106' :  {'A': [1.1], 'R': [1.05], 'N': [0.72], 'D': [1.14], 'C': [0.26], 'Q': [1.31], 'E': [2.3], 'G': [0.55], 'H': [0.83], 'I': [1.06], 'L': [0.84], 'K': [1.08], 'M': [0.9], 'F': [0.9], 'P': [1.67], 'S': [0.81], 'T': [0.77], 'W': [1.26], 'Y': [0.99], 'V': [0.76]} ,
           'aurr980107' :  {'A': [1.39], 'R': [0.95], 'N': [0.67], 'D': [1.64], 'C': [0.52], 'Q': [1.6], 'E': [2.07], 'G': [0.65], 'H': [1.36], 'I': [0.64], 'L': [0.91], 'K': [0.8], 'M': [1.1], 'F': [1.0], 'P': [0.94], 'S': [0.69], 'T': [0.92], 'W': [1.1], 'Y': [0.73], 'V': [0.7]} ,
           'aurr980108' :  {'A': [1.43], 'R': [1.33], 'N': [0.55], 'D': [0.9], 'C': [0.52], 'Q': [1.43], 'E': [1.7], 'G': [0.56], 'H': [0.66], 'I': [1.18], 'L': [1.52], 'K': [0.82], 'M': [1.68], 'F': [1.1], 'P': [0.15], 'S': [0.61], 'T': [0.75], 'W': [1.68], 'Y': [0.65], 'V': [1.14]} ,
           'aurr980109' :  {'A': [1.55], 'R': [1.39], 'N': [0.6], 'D': [0.61], 'C': [0.59], 'Q': [1.43], 'E': [1.34], 'G': [0.37], 'H': [0.89], 'I': [1.47], 'L': [1.36], 'K': [1.27], 'M': [2.13], 'F': [1.39], 'P': [0.03], 'S': [0.44], 'T': [0.65], 'W': [1.1], 'Y': [0.93], 'V': [1.18]} ,
           'aurr980110' :  {'A': [1.8], 'R': [1.73], 'N': [0.73], 'D': [0.9], 'C': [0.55], 'Q': [0.97], 'E': [1.73], 'G': [0.32], 'H': [0.46], 'I': [1.09], 'L': [1.47], 'K': [1.24], 'M': [1.64], 'F': [0.96], 'P': [0.15], 'S': [0.67], 'T': [0.7], 'W': [0.68], 'Y': [0.91], 'V': [0.81]} ,
           'aurr980111' :  {'A': [1.52], 'R': [1.49], 'N': [0.58], 'D': [1.04], 'C': [0.26], 'Q': [1.41], 'E': [1.76], 'G': [0.3], 'H': [0.83], 'I': [1.25], 'L': [1.26], 'K': [1.1], 'M': [1.14], 'F': [1.14], 'P': [0.44], 'S': [0.66], 'T': [0.73], 'W': [0.68], 'Y': [1.04], 'V': [1.03]} ,
           'aurr980112' :  {'A': [1.49], 'R': [1.41], 'N': [0.67], 'D': [0.94], 'C': [0.37], 'Q': [1.52], 'E': [1.55], 'G': [0.29], 'H': [0.96], 'I': [1.04], 'L': [1.4], 'K': [1.17], 'M': [1.84], 'F': [0.86], 'P': [0.2], 'S': [0.68], 'T': [0.79], 'W': [1.52], 'Y': [1.06], 'V': [0.94]} ,
           'aurr980113' :  {'A': [1.73], 'R': [1.24], 'N': [0.7], 'D': [0.68], 'C': [0.63], 'Q': [0.88], 'E': [1.16], 'G': [0.32], 'H': [0.76], 'I': [1.15], 'L': [1.8], 'K': [1.22], 'M': [2.21], 'F': [1.35], 'P': [0.07], 'S': [0.65], 'T': [0.46], 'W': [1.57], 'Y': [1.1], 'V': [0.94]} ,
           'aurr980114' :  {'A': [1.33], 'R': [1.39], 'N': [0.64], 'D': [0.6], 'C': [0.44], 'Q': [1.37], 'E': [1.43], 'G': [0.2], 'H': [1.02], 'I': [1.58], 'L': [1.63], 'K': [1.71], 'M': [1.76], 'F': [1.22], 'P': [0.07], 'S': [0.42], 'T': [0.57], 'W': [1.0], 'Y': [1.02], 'V': [1.08]} ,
           'aurr980115' :  {'A': [1.87], 'R': [1.66], 'N': [0.7], 'D': [0.91], 'C': [0.33], 'Q': [1.24], 'E': [1.88], 'G': [0.33], 'H': [0.89], 'I': [0.9], 'L': [1.65], 'K': [1.63], 'M': [1.35], 'F': [0.67], 'P': [0.03], 'S': [0.71], 'T': [0.5], 'W': [1.0], 'Y': [0.73], 'V': [0.51]} ,
           'aurr980116' :  {'A': [1.19], 'R': [1.45], 'N': [1.33], 'D': [0.72], 'C': [0.44], 'Q': [1.43], 'E': [1.27], 'G': [0.74], 'H': [1.55], 'I': [0.61], 'L': [1.36], 'K': [1.45], 'M': [1.35], 'F': [1.2], 'P': [0.1], 'S': [1.02], 'T': [0.82], 'W': [0.58], 'Y': [1.06], 'V': [0.46]} ,
           'aurr980117' :  {'A': [0.77], 'R': [1.11], 'N': [1.39], 'D': [0.79], 'C': [0.44], 'Q': [0.95], 'E': [0.92], 'G': [2.74], 'H': [1.65], 'I': [0.64], 'L': [0.66], 'K': [1.19], 'M': [0.74], 'F': [1.04], 'P': [0.66], 'S': [0.64], 'T': [0.82], 'W': [0.58], 'Y': [0.93], 'V': [0.53]} ,
           'aurr980118' :  {'A': [0.93], 'R': [0.96], 'N': [0.82], 'D': [1.15], 'C': [0.67], 'Q': [1.02], 'E': [1.07], 'G': [1.08], 'H': [1.4], 'I': [1.14], 'L': [1.16], 'K': [1.27], 'M': [1.11], 'F': [1.05], 'P': [1.01], 'S': [0.71], 'T': [0.84], 'W': [1.06], 'Y': [1.15], 'V': [0.74]} ,
           'aurr980119' :  {'A': [1.09], 'R': [1.29], 'N': [1.03], 'D': [1.17], 'C': [0.26], 'Q': [1.08], 'E': [1.31], 'G': [0.97], 'H': [0.88], 'I': [0.97], 'L': [0.87], 'K': [1.13], 'M': [0.96], 'F': [0.84], 'P': [2.01], 'S': [0.76], 'T': [0.79], 'W': [0.91], 'Y': [0.64], 'V': [0.77]} ,
           'aurr980120' :  {'A': [0.71], 'R': [1.09], 'N': [0.95], 'D': [1.43], 'C': [0.65], 'Q': [0.87], 'E': [1.19], 'G': [1.07], 'H': [1.13], 'I': [1.05], 'L': [0.84], 'K': [1.1], 'M': [0.8], 'F': [0.95], 'P': [1.7], 'S': [0.65], 'T': [0.086], 'W': [1.25], 'Y': [0.85], 'V': [1.12]} ,
           'onek900101' :  {'A': [13.4], 'R': [13.3], 'N': [12.0], 'D': [11.7], 'C': [11.6], 'Q': [12.8], 'E': [12.2], 'G': [11.3], 'H': [11.6], 'I': [12.0], 'L': [13.0], 'K': [13.0], 'M': [12.8], 'F': [12.1], 'P': [6.5], 'S': [12.2], 'T': [11.7], 'W': [12.4], 'Y': [12.1], 'V': [11.9]} ,
           'onek900102' :  {'A': [-0.77], 'R': [-0.68], 'N': [-0.07], 'D': [-0.15], 'C': [-0.23], 'Q': [-0.33], 'E': [-0.27], 'G': [0.0], 'H': [-0.06], 'I': [-0.23], 'L': [-0.62], 'K': [-0.65], 'M': [-0.5], 'F': [-0.41], 'P': [3.0], 'S': [-0.35], 'T': [-0.11], 'W': [-0.45], 'Y': [-0.17], 'V': [-0.14]} ,
           'vinm940101' :  {'A': [0.984], 'R': [1.008], 'N': [1.048], 'D': [1.068], 'C': [0.906], 'Q': [1.037], 'E': [1.094], 'G': [1.031], 'H': [0.95], 'I': [0.927], 'L': [0.935], 'K': [1.102], 'M': [0.952], 'F': [0.915], 'P': [1.049], 'S': [1.046], 'T': [0.997], 'W': [0.904], 'Y': [0.929], 'V': [0.931]} ,
           'vinm940102' :  {'A': [1.315], 'R': [1.31], 'N': [1.38], 'D': [1.372], 'C': [1.196], 'Q': [1.342], 'E': [1.376], 'G': [1.382], 'H': [1.279], 'I': [1.241], 'L': [1.234], 'K': [1.367], 'M': [1.269], 'F': [1.247], 'P': [1.342], 'S': [1.381], 'T': [1.324], 'W': [1.186], 'Y': [1.199], 'V': [1.235]} ,
           'vinm940103' :  {'A': [0.994], 'R': [1.026], 'N': [1.022], 'D': [1.022], 'C': [0.939], 'Q': [1.041], 'E': [1.052], 'G': [1.018], 'H': [0.967], 'I': [0.977], 'L': [0.982], 'K': [1.029], 'M': [0.963], 'F': [0.934], 'P': [1.05], 'S': [1.025], 'T': [0.998], 'W': [0.938], 'Y': [0.981], 'V': [0.968]} ,
           'vinm940104' :  {'A': [0.783], 'R': [0.807], 'N': [0.799], 'D': [0.822], 'C': [0.785], 'Q': [0.817], 'E': [0.826], 'G': [0.784], 'H': [0.777], 'I': [0.776], 'L': [0.783], 'K': [0.834], 'M': [0.806], 'F': [0.774], 'P': [0.809], 'S': [0.811], 'T': [0.795], 'W': [0.796], 'Y': [0.788], 'V': [0.781]} ,
           'munv940101' :  {'A': [0.423], 'R': [0.503], 'N': [0.906], 'D': [0.87], 'C': [0.877], 'Q': [0.594], 'E': [0.167], 'G': [1.162], 'H': [0.802], 'I': [0.566], 'L': [0.494], 'K': [0.615], 'M': [0.444], 'F': [0.706], 'P': [1.945], 'S': [0.928], 'T': [0.884], 'W': [0.69], 'Y': [0.778], 'V': [0.706]} ,
           'munv940102' :  {'A': [0.619], 'R': [0.753], 'N': [1.089], 'D': [0.932], 'C': [1.107], 'Q': [0.77], 'E': [0.675], 'G': [1.361], 'H': [1.034], 'I': [0.876], 'L': [0.74], 'K': [0.784], 'M': [0.736], 'F': [0.968], 'P': [1.78], 'S': [0.969], 'T': [1.053], 'W': [0.91], 'Y': [1.009], 'V': [0.939]} ,
           'munv940103' :  {'A': [1.08], 'R': [0.976], 'N': [1.197], 'D': [1.266], 'C': [0.733], 'Q': [1.05], 'E': [1.085], 'G': [1.104], 'H': [0.906], 'I': [0.583], 'L': [0.789], 'K': [1.026], 'M': [0.812], 'F': [0.685], 'P': [1.412], 'S': [0.987], 'T': [0.784], 'W': [0.755], 'Y': [0.665], 'V': [0.546]} ,
           'munv940104' :  {'A': [0.978], 'R': [0.784], 'N': [0.915], 'D': [1.038], 'C': [0.573], 'Q': [0.863], 'E': [0.962], 'G': [1.405], 'H': [0.724], 'I': [0.502], 'L': [0.766], 'K': [0.841], 'M': [0.729], 'F': [0.585], 'P': [2.613], 'S': [0.784], 'T': [0.569], 'W': [0.671], 'Y': [0.56], 'V': [0.444]} ,
           'munv940105' :  {'A': [1.4], 'R': [1.23], 'N': [1.61], 'D': [1.89], 'C': [1.14], 'Q': [1.33], 'E': [1.42], 'G': [2.06], 'H': [1.25], 'I': [1.02], 'L': [1.33], 'K': [1.34], 'M': [1.12], 'F': [1.07], 'P': [3.9], 'S': [1.2], 'T': [0.99], 'W': [1.1], 'Y': [0.98], 'V': [0.87]} ,
           'wimw960101' :  {'A': [4.08], 'R': [3.91], 'N': [3.83], 'D': [3.02], 'C': [4.49], 'Q': [3.67], 'E': [2.23], 'G': [4.24], 'H': [4.08], 'I': [4.52], 'L': [4.81], 'K': [3.77], 'M': [4.48], 'F': [5.38], 'P': [3.8], 'S': [4.12], 'T': [4.11], 'W': [6.1], 'Y': [5.19], 'V': [4.18]} ,
           'kimc930101' :  {'A': [-0.35], 'R': [-0.44], 'N': [-0.38], 'D': [-0.41], 'C': [-0.47], 'Q': [-0.4], 'E': [-0.41], 'G': [0.0], 'H': [-0.46], 'I': [-0.56], 'L': [-0.48], 'K': [-0.41], 'M': [-0.46], 'F': [-0.55], 'P': [-0.23], 'S': [-0.39], 'T': [-0.48], 'W': [-0.48], 'Y': [-0.5], 'V': [-0.53]} ,
           'monm990101' :  {'A': [0.5], 'R': [1.7], 'N': [1.7], 'D': [1.6], 'C': [0.6], 'Q': [1.6], 'E': [1.6], 'G': [1.3], 'H': [1.6], 'I': [0.6], 'L': [0.4], 'K': [1.6], 'M': [0.5], 'F': [0.4], 'P': [1.7], 'S': [0.7], 'T': [0.4], 'W': [0.7], 'Y': [0.6], 'V': [0.5]} ,
           'blam930101' :  {'A': [0.96], 'R': [0.77], 'N': [0.39], 'D': [0.42], 'C': [0.42], 'Q': [0.8], 'E': [0.53], 'G': [0.0], 'H': [0.57], 'I': [0.84], 'L': [0.92], 'K': [0.73], 'M': [0.86], 'F': [0.59], 'P': [-2.5], 'S': [0.53], 'T': [0.54], 'W': [0.58], 'Y': [0.72], 'V': [0.63]} ,
           'pars000101' :  {'A': [0.343], 'R': [0.353], 'N': [0.409], 'D': [0.429], 'C': [0.319], 'Q': [0.395], 'E': [0.405], 'G': [0.389], 'H': [0.307], 'I': [0.296], 'L': [0.287], 'K': [0.429], 'M': [0.293], 'F': [0.292], 'P': [0.432], 'S': [0.416], 'T': [0.362], 'W': [0.268], 'Y': [0.22], 'V': [0.307]} ,
           'pars000102' :  {'A': [0.32], 'R': [0.327], 'N': [0.384], 'D': [0.424], 'C': [0.198], 'Q': [0.436], 'E': [0.514], 'G': [0.374], 'H': [0.299], 'I': [0.306], 'L': [0.34], 'K': [0.446], 'M': [0.313], 'F': [0.314], 'P': [0.354], 'S': [0.376], 'T': [0.339], 'W': [0.291], 'Y': [0.287], 'V': [0.294]} ,
           'kums000101' :  {'A': [8.9], 'R': [4.6], 'N': [4.4], 'D': [6.3], 'C': [0.6], 'Q': [2.8], 'E': [6.9], 'G': [9.4], 'H': [2.2], 'I': [7.0], 'L': [7.4], 'K': [6.1], 'M': [2.3], 'F': [3.3], 'P': [4.2], 'S': [4.0], 'T': [5.7], 'W': [1.3], 'Y': [4.5], 'V': [8.2]} ,
           'kums000102' :  {'A': [9.2], 'R': [3.6], 'N': [5.1], 'D': [6.0], 'C': [1.0], 'Q': [2.9], 'E': [6.0], 'G': [9.4], 'H': [2.1], 'I': [6.0], 'L': [7.7], 'K': [6.5], 'M': [2.4], 'F': [3.4], 'P': [4.2], 'S': [5.5], 'T': [5.7], 'W': [1.2], 'Y': [3.7], 'V': [8.2]} ,
           'kums000103' :  {'A': [14.1], 'R': [5.5], 'N': [3.2], 'D': [5.7], 'C': [0.1], 'Q': [3.7], 'E': [8.8], 'G': [4.1], 'H': [2.0], 'I': [7.1], 'L': [9.1], 'K': [7.7], 'M': [3.3], 'F': [5.0], 'P': [0.7], 'S': [3.9], 'T': [4.4], 'W': [1.2], 'Y': [4.5], 'V': [5.9]} ,
           'kums000104' :  {'A': [13.4], 'R': [3.9], 'N': [3.7], 'D': [4.6], 'C': [0.8], 'Q': [4.8], 'E': [7.8], 'G': [4.6], 'H': [3.3], 'I': [6.5], 'L': [10.6], 'K': [7.5], 'M': [3.0], 'F': [4.5], 'P': [1.3], 'S': [3.8], 'T': [4.6], 'W': [1.0], 'Y': [3.3], 'V': [7.1]} ,
           'takk010101' :  {'A': [9.8], 'R': [7.3], 'N': [3.6], 'D': [4.9], 'C': [3.0], 'Q': [2.4], 'E': [4.4], 'G': [0.0], 'H': [11.9], 'I': [17.2], 'L': [17.0], 'K': [10.5], 'M': [11.9], 'F': [23.0], 'P': [15.0], 'S': [2.6], 'T': [6.9], 'W': [24.2], 'Y': [17.2], 'V': [15.3]} ,
           'fodm020101' :  {'A': [0.7], 'R': [0.95], 'N': [1.47], 'D': [0.87], 'C': [1.17], 'Q': [0.73], 'E': [0.96], 'G': [0.64], 'H': [1.39], 'I': [1.29], 'L': [1.44], 'K': [0.91], 'M': [0.91], 'F': [1.34], 'P': [0.12], 'S': [0.84], 'T': [0.74], 'W': [1.8], 'Y': [1.68], 'V': [1.2]} ,
           'nadh010101' :  {'A': [58.0], 'R': [-184.0], 'N': [-93.0], 'D': [-97.0], 'C': [116.0], 'Q': [-139.0], 'E': [-131.0], 'G': [-11.0], 'H': [-73.0], 'I': [107.0], 'L': [95.0], 'K': [-24.0], 'M': [78.0], 'F': [92.0], 'P': [-79.0], 'S': [-34.0], 'T': [-7.0], 'W': [59.0], 'Y': [-11.0], 'V': [100.0]} ,
           'nadh010102' :  {'A': [51.0], 'R': [-144.0], 'N': [-84.0], 'D': [-78.0], 'C': [137.0], 'Q': [-128.0], 'E': [-115.0], 'G': [-13.0], 'H': [-55.0], 'I': [106.0], 'L': [103.0], 'K': [-205.0], 'M': [73.0], 'F': [108.0], 'P': [-79.0], 'S': [-26.0], 'T': [-3.0], 'W': [69.0], 'Y': [11.0], 'V': [108.0]} ,
           'nadh010103' :  {'A': [41.0], 'R': [-109.0], 'N': [-74.0], 'D': [-47.0], 'C': [169.0], 'Q': [-104.0], 'E': [-90.0], 'G': [-18.0], 'H': [-35.0], 'I': [104.0], 'L': [103.0], 'K': [-148.0], 'M': [77.0], 'F': [128.0], 'P': [-81.0], 'S': [-31.0], 'T': [10.0], 'W': [102.0], 'Y': [36.0], 'V': [116.0]} ,
           'nadh010104' :  {'A': [32.0], 'R': [-95.0], 'N': [-73.0], 'D': [-29.0], 'C': [182.0], 'Q': [-95.0], 'E': [-74.0], 'G': [-22.0], 'H': [-25.0], 'I': [106.0], 'L': [104.0], 'K': [-124.0], 'M': [82.0], 'F': [132.0], 'P': [-82.0], 'S': [-34.0], 'T': [20.0], 'W': [118.0], 'Y': [44.0], 'V': [113.0]} ,
           'nadh010105' :  {'A': [24.0], 'R': [-79.0], 'N': [-76.0], 'D': [0.0], 'C': [194.0], 'Q': [-87.0], 'E': [-57.0], 'G': [-28.0], 'H': [-31.0], 'I': [102.0], 'L': [103.0], 'K': [-9.0], 'M': [90.0], 'F': [131.0], 'P': [-85.0], 'S': [-36.0], 'T': [34.0], 'W': [116.0], 'Y': [43.0], 'V': [111.0]} ,
           'nadh010106' :  {'A': [5.0], 'R': [-57.0], 'N': [-77.0], 'D': [45.0], 'C': [224.0], 'Q': [-67.0], 'E': [-8.0], 'G': [-47.0], 'H': [-50.0], 'I': [83.0], 'L': [82.0], 'K': [-38.0], 'M': [83.0], 'F': [117.0], 'P': [-103.0], 'S': [-41.0], 'T': [79.0], 'W': [130.0], 'Y': [27.0], 'V': [117.0]} ,
           'nadh010107' :  {'A': [-2.0], 'R': [-41.0], 'N': [-97.0], 'D': [248.0], 'C': [329.0], 'Q': [-37.0], 'E': [117.0], 'G': [-66.0], 'H': [-70.0], 'I': [28.0], 'L': [36.0], 'K': [115.0], 'M': [62.0], 'F': [120.0], 'P': [-132.0], 'S': [-52.0], 'T': [174.0], 'W': [179.0], 'Y': [-7.0], 'V': [114.0]} ,
           'monm990201' :  {'A': [0.4], 'R': [1.5], 'N': [1.6], 'D': [1.5], 'C': [0.7], 'Q': [1.4], 'E': [1.3], 'G': [1.1], 'H': [1.4], 'I': [0.5], 'L': [0.3], 'K': [1.4], 'M': [0.5], 'F': [0.3], 'P': [1.6], 'S': [0.9], 'T': [0.7], 'W': [0.9], 'Y': [0.9], 'V': [0.4]} ,
           'koep990101' :  {'A': [-0.04], 'R': [-0.3], 'N': [0.25], 'D': [0.27], 'C': [0.57], 'Q': [-0.02], 'E': [-0.33], 'G': [1.24], 'H': [-0.11], 'I': [-0.26], 'L': [-0.38], 'K': [-0.18], 'M': [-0.09], 'F': [-0.01], 'P': [0.0], 'S': [0.15], 'T': [0.39], 'W': [0.21], 'Y': [0.05], 'V': [-0.06]} ,
           'koep990102' :  {'A': [-0.12], 'R': [0.34], 'N': [1.05], 'D': [1.12], 'C': [-0.63], 'Q': [1.67], 'E': [0.91], 'G': [0.76], 'H': [1.34], 'I': [-0.77], 'L': [0.15], 'K': [0.29], 'M': [-0.71], 'F': [-0.67], 'P': [0.0], 'S': [1.45], 'T': [-0.7], 'W': [-0.14], 'Y': [-0.49], 'V': [-0.7]} ,
           'cedj970101' :  {'A': [8.6], 'R': [4.2], 'N': [4.6], 'D': [4.9], 'C': [2.9], 'Q': [4.0], 'E': [5.1], 'G': [7.8], 'H': [2.1], 'I': [4.6], 'L': [8.8], 'K': [6.3], 'M': [2.5], 'F': [3.7], 'P': [4.9], 'S': [7.3], 'T': [6.0], 'W': [1.4], 'Y': [3.6], 'V': [6.7]} ,
           'cedj970102' :  {'A': [7.6], 'R': [5.0], 'N': [4.4], 'D': [5.2], 'C': [2.2], 'Q': [4.1], 'E': [6.2], 'G': [6.9], 'H': [2.1], 'I': [5.1], 'L': [9.4], 'K': [5.8], 'M': [2.1], 'F': [4.0], 'P': [5.4], 'S': [7.2], 'T': [6.1], 'W': [1.4], 'Y': [3.2], 'V': [6.7]} ,
           'cedj970103' :  {'A': [8.1], 'R': [4.6], 'N': [3.7], 'D': [3.8], 'C': [2.0], 'Q': [3.1], 'E': [4.6], 'G': [7.0], 'H': [2.0], 'I': [6.7], 'L': [11.0], 'K': [4.4], 'M': [2.8], 'F': [5.6], 'P': [4.7], 'S': [7.3], 'T': [5.6], 'W': [1.8], 'Y': [3.3], 'V': [7.7]} ,
           'cedj970104' :  {'A': [7.9], 'R': [4.9], 'N': [4.0], 'D': [5.5], 'C': [1.9], 'Q': [4.4], 'E': [7.1], 'G': [7.1], 'H': [2.1], 'I': [5.2], 'L': [8.6], 'K': [6.7], 'M': [2.4], 'F': [3.9], 'P': [5.3], 'S': [6.6], 'T': [5.3], 'W': [1.2], 'Y': [3.1], 'V': [6.8]} ,
           'cedj970105' :  {'A': [8.3], 'R': [8.7], 'N': [3.7], 'D': [4.7], 'C': [1.6], 'Q': [4.7], 'E': [6.5], 'G': [6.3], 'H': [2.1], 'I': [3.7], 'L': [7.4], 'K': [7.9], 'M': [2.3], 'F': [2.7], 'P': [6.9], 'S': [8.8], 'T': [5.1], 'W': [0.7], 'Y': [2.4], 'V': [5.3]} ,
           'fuks010101' :  {'A': [4.47], 'R': [8.48], 'N': [3.89], 'D': [7.05], 'C': [0.29], 'Q': [2.87], 'E': [16.56], 'G': [8.29], 'H': [1.74], 'I': [3.3], 'L': [5.06], 'K': [12.98], 'M': [1.71], 'F': [2.32], 'P': [5.41], 'S': [4.27], 'T': [3.83], 'W': [0.67], 'Y': [2.75], 'V': [4.05]} ,
           'fuks010102' :  {'A': [6.77], 'R': [6.87], 'N': [5.5], 'D': [8.57], 'C': [0.31], 'Q': [5.24], 'E': [12.93], 'G': [7.95], 'H': [2.8], 'I': [2.72], 'L': [4.43], 'K': [10.2], 'M': [1.87], 'F': [1.92], 'P': [4.79], 'S': [5.41], 'T': [5.36], 'W': [0.54], 'Y': [2.26], 'V': [3.57]} ,
           'fuks010103' :  {'A': [7.43], 'R': [4.51], 'N': [9.12], 'D': [8.71], 'C': [0.42], 'Q': [5.42], 'E': [5.86], 'G': [9.4], 'H': [1.49], 'I': [1.76], 'L': [2.74], 'K': [9.67], 'M': [0.6], 'F': [1.18], 'P': [5.6], 'S': [9.6], 'T': [8.95], 'W': [1.18], 'Y': [3.26], 'V': [3.1]} ,
           'fuks010104' :  {'A': [5.22], 'R': [7.3], 'N': [6.06], 'D': [7.91], 'C': [1.01], 'Q': [6.0], 'E': [10.66], 'G': [5.81], 'H': [2.27], 'I': [2.36], 'L': [4.52], 'K': [12.68], 'M': [1.85], 'F': [1.68], 'P': [5.7], 'S': [6.99], 'T': [5.16], 'W': [0.56], 'Y': [2.16], 'V': [4.1]} ,
           'fuks010105' :  {'A': [9.88], 'R': [3.71], 'N': [2.35], 'D': [3.5], 'C': [1.12], 'Q': [1.66], 'E': [4.02], 'G': [6.88], 'H': [1.88], 'I': [10.08], 'L': [13.21], 'K': [3.39], 'M': [2.44], 'F': [5.27], 'P': [3.8], 'S': [4.1], 'T': [4.98], 'W': [1.11], 'Y': [4.07], 'V': [12.53]} ,
           'fuks010106' :  {'A': [10.98], 'R': [3.26], 'N': [2.85], 'D': [3.37], 'C': [1.47], 'Q': [2.3], 'E': [3.51], 'G': [7.48], 'H': [2.2], 'I': [9.74], 'L': [12.79], 'K': [2.54], 'M': [3.1], 'F': [4.97], 'P': [3.42], 'S': [4.93], 'T': [5.55], 'W': [1.28], 'Y': [3.55], 'V': [10.69]} ,
           'fuks010107' :  {'A': [9.95], 'R': [3.05], 'N': [4.84], 'D': [4.46], 'C': [1.3], 'Q': [2.64], 'E': [2.58], 'G': [8.87], 'H': [1.99], 'I': [7.73], 'L': [9.66], 'K': [2.0], 'M': [2.45], 'F': [5.41], 'P': [3.2], 'S': [6.03], 'T': [5.62], 'W': [2.6], 'Y': [6.15], 'V': [9.46]} ,
           'fuks010108' :  {'A': [8.26], 'R': [2.8], 'N': [2.54], 'D': [2.8], 'C': [2.67], 'Q': [2.86], 'E': [2.67], 'G': [5.62], 'H': [1.98], 'I': [8.95], 'L': [16.46], 'K': [1.89], 'M': [2.67], 'F': [7.32], 'P': [3.3], 'S': [6.0], 'T': [5.0], 'W': [2.01], 'Y': [3.96], 'V': [10.24]} ,
           'fuks010109' :  {'A': [7.39], 'R': [5.91], 'N': [3.06], 'D': [5.14], 'C': [0.74], 'Q': [2.22], 'E': [9.8], 'G': [7.53], 'H': [1.82], 'I': [6.96], 'L': [9.45], 'K': [7.81], 'M': [2.1], 'F': [3.91], 'P': [4.54], 'S': [4.18], 'T': [4.45], 'W': [0.9], 'Y': [3.46], 'V': [8.62]} ,
           'fuks010110' :  {'A': [9.07], 'R': [4.9], 'N': [4.05], 'D': [5.73], 'C': [0.95], 'Q': [3.63], 'E': [7.77], 'G': [7.69], 'H': [2.47], 'I': [6.56], 'L': [9.0], 'K': [6.01], 'M': [2.54], 'F': [3.59], 'P': [4.04], 'S': [5.15], 'T': [5.46], 'W': [0.95], 'Y': [2.96], 'V': [7.47]} ,
           'fuks010111' :  {'A': [8.82], 'R': [3.71], 'N': [6.77], 'D': [6.38], 'C': [0.9], 'Q': [3.89], 'E': [4.05], 'G': [9.11], 'H': [1.77], 'I': [5.05], 'L': [6.54], 'K': [5.45], 'M': [1.62], 'F': [3.51], 'P': [4.28], 'S': [7.64], 'T': [7.12], 'W': [1.96], 'Y': [4.85], 'V': [6.6]} ,
           'fuks010112' :  {'A': [6.65], 'R': [5.17], 'N': [4.4], 'D': [5.5], 'C': [1.79], 'Q': [4.52], 'E': [6.89], 'G': [5.72], 'H': [2.13], 'I': [5.47], 'L': [10.15], 'K': [7.59], 'M': [2.24], 'F': [4.34], 'P': [4.56], 'S': [6.52], 'T': [5.08], 'W': [1.24], 'Y': [3.01], 'V': [7.0]} ,
           'avbf000101' :  {'A': [0.0], 'R': [2.45], 'N': [0.0], 'D': [0.0], 'C': [0.0], 'Q': [1.25], 'E': [1.27], 'G': [0.0], 'H': [1.45], 'I': [0.0], 'L': [0.0], 'K': [3.67], 'M': [0.0], 'F': [0.0], 'P': [0.0], 'S': [0.0], 'T': [0.0], 'W': [6.93], 'Y': [5.06], 'V': [0.0]} ,
           'avbf000102' :  {'A': [89.3], 'R': [190.3], 'N': [122.4], 'D': [114.4], 'C': [102.5], 'Q': [146.9], 'E': [138.8], 'G': [63.8], 'H': [157.5], 'I': [163.0], 'L': [163.1], 'K': [165.1], 'M': [165.8], 'F': [190.8], 'P': [121.6], 'S': [94.2], 'T': [119.6], 'W': [226.4], 'Y': [194.6], 'V': [138.2]} ,
           'avbf000103' :  {'A': [90.0], 'R': [194.0], 'N': [124.7], 'D': [117.3], 'C': [103.3], 'Q': [149.4], 'E': [142.2], 'G': [64.9], 'H': [160.0], 'I': [163.9], 'L': [164.0], 'K': [167.3], 'M': [167.0], 'F': [191.9], 'P': [122.9], 'S': [95.4], 'T': [121.5], 'W': [228.2], 'Y': [197.0], 'V': [139.0]} ,
           'avbf000104' :  {'A': [0.0373], 'R': [0.0959], 'N': [0.0036], 'D': [0.1263], 'C': [0.0829], 'Q': [0.0761], 'E': [0.0058], 'G': [0.005], 'H': [0.0242], 'I': [0.0], 'L': [0.0], 'K': [0.0371], 'M': [0.0823], 'F': [0.0946], 'P': [0.0198], 'S': [0.0829], 'T': [0.0941], 'W': [0.0548], 'Y': [0.0516], 'V': [0.0057]} ,
           'avbf000105' :  {'A': [0.85], 'R': [0.2], 'N': [-0.48], 'D': [-1.1], 'C': [2.1], 'Q': [-0.42], 'E': [-0.79], 'G': [0.0], 'H': [0.22], 'I': [3.14], 'L': [1.99], 'K': [-1.19], 'M': [1.42], 'F': [1.69], 'P': [-1.14], 'S': [-0.52], 'T': [-0.08], 'W': [1.76], 'Y': [1.37], 'V': [2.53]} ,
           'avbf000106' :  {'A': [0.06], 'R': [-0.85], 'N': [0.25], 'D': [-0.2], 'C': [0.49], 'Q': [0.31], 'E': [-0.1], 'G': [0.21], 'H': [-2.24], 'I': [3.48], 'L': [3.5], 'K': [-1.62], 'M': [0.21], 'F': [4.8], 'P': [0.71], 'S': [-0.62], 'T': [0.65], 'W': [2.29], 'Y': [1.89], 'V': [1.59]} ,
           'avbf000107' :  {'A': [2.62], 'R': [1.26], 'N': [-1.27], 'D': [-2.84], 'C': [0.73], 'Q': [-1.69], 'E': [-0.45], 'G': [-1.15], 'H': [-0.74], 'I': [4.38], 'L': [6.57], 'K': [-2.78], 'M': [-3.12], 'F': [9.14], 'P': [-0.12], 'S': [-1.39], 'T': [1.81], 'W': [5.91], 'Y': [1.39], 'V': [2.3]} ,
           'avbf000108' :  {'A': [-1.64], 'R': [-3.28], 'N': [0.83], 'D': [0.7], 'C': [9.3], 'Q': [-0.04], 'E': [1.18], 'G': [-1.85], 'H': [7.17], 'I': [3.02], 'L': [0.83], 'K': [-2.36], 'M': [4.26], 'F': [-1.36], 'P': [3.12], 'S': [1.59], 'T': [2.31], 'W': [2.61], 'Y': [2.37], 'V': [0.52]} ,
           'avbf000109' :  {'A': [-2.34], 'R': [1.6], 'N': [2.81], 'D': [-0.48], 'C': [5.03], 'Q': [0.16], 'E': [1.3], 'G': [-1.06], 'H': [-3.0], 'I': [7.26], 'L': [1.09], 'K': [1.56], 'M': [0.62], 'F': [2.57], 'P': [-0.15], 'S': [1.93], 'T': [0.19], 'W': [3.59], 'Y': [-2.58], 'V': [2.06]} ,
           'yanj020101' :  {'A': [0.78], 'R': [1.58], 'N': [1.2], 'D': [1.35], 'C': [0.55], 'Q': [1.19], 'E': [1.45], 'G': [0.68], 'H': [0.99], 'I': [0.47], 'L': [0.56], 'K': [1.1], 'M': [0.66], 'F': [0.47], 'P': [0.69], 'S': [1.0], 'T': [1.05], 'W': [0.7], 'Y': [1.0], 'V': [0.51]} ,
           'mits020101' :  {'A': [25.0], 'R': [-7.0], 'N': [-7.0], 'D': [2.0], 'C': [32.0], 'Q': [0.0], 'E': [14.0], 'G': [-2.0], 'H': [-26.0], 'I': [91.0], 'L': [100.0], 'K': [-26.0], 'M': [68.0], 'F': [100.0], 'P': [25.0], 'S': [-2.0], 'T': [7.0], 'W': [109.0], 'Y': [56.0], 'V': [62.0]} ,
           'tsaj990101' :  {'A': [1.1], 'R': [-5.1], 'N': [-3.5], 'D': [-3.6], 'C': [2.5], 'Q': [-3.68], 'E': [-3.2], 'G': [-0.64], 'H': [-3.2], 'I': [4.5], 'L': [3.8], 'K': [-4.11], 'M': [1.9], 'F': [2.8], 'P': [-1.9], 'S': [-0.5], 'T': [-0.7], 'W': [-0.46], 'Y': [-1.3], 'V': [4.2]} ,
           'tsaj990102' :  {'A': [0.1366], 'R': [0.0363], 'N': [-0.0345], 'D': [-0.1233], 'C': [0.2745], 'Q': [0.0325], 'E': [-0.0484], 'G': [-0.0464], 'H': [0.0549], 'I': [0.4172], 'L': [0.4251], 'K': [-0.0101], 'M': [0.1747], 'F': [0.4076], 'P': [0.0019], 'S': [-0.0433], 'T': [0.0589], 'W': [0.2362], 'Y': [0.3167], 'V': [0.4084]} ,
           'cosi940101' :  {'A': [0.0728], 'R': [0.0394], 'N': [-0.039], 'D': [-0.0552], 'C': [0.3557], 'Q': [0.0126], 'E': [-0.0295], 'G': [-0.0589], 'H': [0.0874], 'I': [0.3805], 'L': [0.3819], 'K': [-0.0053], 'M': [0.1613], 'F': [0.4201], 'P': [-0.0492], 'S': [-0.0282], 'T': [0.0239], 'W': [0.4114], 'Y': [0.3113], 'V': [0.2947]} ,
           'ponp930101' :  {'A': [0.151], 'R': [-0.0103], 'N': [0.0381], 'D': [0.0047], 'C': [0.3222], 'Q': [0.0246], 'E': [-0.0639], 'G': [0.0248], 'H': [0.1335], 'I': [0.4238], 'L': [0.3926], 'K': [-0.0158], 'M': [0.216], 'F': [0.3455], 'P': [0.0844], 'S': [0.004], 'T': [0.1462], 'W': [0.2657], 'Y': [0.2998], 'V': [0.3997]} ,
           'wilm950101' :  {'A': [-0.058], 'R': [0.0], 'N': [0.027], 'D': [0.016], 'C': [0.447], 'Q': [-0.073], 'E': [-0.128], 'G': [0.331], 'H': [0.195], 'I': [0.06], 'L': [0.138], 'K': [-0.112], 'M': [0.275], 'F': [0.24], 'P': [-0.478], 'S': [-0.177], 'T': [-0.163], 'W': [0.564], 'Y': [0.322], 'V': [-0.052]} ,
           'wilm950102' :  {'A': [-0.17], 'R': [0.37], 'N': [0.18], 'D': [0.37], 'C': [-0.06], 'Q': [0.26], 'E': [0.15], 'G': [0.01], 'H': [-0.02], 'I': [-0.28], 'L': [-0.28], 'K': [0.32], 'M': [-0.26], 'F': [-0.41], 'P': [0.13], 'S': [0.05], 'T': [0.02], 'W': [-0.15], 'Y': [-0.09], 'V': [-0.17]} ,
           'wilm950103' :  {'A': [-0.15], 'R': [0.32], 'N': [0.22], 'D': [0.41], 'C': [-0.15], 'Q': [0.03], 'E': [0.3], 'G': [0.08], 'H': [0.06], 'I': [-0.29], 'L': [-0.36], 'K': [0.24], 'M': [-0.19], 'F': [-0.22], 'P': [0.15], 'S': [0.16], 'T': [-0.08], 'W': [-0.28], 'Y': [-0.03], 'V': [-0.24]} ,
           'wilm950104' :  {'A': [0.964], 'R': [1.143], 'N': [0.944], 'D': [0.916], 'C': [0.778], 'Q': [1.047], 'E': [1.051], 'G': [0.835], 'H': [1.014], 'I': [0.922], 'L': [1.085], 'K': [0.944], 'M': [1.032], 'F': [1.119], 'P': [1.299], 'S': [0.947], 'T': [1.017], 'W': [0.895], 'Y': [1.0], 'V': [0.955]} ,
           'kuhl950101' :  {'A': [0.974], 'R': [1.129], 'N': [0.988], 'D': [0.892], 'C': [0.972], 'Q': [1.092], 'E': [1.054], 'G': [0.845], 'H': [0.949], 'I': [0.928], 'L': [1.11], 'K': [0.946], 'M': [0.923], 'F': [1.122], 'P': [1.362], 'S': [0.932], 'T': [1.023], 'W': [0.879], 'Y': [0.902], 'V': [0.923]} ,
           'guod860101' :  {'A': [0.938], 'R': [1.137], 'N': [0.902], 'D': [0.857], 'C': [0.6856], 'Q': [0.916], 'E': [1.139], 'G': [0.892], 'H': [1.109], 'I': [0.986], 'L': [1.0], 'K': [0.952], 'M': [1.077], 'F': [1.11], 'P': [1.266], 'S': [0.956], 'T': [1.018], 'W': [0.971], 'Y': [1.157], 'V': [0.959]} ,
           'jurd980101' :  {'A': [1.042], 'R': [1.069], 'N': [0.828], 'D': [0.97], 'C': [0.5], 'Q': [1.111], 'E': [0.992], 'G': [0.743], 'H': [1.034], 'I': [0.852], 'L': [1.193], 'K': [0.979], 'M': [0.998], 'F': [0.981], 'P': [1.332], 'S': [0.984], 'T': [0.992], 'W': [0.96], 'Y': [1.12], 'V': [1.001]} ,
           'basu050101' :  {'A': [1.065], 'R': [1.131], 'N': [0.762], 'D': [0.836], 'C': [1.015], 'Q': [0.861], 'E': [0.736], 'G': [1.022], 'H': [0.973], 'I': [1.189], 'L': [1.192], 'K': [0.478], 'M': [1.369], 'F': [1.368], 'P': [1.241], 'S': [1.097], 'T': [0.822], 'W': [1.017], 'Y': [0.836], 'V': [1.14]} ,
           'basu050102' :  {'A': [0.99], 'R': [1.132], 'N': [0.873], 'D': [0.915], 'C': [0.644], 'Q': [0.999], 'E': [1.053], 'G': [0.785], 'H': [1.054], 'I': [0.95], 'L': [1.106], 'K': [1.003], 'M': [1.093], 'F': [1.121], 'P': [1.314], 'S': [0.911], 'T': [0.988], 'W': [0.939], 'Y': [1.09], 'V': [0.957]} ,
           'basu050103' :  {'A': [0.892], 'R': [1.154], 'N': [1.144], 'D': [0.925], 'C': [1.035], 'Q': [1.2], 'E': [1.115], 'G': [0.917], 'H': [0.992], 'I': [0.817], 'L': [0.994], 'K': [0.944], 'M': [0.782], 'F': [1.058], 'P': [1.309], 'S': [0.986], 'T': [1.11], 'W': [0.841], 'Y': [0.866], 'V': [0.9]} ,
           'suym030101' :  {'A': [1.092], 'R': [1.239], 'N': [0.927], 'D': [0.919], 'C': [0.662], 'Q': [1.124], 'E': [1.199], 'G': [0.698], 'H': [1.012], 'I': [0.912], 'L': [1.276], 'K': [1.008], 'M': [1.171], 'F': [1.09], 'P': [0.8], 'S': [0.886], 'T': [0.832], 'W': [0.981], 'Y': [1.075], 'V': [0.908]} ,
           'punt030101' :  {'A': [0.843], 'R': [1.038], 'N': [0.956], 'D': [0.906], 'C': [0.896], 'Q': [0.968], 'E': [0.9], 'G': [0.978], 'H': [1.05], 'I': [0.946], 'L': [0.885], 'K': [0.893], 'M': [0.878], 'F': [1.151], 'P': [1.816], 'S': [1.003], 'T': [1.189], 'W': [0.852], 'Y': [0.945], 'V': [0.999]} ,
           'punt030102' :  {'A': [2.18], 'R': [2.71], 'N': [1.85], 'D': [1.75], 'C': [3.89], 'Q': [2.16], 'E': [1.89], 'G': [1.17], 'H': [2.51], 'I': [4.5], 'L': [4.71], 'K': [2.12], 'M': [3.63], 'F': [5.88], 'P': [2.09], 'S': [1.66], 'T': [2.18], 'W': [6.46], 'Y': [5.01], 'V': [3.77]} ,
           'geor030101' :  {'A': [1.79], 'R': [3.2], 'N': [2.83], 'D': [2.33], 'C': [2.22], 'Q': [2.37], 'E': [2.52], 'G': [0.7], 'H': [3.06], 'I': [4.59], 'L': [4.72], 'K': [2.5], 'M': [3.91], 'F': [4.84], 'P': [2.45], 'S': [1.82], 'T': [2.45], 'W': [5.64], 'Y': [4.46], 'V': [3.67]} ,
           'geor030102' :  {'A': [13.4], 'R': [8.5], 'N': [7.6], 'D': [8.2], 'C': [22.6], 'Q': [8.5], 'E': [7.3], 'G': [7.0], 'H': [11.3], 'I': [20.3], 'L': [20.8], 'K': [6.1], 'M': [15.7], 'F': [23.9], 'P': [9.9], 'S': [8.2], 'T': [10.3], 'W': [24.5], 'Y': [19.5], 'V': [19.5]} ,
           'geor030103' :  {'A': [0.0166], 'R': [-0.0762], 'N': [-0.0786], 'D': [-0.1278], 'C': [0.5724], 'Q': [-0.1051], 'E': [-0.1794], 'G': [-0.0442], 'H': [0.1643], 'I': [0.2758], 'L': [0.2523], 'K': [-0.2134], 'M': [0.0197], 'F': [0.3561], 'P': [-0.4188], 'S': [-0.1629], 'T': [-0.0701], 'W': [0.3836], 'Y': [0.25], 'V': [0.1782]} ,
           'geor030104' :  {'A': [90.1], 'R': [192.8], 'N': [127.5], 'D': [117.1], 'C': [113.2], 'Q': [149.4], 'E': [140.8], 'G': [63.8], 'H': [159.3], 'I': [164.9], 'L': [164.6], 'K': [170.0], 'M': [167.7], 'F': [193.5], 'P': [123.1], 'S': [94.2], 'T': [120.0], 'W': [197.1], 'Y': [231.7], 'V': [139.1]} ,
           'geor030105' :  {'A': [91.5], 'R': [196.1], 'N': [138.3], 'D': [135.2], 'C': [114.4], 'Q': [156.4], 'E': [154.6], 'G': [67.5], 'H': [163.2], 'I': [162.6], 'L': [163.4], 'K': [162.5], 'M': [165.9], 'F': [198.8], 'P': [123.4], 'S': [102.0], 'T': [126.0], 'W': [209.8], 'Y': [237.2], 'V': [138.4]} ,
           'geor030106' :  {'A': [1.076], 'R': [1.361], 'N': [1.056], 'D': [1.29], 'C': [0.753], 'Q': [0.729], 'E': [1.118], 'G': [1.346], 'H': [0.985], 'I': [0.926], 'L': [1.054], 'K': [1.105], 'M': [0.974], 'F': [0.869], 'P': [0.82], 'S': [1.342], 'T': [0.871], 'W': [0.666], 'Y': [0.531], 'V': [1.131]} ,
           'geor030107' :  {'A': [1.12], 'R': [-2.55], 'N': [-0.83], 'D': [-0.83], 'C': [0.59], 'Q': [-0.78], 'E': [-0.92], 'G': [1.2], 'H': [-0.93], 'I': [1.16], 'L': [1.18], 'K': [-0.8], 'M': [0.55], 'F': [0.67], 'P': [0.54], 'S': [-0.05], 'T': [-0.02], 'W': [-0.19], 'Y': [-0.23], 'V': [1.13]} ,
           'geor030108' :  {'A': [1.38], 'R': [0.0], 'N': [0.37], 'D': [0.52], 'C': [1.43], 'Q': [0.22], 'E': [0.71], 'G': [1.34], 'H': [0.66], 'I': [2.32], 'L': [1.47], 'K': [0.15], 'M': [1.78], 'F': [1.72], 'P': [0.85], 'S': [0.86], 'T': [0.89], 'W': [0.82], 'Y': [0.47], 'V': [1.99]} ,
           'geor030109' :  {'A': [-0.27], 'R': [1.87], 'N': [0.81], 'D': [0.81], 'C': [-1.05], 'Q': [1.1], 'E': [1.17], 'G': [-0.16], 'H': [0.28], 'I': [-0.77], 'L': [-1.1], 'K': [1.7], 'M': [-0.73], 'F': [-1.43], 'P': [-0.75], 'S': [0.42], 'T': [0.63], 'W': [-1.57], 'Y': [-0.56], 'V': [-0.4]} ,
           'zhoh040101' :  {'A': [0.05], 'R': [0.12], 'N': [0.29], 'D': [0.41], 'C': [-0.84], 'Q': [0.46], 'E': [0.38], 'G': [0.31], 'H': [-0.41], 'I': [-0.69], 'L': [-0.62], 'K': [0.57], 'M': [-0.38], 'F': [-0.45], 'P': [0.46], 'S': [0.12], 'T': [0.38], 'W': [-0.98], 'Y': [-0.25], 'V': [-0.46]} ,
           'zhoh040102' :  {'A': [-0.31], 'R': [1.3], 'N': [0.49], 'D': [0.58], 'C': [-0.87], 'Q': [0.7], 'E': [0.68], 'G': [-0.33], 'H': [0.13], 'I': [-0.66], 'L': [-0.53], 'K': [1.79], 'M': [-0.38], 'F': [-0.45], 'P': [0.34], 'S': [0.1], 'T': [0.21], 'W': [-0.27], 'Y': [0.4], 'V': [-0.62]} ,
           'zhoh040103' :  {'A': [-0.27], 'R': [2.0], 'N': [0.61], 'D': [0.5], 'C': [-0.23], 'Q': [1.0], 'E': [0.33], 'G': [-0.22], 'H': [0.37], 'I': [-0.8], 'L': [-0.44], 'K': [1.17], 'M': [-0.31], 'F': [-0.55], 'P': [0.36], 'S': [0.17], 'T': [0.18], 'W': [0.05], 'Y': [0.48], 'V': [-0.65]} ,
           'baek050101' :  {'A': [0.18], 'R': [-5.4], 'N': [-1.3], 'D': [-2.36], 'C': [0.27], 'Q': [-1.22], 'E': [-2.1], 'G': [0.09], 'H': [-1.48], 'I': [0.37], 'L': [0.41], 'K': [-2.53], 'M': [0.44], 'F': [0.5], 'P': [-0.2], 'S': [-0.4], 'T': [-0.34], 'W': [-0.01], 'Y': [-0.08], 'V': [0.32]} ,
           'hary940101' :  {'A': [0.42], 'R': [-1.56], 'N': [-1.03], 'D': [-0.51], 'C': [0.84], 'Q': [-0.96], 'E': [-0.37], 'G': [0.0], 'H': [-2.28], 'I': [1.81], 'L': [1.8], 'K': [-2.03], 'M': [1.18], 'F': [1.74], 'P': [0.86], 'S': [-0.64], 'T': [-0.26], 'W': [1.46], 'Y': [0.51], 'V': [1.34]} ,
           'ponj960101' :  {'A': [0.616], 'R': [0.0], 'N': [0.236], 'D': [0.028], 'C': [0.68], 'Q': [0.251], 'E': [0.043], 'G': [0.501], 'H': [0.165], 'I': [0.943], 'L': [0.943], 'K': [0.283], 'M': [0.738], 'F': [1.0], 'P': [0.711], 'S': [0.359], 'T': [0.45], 'W': [0.878], 'Y': [0.88], 'V': [0.825]} ,
           'digm050101' :  {'A': [0.2], 'R': [-0.7], 'N': [-0.5], 'D': [-1.4], 'C': [1.9], 'Q': [-1.1], 'E': [-1.3], 'G': [-0.1], 'H': [0.4], 'I': [1.4], 'L': [0.5], 'K': [-1.6], 'M': [0.5], 'F': [1.0], 'P': [-1.0], 'S': [-0.7], 'T': [-0.4], 'W': [1.6], 'Y': [0.5], 'V': [0.7]} ,
           'wolr790101' :  {'A': [50.76], 'R': [48.66], 'N': [45.8], 'D': [43.17], 'C': [58.74], 'Q': [46.09], 'E': [43.48], 'G': [50.27], 'H': [49.33], 'I': [57.3], 'L': [53.89], 'K': [42.92], 'M': [52.75], 'F': [53.45], 'P': [45.39], 'S': [47.24], 'T': [49.26], 'W': [53.59], 'Y': [51.79], 'V': [56.12]} ,
           'olsk800101' :  {'A': [-0.414], 'R': [-0.584], 'N': [-0.916], 'D': [-1.31], 'C': [0.162], 'Q': [-0.905], 'E': [-1.218], 'G': [-0.684], 'H': [-0.63], 'I': [1.237], 'L': [1.215], 'K': [-0.67], 'M': [1.02], 'F': [1.938], 'P': [-0.503], 'S': [-0.563], 'T': [-0.289], 'W': [0.514], 'Y': [1.699], 'V': [0.899]} ,
           'kida850101' :  {'A': [-0.96], 'R': [0.75], 'N': [-1.94], 'D': [-5.68], 'C': [4.54], 'Q': [-5.3], 'E': [-3.86], 'G': [-1.28], 'H': [-0.62], 'I': [5.54], 'L': [6.81], 'K': [-5.62], 'M': [4.76], 'F': [5.06], 'P': [-4.47], 'S': [-1.92], 'T': [-3.99], 'W': [0.21], 'Y': [3.34], 'V': [5.39]} ,
           'guyh850102' :  {'A': [-0.26], 'R': [0.08], 'N': [-0.46], 'D': [-1.3], 'C': [0.83], 'Q': [-0.83], 'E': [-0.73], 'G': [-0.4], 'H': [-0.18], 'I': [1.1], 'L': [1.52], 'K': [-1.01], 'M': [1.09], 'F': [1.09], 'P': [-0.62], 'S': [-0.55], 'T': [-0.71], 'W': [-0.13], 'Y': [0.69], 'V': [1.15]} ,
           'guyh850103' :  {'A': [-0.73], 'R': [-1.03], 'N': [-5.29], 'D': [-6.13], 'C': [0.64], 'Q': [-0.96], 'E': [-2.9], 'G': [-2.67], 'H': [3.03], 'I': [5.04], 'L': [4.91], 'K': [-5.99], 'M': [3.34], 'F': [5.2], 'P': [-4.32], 'S': [-3.0], 'T': [-1.91], 'W': [0.51], 'Y': [2.87], 'V': [3.98]} ,
           'guyh850104' :  {'A': [-1.35], 'R': [-3.89], 'N': [-10.96], 'D': [-11.88], 'C': [4.37], 'Q': [-1.34], 'E': [-4.56], 'G': [-5.82], 'H': [6.54], 'I': [10.93], 'L': [9.88], 'K': [-11.92], 'M': [7.47], 'F': [11.35], 'P': [-10.86], 'S': [-6.21], 'T': [-4.83], 'W': [1.8], 'Y': [7.61], 'V': [8.2]} ,
           'guyh850105' :  {'A': [-0.56], 'R': [-0.26], 'N': [-2.87], 'D': [-4.31], 'C': [1.78], 'Q': [-2.31], 'E': [-2.35], 'G': [-1.35], 'H': [0.81], 'I': [3.83], 'L': [4.09], 'K': [-4.08], 'M': [3.11], 'F': [3.67], 'P': [-3.22], 'S': [-1.85], 'T': [-1.97], 'W': [-0.11], 'Y': [2.17], 'V': [3.31]} ,
           'rosm880104' :  {'A': [1.37], 'R': [1.33], 'N': [6.29], 'D': [8.93], 'C': [-4.47], 'Q': [3.88], 'E': [4.04], 'G': [3.39], 'H': [-1.65], 'I': [-7.92], 'L': [-8.68], 'K': [7.7], 'M': [-7.13], 'F': [-7.96], 'P': [6.25], 'S': [4.08], 'T': [4.02], 'W': [0.79], 'Y': [-4.73], 'V': [-6.94]} ,
           'rosm880105' :  {'A': [-0.02], 'R': [0.44], 'N': [0.63], 'D': [0.72], 'C': [-0.96], 'Q': [0.56], 'E': [0.74], 'G': [0.38], 'H': [0.0], 'I': [-1.89], 'L': [-2.29], 'K': [1.01], 'M': [-1.36], 'F': [-2.22], 'P': [0.47], 'S': [0.55], 'T': [0.25], 'W': [-1.28], 'Y': [-0.88], 'V': [-1.34]} ,
           'jacr890101' :  {'A': [0.0], 'R': [0.07], 'N': [0.1], 'D': [0.12], 'C': [-0.16], 'Q': [0.09], 'E': [0.12], 'G': [0.06], 'H': [0.0], 'I': [-0.31], 'L': [-0.37], 'K': [0.17], 'M': [-0.22], 'F': [-0.36], 'P': [0.08], 'S': [0.09], 'T': [0.04], 'W': [-0.21], 'Y': [-0.14], 'V': [-0.22]} ,
           'cowr900101' :  {'A': [-0.03], 'R': [0.09], 'N': [0.13], 'D': [0.17], 'C': [-0.36], 'Q': [0.13], 'E': [0.23], 'G': [0.09], 'H': [-0.04], 'I': [-0.33], 'L': [-0.38], 'K': [0.32], 'M': [-0.3], 'F': [-0.34], 'P': [0.2], 'S': [0.1], 'T': [0.01], 'W': [-0.24], 'Y': [-0.23], 'V': [-0.29]} ,
           'blas910101' :  {'A': [-0.04], 'R': [0.07], 'N': [0.13], 'D': [0.19], 'C': [-0.38], 'Q': [0.14], 'E': [0.23], 'G': [0.09], 'H': [-0.04], 'I': [-0.34], 'L': [-0.37], 'K': [0.33], 'M': [-0.3], 'F': [-0.38], 'P': [0.19], 'S': [0.12], 'T': [0.03], 'W': [-0.33], 'Y': [-0.29], 'V': [-0.29]} ,
           'casg920101' :  {'A': [-0.02], 'R': [0.08], 'N': [0.1], 'D': [0.19], 'C': [-0.32], 'Q': [0.15], 'E': [0.21], 'G': [-0.02], 'H': [-0.02], 'I': [-0.28], 'L': [-0.32], 'K': [0.3], 'M': [-0.25], 'F': [-0.33], 'P': [0.11], 'S': [0.11], 'T': [0.05], 'W': [-0.27], 'Y': [-0.23], 'V': [-0.23]} ,
           'corj870101' :  {'A': [-1.6], 'R': [12.3], 'N': [4.8], 'D': [9.2], 'C': [-2.0], 'Q': [4.1], 'E': [8.2], 'G': [-1.0], 'H': [3.0], 'I': [-3.1], 'L': [-2.8], 'K': [8.8], 'M': [-3.4], 'F': [-3.7], 'P': [0.2], 'S': [-0.6], 'T': [-1.2], 'W': [-1.9], 'Y': [0.7], 'V': [-2.6]} ,
           'corj870102' :  {'A': [-0.21], 'R': [2.11], 'N': [0.96], 'D': [1.36], 'C': [-6.04], 'Q': [1.52], 'E': [2.3], 'G': [0.0], 'H': [-1.23], 'I': [-4.81], 'L': [-4.68], 'K': [3.88], 'M': [-3.66], 'F': [-4.65], 'P': [0.75], 'S': [1.74], 'T': [0.78], 'W': [-3.32], 'Y': [-1.01], 'V': [-3.5]} ,
           'corj870103' :  {'A': [2.0], 'R': [8.0], 'N': [5.0], 'D': [5.0], 'C': [3.0], 'Q': [6.0], 'E': [6.0], 'G': [1.0], 'H': [7.0], 'I': [5.0], 'L': [5.0], 'K': [6.0], 'M': [5.0], 'F': [8.0], 'P': [4.0], 'S': [3.0], 'T': [4.0], 'W': [11.0], 'Y': [9.0], 'V': [4.0]} ,
           'corj870104' :  {'A': [1.0], 'R': [7.0], 'N': [4.0], 'D': [4.0], 'C': [2.0], 'Q': [5.0], 'E': [5.0], 'G': [0.0], 'H': [6.0], 'I': [4.0], 'L': [4.0], 'K': [5.0], 'M': [4.0], 'F': [8.0], 'P': [4.0], 'S': [2.0], 'T': [3.0], 'W': [12.0], 'Y': [9.0], 'V': [3.0]} ,
           
           
           }
    if scalename == 'all':
        d = {'I': [], 'F': [], 'V': [], 'L': [], 'W': [], 'M': [], 'A': [], 'G': [], 'C': [], 'Y': [], 'P': [],
             'T': [], 'S': [], 'H': [], 'E': [], 'N': [], 'Q': [], 'D': [], 'K': [], 'R': []}
        for scale in scales.keys():
            for k, v in scales[scale].items():
                d[k].extend(v)
        return 'all', d

    elif scalename == 'instability':
        d = {
            "A": {"A": 1.0, "C": 44.94, "E": 1.0, "D": -7.49, "G": 1.0, "F": 1.0, "I": 1.0, "H": -7.49, "K": 1.0,
                  "M": 1.0, "L": 1.0, "N": 1.0, "Q": 1.0, "P": 20.26, "S": 1.0, "R": 1.0, "T": 1.0, "W": 1.0, "V": 1.0,
                  "Y": 1.0},
            "C": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 20.26, "G": 1.0, "F": 1.0, "I": 1.0, "H": 33.6, "K": 1.0,
                  "M": 33.6, "L": 20.26, "N": 1.0, "Q": -6.54, "P": 20.26, "S": 1.0, "R": 1.0, "T": 33.6, "W": 24.68,
                  "V": -6.54, "Y": 1.0},
            "E": {"A": 1.0, "C": 44.94, "E": 33.6, "D": 20.26, "G": 1.0, "F": 1.0, "I": 20.26, "H": -6.54, "K": 1.0,
                  "M": 1.0, "L": 1.0, "N": 1.0, "Q": 20.26, "P": 20.26, "S": 20.26, "R": 1.0, "T": 1.0, "W": -14.03,
                  "V": 1.0, "Y": 1.0},
            "D": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 1.0, "G": 1.0, "F": -6.54, "I": 1.0, "H": 1.0, "K": -7.49,
                  "M": 1.0, "L": 1.0, "N": 1.0, "Q": 1.0, "P": 1.0, "S": 20.26, "R": -6.54, "T": -14.03, "W": 1.0,
                  "V": 1.0, "Y": 1.0},
            "G": {"A": -7.49, "C": 1.0, "E": -6.54, "D": 1.0, "G": 13.34, "F": 1.0, "I": -7.49, "H": 1.0, "K": -7.49,
                  "M": 1.0, "L": 1.0, "N": -7.49, "Q": 1.0, "P": 1.0, "S": 1.0, "R": 1.0, "T": -7.49, "W": 13.34,
                  "V": 1.0, "Y": -7.49},
            "F": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 13.34, "G": 1.0, "F": 1.0, "I": 1.0, "H": 1.0, "K": -14.03,
                  "M": 1.0, "L": 1.0, "N": 1.0, "Q": 1.0, "P": 20.26, "S": 1.0, "R": 1.0, "T": 1.0, "W": 1.0, "V": 1.0,
                  "Y": 33.601},
            "I": {"A": 1.0, "C": 1.0, "E": 44.94, "D": 1.0, "G": 1.0, "F": 1.0, "I": 1.0, "H": 13.34, "K": -7.49,
                  "M": 1.0, "L": 20.26, "N": 1.0, "Q": 1.0, "P": -1.88, "S": 1.0, "R": 1.0, "T": 1.0, "W": 1.0,
                  "V": -7.49, "Y": 1.0},
            "H": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 1.0, "G": -9.37, "F": -9.37, "I": 44.94, "H": 1.0, "K": 24.68,
                  "M": 1.0, "L": 1.0, "N": 24.68, "Q": 1.0, "P": -1.88, "S": 1.0, "R": 1.0, "T": -6.54, "W": -1.88,
                  "V": 1.0, "Y": 44.94},
            "K": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 1.0, "G": -7.49, "F": 1.0, "I": -7.49, "H": 1.0, "K": 1.0,
                  "M": 33.6, "L": -7.49, "N": 1.0, "Q": 24.64, "P": -6.54, "S": 1.0, "R": 33.6, "T": 1.0, "W": 1.0,
                  "V": -7.49, "Y": 1.0},
            "M": {"A": 13.34, "C": 1.0, "E": 1.0, "D": 1.0, "G": 1.0, "F": 1.0, "I": 1.0, "H": 58.28, "K": 1.0,
                  "M": -1.88, "L": 1.0, "N": 1.0, "Q": -6.54, "P": 44.94, "S": 44.94, "R": -6.54, "T": -1.88, "W": 1.0,
                  "V": 1.0, "Y": 24.68},
            "L": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 1.0, "G": 1.0, "F": 1.0, "I": 1.0, "H": 1.0, "K": -7.49, "M": 1.0,
                  "L": 1.0, "N": 1.0, "Q": 33.6, "P": 20.26, "S": 1.0, "R": 20.26, "T": 1.0, "W": 24.68, "V": 1.0,
                  "Y": 1.0},
            "N": {"A": 1.0, "C": -1.88, "E": 1.0, "D": 1.0, "G": -14.03, "F": -14.03, "I": 44.94, "H": 1.0, "K": 24.68,
                  "M": 1.0, "L": 1.0, "N": 1.0, "Q": -6.54, "P": -1.88, "S": 1.0, "R": 1.0, "T": -7.49, "W": -9.37,
                  "V": 1.0, "Y": 1.0},
            "Q": {"A": 1.0, "C": -6.54, "E": 20.26, "D": 20.26, "G": 1.0, "F": -6.54, "I": 1.0, "H": 1.0, "K": 1.0,
                  "M": 1.0, "L": 1.0, "N": 1.0, "Q": 20.26, "P": 20.26, "S": 44.94, "R": 1.0, "T": 1.0, "W": 1.0,
                  "V": -6.54, "Y": -6.54},
            "P": {"A": 20.26, "C": -6.54, "E": 18.38, "D": -6.54, "G": 1.0, "F": 20.26, "I": 1.0, "H": 1.0, "K": 1.0,
                  "M": -6.54, "L": 1.0, "N": 1.0, "Q": 20.26, "P": 20.26, "S": 20.26, "R": -6.54, "T": 1.0, "W": -1.88,
                  "V": 20.26, "Y": 1.0},
            "S": {"A": 1.0, "C": 33.6, "E": 20.26, "D": 1.0, "G": 1.0, "F": 1.0, "I": 1.0, "H": 1.0, "K": 1.0, "M": 1.0,
                  "L": 1.0, "N": 1.0, "Q": 20.26, "P": 44.94, "S": 20.26, "R": 20.26, "T": 1.0, "W": 1.0, "V": 1.0,
                  "Y": 1.0},
            "R": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 1.0, "G": -7.49, "F": 1.0, "I": 1.0, "H": 20.26, "K": 1.0,
                  "M": 1.0, "L": 1.0, "N": 13.34, "Q": 20.26, "P": 20.26, "S": 44.94, "R": 58.28, "T": 1.0, "W": 58.28,
                  "V": 1.0, "Y": -6.54},
            "T": {"A": 1.0, "C": 1.0, "E": 20.26, "D": 1.0, "G": -7.49, "F": 13.34, "I": 1.0, "H": 1.0, "K": 1.0,
                  "M": 1.0, "L": 1.0, "N": -14.03, "Q": -6.54, "P": 1.0, "S": 1.0, "R": 1.0, "T": 1.0, "W": -14.03,
                  "V": 1.0, "Y": 1.0},
            "W": {"A": -14.03, "C": 1.0, "E": 1.0, "D": 1.0, "G": -9.37, "F": 1.0, "I": 1.0, "H": 24.68, "K": 1.0,
                  "M": 24.68, "L": 13.34, "N": 13.34, "Q": 1.0, "P": 1.0, "S": 1.0, "R": 1.0, "T": -14.03, "W": 1.0,
                  "V": -7.49, "Y": 1.0},
            "V": {"A": 1.0, "C": 1.0, "E": 1.0, "D": -14.03, "G": -7.49, "F": 1.0, "I": 1.0, "H": 1.0, "K": -1.88,
                  "M": 1.0, "L": 1.0, "N": 1.0, "Q": 1.0, "P": 20.26, "S": 1.0, "R": 1.0, "T": -7.49, "W": 1.0,
                  "V": 1.0, "Y": -6.54},
            "Y": {"A": 24.68, "C": 1.0, "E": -6.54, "D": 24.68, "G": -7.49, "F": 1.0, "I": 1.0, "H": 13.34, "K": 1.0,
                  "M": 44.94, "L": 1.0, "N": 1.0, "Q": 1.0, "P": 13.34, "S": 1.0, "R": -15.91, "T": -7.49, "W": -9.37,
                  "V": 1.0, "Y": 13.34}}
        return 'instability', d

    else:
        return scalename, scales[scalename]


def read_fasta(inputfile):
    """Method for loading sequences from a FASTA formatted file into :py:attr:`sequences` & :py:attr:`names`.
    This method is used by the base class :class:`modlamp.descriptors.PeptideDescriptor` if the input is a FASTA file.

    :param inputfile: .fasta file with sequences and headers to read
    :return: list of sequences in the attribute :py:attr:`sequences` with corresponding sequence names in
        :py:attr:`names`.
    """
    names = list()  # list for storing names
    sequences = list()  # list for storing sequences
    seq = str()
    with open(inputfile) as f:
        all = f.readlines()
        last = all[-1]
        for line in all:
            if line.startswith('>'):
                names.append(line.split(' ')[0][1:].strip())  # add FASTA name without description as molecule name
                sequences.append(seq.strip())
                seq = str()
            elif line == last:
                seq += line.strip()  # remove potential white space
                sequences.append(seq.strip())
            else:
                seq += line.strip()  # remove potential white space
    return sequences[1:], names


def save_fasta(filename, sequences, names=None):
    """Method for saving sequences in the instance :py:attr:`sequences` to a file in FASTA format.

    :param filename: {str} output filename (ending .fasta)
    :param sequences: {list} sequences to be saved to file
    :param names: {list} whether sequence names from self.names should be saved as sequence identifiers
    :return: a FASTA formatted file containing the generated sequences
    """
    if os.path.exists(filename):
        os.remove(filename)  # remove outputfile, it it exists

    with open(filename, 'w') as o:
        for n, seq in enumerate(sequences):
            if names:
                o.write('>' + str(names[n]) + '\n')
            else:
                o.write('>Seq_' + str(n) + '\n')
            o.write(seq + '\n')


def aa_weights():
    """Function holding molecular weight data on all natural amino acids.

    :return: dictionary with amino acid letters and corresponding weights

    .. versionadded:: v2.4.1
    """
    weights = {'A': 89.093, 'C': 121.158, 'D': 133.103, 'E': 147.129, 'F': 165.189, 'G': 75.067,
               'H': 155.155, 'I': 131.173, 'K': 146.188, 'L': 131.173, 'M': 149.211, 'N': 132.118,
               'P': 115.131, 'Q': 146.145, 'R': 174.20, 'S': 105.093, 'T': 119.119, 'V': 117.146,
               'W': 204.225, 'Y': 181.189}
    return weights


def count_aas(seq, scale='relative'):
    """Function to count the amino acids occuring in a given sequence.

    :param seq: {str} amino acid sequence
    :param scale: {'absolute' or 'relative'} defines whether counts or frequencies are given for each AA
    :return: {dict} dictionary with amino acids as keys and their counts in the sequence as values.
    """
    if seq == '':  # error if len(seq) == 0
        seq = ' '
    aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    scl = 1.
    if scale == 'relative':
        scl = len(seq)
    aa = {a: (float(seq.count(a)) / scl) for a in aas}
    aa = collections.OrderedDict(sorted(list(aa.items())))
    return aa


def count_ngrams(seq, n):
    """Function to count the n-grams of an amino acid sequence. N can be one integer or a list of integers

    :param seq: {str} amino acid sequence
    :param n: {int or list of ints} defines whether counts or frequencies are given for each AA
    :return: {dict} dictionary with n-grams as keys and their counts in the sequence as values.
    """
    if seq == '':
        seq = ' '
    if isinstance(n, int):
        n = [n]
    ngrams = list()
    for i in n:
        ngrams.extend([seq[j:j+i] for j in range(len(seq) - (i-1))])
    counts = {g: (seq.count(g)) for g in set(ngrams)}
    counts = collections.OrderedDict(sorted(counts.items(), key=operator.itemgetter(1), reverse=True))
    return counts


def aa_energies():
    """Function holding free energies of transfer between cyclohexane and water for all natural amino acids.
    H. G. Boman, D. Wade, I. a Boman, B. Wåhlin, R. B. Merrifield, *FEBS Lett*. **1989**, *259*, 103–106.

    :return: dictionary with amino acid letters and corresponding energies.
    """
    energies = {'L': -4.92, 'I': -4.92, 'V': -4.04, 'F': -2.98, 'M': -2.35, 'W': -2.33, 'A': -1.81, 'C': -1.28,
                'G': -0.94, 'Y': 0.14, 'T': 2.57, 'S': 3.40, 'H': 4.66, 'Q': 5.54, 'K': 5.55, 'N': 6.64, 'E': 6.81,
                'D': 8.72, 'R': 14.92, 'P': 0.}
    return energies


def ngrams_apd():
    """Function returning the most frequent 2-, 3- and 4-grams from all sequences in the `APD3
    <http://aps.unmc.edu/AP/>`_, version August 2016 with 2727 sequences.
    For all 2, 3 and 4grams, all possible ngrams were generated from all sequences and the top 50 most frequent
    assembled into a list. Finally, leading and tailing spaces were striped and duplicates as well as ngrams containing
    spaces were removed.

    :return: numpy.array containing most frequent ngrams
    """
    return np.array(['AGK', 'CKI', 'RR', 'YGGG', 'LSGL', 'RG', 'YGGY', 'PRP', 'LGGG',
                     'GV', 'GT', 'GS', 'GR', 'IAG', 'GG', 'GF', 'GC', 'GGYG', 'GA', 'GL',
                     'GK', 'GI', 'IPC', 'KAA', 'LAK', 'GLGG', 'GGLG', 'CKIT', 'GAGK',
                     'LLSG', 'LKK', 'FLP', 'LSG', 'SCK', 'LLS', 'GETC', 'VLG', 'GKLL',
                     'LLG', 'C', 'KCKI', 'G', 'VGK', 'CSC', 'TKKC', 'GCS', 'GKA', 'IGK',
                     'GESC', 'KVCY', 'KKL', 'KKI', 'KKC', 'LGGL', 'GLL', 'CGE', 'GGYC',
                     'GLLS', 'GLF', 'AKK', 'GKAA', 'ESCV', 'GLP', 'CGES', 'PCGE', 'FL',
                     'CGET', 'GLW', 'KGAA', 'KAAL', 'GGY', 'GGG', 'IKG', 'LKG', 'GGL',
                     'CK', 'GTC', 'CG', 'SKKC', 'CS', 'CR', 'KC', 'AGKA', 'KA', 'KG',
                     'LKCK', 'SCKL', 'KK', 'KI', 'KN', 'KL', 'SK', 'KV', 'SL', 'SC',
                     'SG', 'AAA', 'VAK', 'AAL', 'AAK', 'GGGG', 'KNVA', 'GGGL', 'GYG',
                     'LG', 'LA', 'LL', 'LK', 'LS', 'LP', 'GCSC', 'TC', 'GAA', 'AA', 'VA',
                     'VC', 'AG', 'VG', 'AI', 'AK', 'VL', 'AL', 'TPGC', 'IK', 'IA', 'IG',
                     'YGG', 'LGK', 'CSCK', 'GYGG', 'LGG', 'KGA'])


def aa_formulas():
    """
    Function returning the molecular formulas of all amino acids. All amino acids are considered in the neutral form
    (uncharged).
    """
    formulas = {'A': {'C': 3, 'H': 7, 'N': 1, 'O': 2, 'S': 0},
                'C': {'C': 3, 'H': 7, 'N': 1, 'O': 2, 'S': 1},
                'D': {'C': 4, 'H': 7, 'N': 1, 'O': 4, 'S': 0},
                'E': {'C': 5, 'H': 9, 'N': 1, 'O': 4, 'S': 0},
                'F': {'C': 9, 'H': 11, 'N': 1, 'O': 2, 'S': 0},
                'G': {'C': 2, 'H': 5, 'N': 1, 'O': 2, 'S': 0},
                'H': {'C': 6, 'H': 9, 'N': 3, 'O': 2, 'S': 0},
                'I': {'C': 6, 'H': 13, 'N': 1, 'O': 2, 'S': 0},
                'K': {'C': 6, 'H': 14, 'N': 2, 'O': 2, 'S': 0},
                'L': {'C': 6, 'H': 13, 'N': 1, 'O': 2, 'S': 0},
                'M': {'C': 5, 'H': 11, 'N': 1, 'O': 2, 'S': 1},
                'N': {'C': 4, 'H': 8, 'N': 2, 'O': 3, 'S': 0},
                'P': {'C': 5, 'H': 9, 'N': 1, 'O': 2, 'S': 0},
                'Q': {'C': 5, 'H': 10, 'N': 2, 'O': 3, 'S': 0},
                'R': {'C': 6, 'H': 14, 'N': 4, 'O': 2, 'S': 0},
                'S': {'C': 3, 'H': 7, 'N': 1, 'O': 3, 'S': 0},
                'T': {'C': 4, 'H': 9, 'N': 1, 'O': 3, 'S': 0},
                'V': {'C': 5, 'H': 11, 'N': 1, 'O': 2, 'S': 0},
                'W': {'C': 11, 'H': 12, 'N': 2, 'O': 2, 'S': 0},
                'Y': {'C': 9, 'H': 11, 'N': 1, 'O': 3, 'S': 0}
                }
    return formulas
