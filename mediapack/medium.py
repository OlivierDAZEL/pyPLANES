#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# medium.py
#
# This file is part of pymls, a software distributed under the MIT license.
# For any question, please contact one of the authors cited below.
#
# Copyright (c) 2017
# 	Olivier Dazel <olivier.dazel@univ-lemans.fr>
# 	Mathieu Gaborit <gaborit@kth.se>
# 	Peter Göransson <pege@kth.se>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.


class Medium(object):
    """ Base class for medium definition and manipulation

    All derived classes have a few class parameters explained below.

    Attributes
    ----------

    EXPECTED_PARAMS, OPT_PARAMS : list of tuple, class attribute
        List of all expected parameters (resp. optional parameters) for this medium to be well defined.
        The format is `(param_name, param_type)` where `param_name` is a `str` and
        `param_type` a `type`.
    MEDIUM_TYPE : str, class attribute
        State the type of medium , used to determine the real physics hidden in the medium
        (to distinguish a fluid from and equivalent fluid for instance)
    MODEL : str, class attribute
        State which type of propagation model to use.
    name : str
        Medium's name (default to Generic Medium)
    omega : positive float or -1
        Last circular frequency for which the parameters have been updated
    """

    EXPECTED_PARAMS = []
    OPT_PARAMS = []
    MEDIUM_TYPE = 'generic'
    MODEL = ''

    def __init__(self, **params):
        self.omega = -1
        self.name = 'Generic Medium'
        self.extras = {}

        if params:
            self.from_dict(params)

    def __str__(self):
        return '{} (type: {}, model: {})'.format(
            self.name,
            self.__class__.MEDIUM_TYPE,
            self.__class__.MODEL
        )

    def as_dict(self):
        return {k: self.__getattribute__(k) for k, _ in self.EXPECTED_PARAMS+self.OPT_PARAMS if hasattr(self, k) and self.__getattribute__(k) is not None}

    def update_frequency(self, omega):
        """ Computes parameters' value for the given circular frequency

        Notes
        -----

        Implemented in derived classes
        """
        pass

    def _compute_missing(self):
        """ Computes the required constant parameters missing from the definition

        Notes
        -----

        Implemented in derived classes
        """
        pass

    def from_dict(self, parameters):
        """Reads medium definition from a hashmap of params.

        Load all possible & recognised parameters and run `_compute_missing` to complete
        the definition.

        Parameters
        ----------
        parameters : dict
            Linking parameter names and values.

        Raises
        ------
        LookupError
            If the parameter definition is incomplete (missing parameters listed in EXPECTED_PARAMS).
        """

        for param, param_type in self.__class__.EXPECTED_PARAMS:
            param_value = parameters.get(param)
            if param_value is None:
                raise LookupError('Unable to find definition of parameter "{}"'.format(param))
            else:
                setattr(self, param, param_type(param_value))
        for param, param_type in self.__class__.OPT_PARAMS:
            param_value = parameters.get(param)
            if param_value is not None:
                setattr(self, param, param_type(param_value))
        self.name = parameters.get('name', "Unnamed Medium")
        extra_params = set(parameters.keys()) - set([_[0] for _ in self.__class__.EXPECTED_PARAMS + self.__class__.OPT_PARAMS])
        for param in extra_params:
            self.extras[param] = parameters[param]

        self._compute_missing()
