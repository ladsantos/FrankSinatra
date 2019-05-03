#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import scipy.optimize as sp
import astropy.constants as c
import astropy.units as u
import astropy.time as t

"""
This module computes the radial velocities of a massive object being orbited by
a secondary massive object, and is based on the formalism from Murray & Correia
(2011), available freely at http://arxiv.org/abs/1009.1738. The equation numbers
are from this article, unless otherwise noted.
"""


class BinarySystem(object):
    """
    A class that computes the radial velocities given the orbital parameters of
    the binary system.

    Parameters
    ----------
    mass_main : ``astropy.Quantity``
        Mass of the main object of which the radial velocity is desired.

    mass_secondary : ``astropy.Quantity``
        Mass of the secondary object in the system.

    time_periastron : ``astropy.Time``
        Time of periastron passage.

    inclination : ``astropy.Quantity``
        Inclination of the orbit of the main object in relation to the
        reference plane.

    argument_periapse : ``astropy.Quantity``
        Argument of periapse.

    eccentricity : scalar, optional
        Eccentricity of the orbit. Default is 0 (circular orbit).

    orbital_period : ``astropy.Quantity``, optional
        The orbital period. If not provided, then the semi-major axis of the
        orbit needs to be provided.

    semi_major_axis : ``astropy.Quantity``, optional
        Semi-major axis of the orbit. If not provided, then the orbital period
        needs to be provided.

    reference_velocity : ``astropy.Quantity``, optional
        Systemic velocity of the system. Default is 0.
    """
    def __init__(self, mass_main, mass_secondary,
                 time_periastron, inclination, argument_periapse,
                 eccentricity, orbital_period=None,
                 semi_major_axis=None, reference_velocity=None):

        # Assign all global variables
        self.m1 = mass_main
        self.m2 = mass_secondary
        self.t0 = time_periastron
        self.gamma = reference_velocity
        self.i = inclination
        self.omega = argument_periapse
        self.ecc = eccentricity

        # Either the orbital period or the semi-major axis has to be provided
        if orbital_period is not None:
            semi_major_axis = (orbital_period ** 2 * c.G *
                               (self.m1 + self.m2) / 4 / np.pi ** 2) ** (1 / 3)
        elif semi_major_axis is not None:
            orbital_period = (semi_major_axis ** 3 * 4 * np.pi
                              ** 2 / c.G / (self.m1 + self.m2)) ** 0.5
        else:
            raise ValueError('Either the orbital period or the semi-major axis '
                             'has to be provided.')
        self.period = orbital_period.to(u.d)
        self.a = semi_major_axis.to(u.au)

        # We perform calculations with all units already converted
        self._m1 = self.m1.to(u.solMass).value
        self._m2 = self.m2.to(u.solMass).value
        self._t0 = self.t0.jd
        self._i = self.i.to(u.rad).value
        self._omega = self.omega.to(u.rad).value
        self._period = self.period.to(u.d).value
        self._a = self.a.to(u.au).value
        self._grav = c.G.to(u.au ** 3 / u.solMass / u.d ** 2).value

        if self.ecc > 1:
            raise ValueError('Keplerian orbits are ellipses, therefore ecc <= '
                             '1')

    # Compute Eq. 65
    def rv_eq(self, f):
        """
        The radial velocities equation.

        Parameters
        ----------
        f : scalar or ``numpy.ndarray``
            True anomaly in radians.

        Returns
        -------
        rvs : scalar or ``numpy.ndarray``
            Radial velocity
        """
        # Compute the radial velocity in AU / d
        k1 = (self._grav / ((self._m1 + self._m2) * self._a *
                            (1 - self.ecc ** 2))) ** 0.5
        k2 = self._m2 * np.sin(self._i)
        k3 = np.cos(self._omega + f) + self.ecc * np.cos(self._omega)
        rv = k1 * k2 * k3
        return rv

    # Calculates the Kepler equation (Eq. 41)
    def kep_eq(self, e_ano, m_ano):
        """
        The Kepler equation.

        Parameters
        ----------
        e_ano : scalar
            Eccentric anomaly in radians.

        m_ano : scalar
            Mean anomaly in radians.

        Returns
        -------
        kep: scalar
            Value of E-e*sin(E)-M
        """
        kep = e_ano - self.ecc * np.sin(e_ano) - m_ano
        return kep

    # Calculates the radial velocities for given orbital parameters
    def get_rvs(self, ts):
        """
        Computes the radial velocity given the orbital parameters.

        Parameters
        ----------
        ts : ``astropy.Quantity`` or ``numpy.ndarray``
            Sample time in which to compute the radial velocity.

        Returns
        -------
        rvs : ``astropy.Quantity`` or ``numpy.ndarray``
            Radial velocities.
        """
        if isinstance(ts, t.Time):
            ts = np.array([ts.jd])
        elif isinstance(ts, np.ndarray) or isinstance(ts, list):
            ts = np.array([tk.jd for tk in ts])

        m_ano = 2 * np.pi / self._period * (ts - self._t0)  # Mean anomaly
        e_ano = np.array([sp.newton(func=self.kep_eq, x0=mk, args=(mk,))
                          for mk in m_ano])      # Eccentric anomaly
        # Computing the true anomaly
        f = 2 * np.arctan2(np.sqrt(1. + self.ecc) * np.sin(e_ano / 2),
                           np.sqrt(1. - self.ecc) * np.cos(e_ano / 2))
        # Why do we compute the true anomaly in this weird way? Because
        # arc-cosine is degenerate in the interval 0-360 degrees.
        rvs = (self.rv_eq(f) * u.au / u.d).to(u.km / u.s) + self.gamma
        return rvs
