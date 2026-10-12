"""Physical constants and default device/material parameters for the 1D diode TCAD solver.

All quantities are in CGS-practical semiconductor units:
  length   -> cm
  time     -> s
  charge   -> C
  potential-> V
  concentration -> cm^-3
  current density -> A/cm^2
"""
from dataclasses import dataclass, field

from core.doping_profiles import DopingProfile


# ---- Universal constants ----
Q = 1.602176634e-19        # elementary charge, C
KB = 1.380649e-23          # Boltzmann constant, J/K
EPS0 = 8.8541878128e-14    # vacuum permittivity, F/cm


@dataclass
class Material:
    T: float = 300.0                # temperature, K
    eps_r: float = 11.7             # Si relative permittivity
    ni: float = 1.0e10              # intrinsic carrier concentration, cm^-3
    mu_n: float = 1350.0            # electron LOW-FIELD mobility, cm^2/V/s
    mu_p: float = 480.0             # hole LOW-FIELD mobility, cm^2/V/s
    # Caughey-Thomas velocity-saturation model (see solver2d/newton_solver_qf_2d.py's
    # mobility_field() for the formula and derivation): mu(E) = mu0/(1+(mu0*|E|/vsat)^beta)^(1/beta).
    # Values are the standard textbook/literature set for 300K silicon (e.g. Sze & Ng,
    # "Physics of Semiconductor Devices", 3rd ed.; FLOOXS's Canali model
    # ~/Desktop/github_flooxs/flooxs/TclLib/Device/floods/Silicon/Mobility/Canali/canali.tcl
    # uses the same functional form with continuously-fitted exponents
    # beta_n=1.109, beta_p=1.213 and vsat_n0=1.07e7, vsat_p0=8.73e6 cm/s at 300K - close
    # to, but not identical to, the simpler integer/near-integer beta values used here).
    vsat_n: float = 1.0e7           # electron saturation velocity, cm/s
    vsat_p: float = 8.0e6           # hole saturation velocity, cm/s
    beta_n: float = 2.0             # electron Caughey-Thomas exponent (dimensionless)
    beta_p: float = 1.0             # hole Caughey-Thomas exponent (dimensionless)
    # Caughey-Thomas DOPING-dependence low-field mobility model (D. M.
    # Caughey and R. E. Thomas, "Carrier mobilities in silicon empirically
    # related to doping and field," Proc. IEEE 55, 2192-2193, 1967), with the
    # standard 300K silicon parameter set reproduced in Sentaurus Device's
    # "PhuMob"/doping-dependence table and Silvaco ATLAS's CONMOB model
    # (also Selberherr, "Analysis and Simulation of Semiconductor Devices",
    # 1984, Table 4.1): mu(N) = mu_min + (mu_max-mu_min)/(1+(N/N_ref)^alpha),
    # evaluated at the LOCAL TOTAL (ionized) doping magnitude |Cdop|. This
    # is the "doping-dependent mobility" solver2d/newton_solver_qf_2d.py
    # actually uses in its active solve path (see mobility_doping()) -
    # mu_n/mu_p above remain the mu(N->0) reference constants used
    # elsewhere (analytic formulas, 1D solvers, etc.) but are NOT what the
    # 2D solver evaluates on an edge.
    mu_n_max: float = 1417.0        # electron mu_max, cm^2/V/s
    mu_n_min: float = 68.5          # electron mu_min, cm^2/V/s
    N_ref_n: float = 9.2e16         # electron reference doping, cm^-3
    alpha_n: float = 0.711          # electron doping-dependence exponent
    mu_p_max: float = 470.5         # hole mu_max, cm^2/V/s
    mu_p_min: float = 44.9          # hole mu_min, cm^2/V/s
    N_ref_p: float = 2.23e17        # hole reference doping, cm^-3
    alpha_p: float = 0.719          # hole doping-dependence exponent
    # Which doping dependence the 2D/3D solver uses: "masetti" (default) or
    # "caughey_thomas" (the parameters above). Masetti, Severi and Solmi,
    # "Modeling of carrier mobility against carrier concentration in
    # arsenic-, phosphorus-, and boron-doped silicon," IEEE TED 30, 764
    # (1983) - Sentaurus Device's default DopingDependence for silicon:
    #   mu(N) = mu_min1*exp(-Pc/N) + (mu_max - mu_min2)/(1 + (N/Cr)^alpha)
    #           - mu1/(1 + (Cs/N)^beta)
    # Below ~1e18 it is Caughey-Thomas within a few %; the extra terms fit
    # the measured mobility in heavily doped (source/drain, 1e19-1e21)
    # silicon, where Caughey-Thomas runs high (electrons: 6% at 1e19, 28% at
    # 1e20; DEVELOPMENT_LOG 55).
    # Coefficients: arsenic (electrons) and boron (holes), Masetti's Table I.
    doping_mobility: str = "masetti"
    masetti_n: tuple = (52.2, 52.2, 43.4, 0.0, 9.68e16, 3.43e20, 0.680, 2.0, 1417.0)
    masetti_p: tuple = (44.9, 0.0, 29.0, 9.23e16, 2.23e17, 6.10e20, 0.719, 2.0, 470.5)
    #                  (mu_min1, mu_min2, mu1, Pc, Cr, Cs, alpha, beta, mu_max)
    tau_n: float = 1.0e-9           # SRH electron lifetime, s
    tau_p: float = 1.0e-9           # SRH hole lifetime, s
    chi_eV: float = 4.05            # Si electron affinity, eV (band diagrams only)
    Eg_eV: float = 1.12             # Si bandgap, eV (300K; band diagrams only)
    Nc: float = 2.8e19              # conduction-band effective density of states, cm^-3
                                     # (300K Si) - only used by fermi_dirac.py's degenerate
                                     # (Fermi-Dirac statistics) equilibrium relations, not by
                                     # the Boltzmann relations solve_poisson/solve_continuity_*
                                     # use throughout the actual PDE solve.
    Nv: float = 1.04e19             # valence-band effective density of states, cm^-3 (300K Si)

    @property
    def Vt(self) -> float:
        return KB * self.T / Q

    @property
    def eps(self) -> float:
        return self.eps_r * EPS0

    @property
    def Dn(self) -> float:
        return self.mu_n * self.Vt

    @property
    def Dp(self) -> float:
        return self.mu_p * self.Vt

    @property
    def Ln(self) -> float:
        return (self.Dn * self.tau_n) ** 0.5

    @property
    def Lp(self) -> float:
        return (self.Dp * self.tau_p) ** 0.5


@dataclass
class Device:
    """Step p-n junction. Region x<0 is p-type (Na), x>=0 is n-type (Nd).

    Lengths of the quasi-neutral regions are set automatically (in Grid) to a
    multiple of the relevant minority-carrier diffusion length unless overridden.
    """
    Na: float = 1.0e17   # p-side acceptor doping, cm^-3
    Nd: float = 1.0e16   # n-side donor doping, cm^-3
    area: float = 1.0e-4  # device cross-sectional area, cm^2 (1e-4 cm^2 = 100 um x 100 um)

    Wp: float = None      # p-side length, cm (None -> auto)
    Wn: float = None      # n-side length, cm (None -> auto)
    n_diffusion_lengths: float = 5.0  # how many L's of quasi-neutral region to keep

    p_profile: DopingProfile = None  # None -> flat at Na (see mesh.build_diode_grid)
    n_profile: DopingProfile = None  # None -> flat at Nd
