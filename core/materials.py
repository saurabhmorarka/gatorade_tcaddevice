"""Material-properties catalog schema.

Distinct from core.params.Material on purpose: Material is the *live,
T-resolved, solver-facing* object each simulation actually uses (has a
concrete T, has derived @property quantities like Vt/Dn/Dp). MaterialProperties
is the *catalog* entry a named material is looked up as (no T, no derived
properties) - the same separation this project already uses between
DopingProfile (pure data) and Material (data + derived @property values).

All quantities use this project's existing CGS-practical unit convention
(see params.py's module docstring): length -> cm, energy -> eV where noted,
concentration -> cm^-3.
"""
import dataclasses
from dataclasses import dataclass

import numpy as np

from core.params import KB, Q, Material


@dataclass
class MaterialProperties:
    """One catalog entry: a named material's fixed reference-temperature
    properties. Fields that don't apply to a given category (e.g. eps_r for
    a bare metal work-function entry) are left None rather than given a
    fabricated numeric default.
    """
    name: str
    category: str                          # "semiconductor" | "insulator" | "metal"

    eps_r: float = None                    # relative permittivity
    chi_eV: float = None                   # electron affinity, eV (semiconductors)
    Eg_eV_300K: float = None               # bandgap at 300K, eV (semiconductors)
    Nc_300K: float = None                  # conduction-band DOS at 300K, cm^-3
    Nv_300K: float = None                  # valence-band DOS at 300K, cm^-3
    mu_n: float = None                     # electron mobility, cm^2/V/s (constant-mobility model)
    mu_p: float = None                     # hole mobility, cm^2/V/s
    tau_n: float = None                    # SRH electron lifetime, s
    tau_p: float = None                    # SRH hole lifetime, s
    workfunction_eV: float = None          # metals: bare work function (no chi/Eg split)

    # Varshni Eg(T) coefficients - None (no scaling data) leaves Eg(T) flat
    # at Eg_eV_300K wherever it's used (see materials.eg_at_T).
    varshni_alpha_eV_per_K: float = None
    varshni_beta_K: float = None

    # Mechanics (mechanics/ stress solver). Every material has an isotropic
    # E and nu; cubic crystals (Si, Ge, SiGe) also carry their stiffness
    # tensor C11/C12/C44 (crystal axes), which the solver then uses
    # instead of E/nu, rotated to the device frame (mechanics/elastic.py).
    youngs_modulus_GPa: float = None
    poisson_ratio: float = None
    c11_GPa: float = None
    c12_GPa: float = None
    c44_GPa: float = None
    lattice_A: float = None                # cubic lattice constant, Angstrom (epitaxial mismatch)
    # Typical as-deposited intrinsic stress (GPa, + tensile) of a film of
    # this material: a rough default for a stressor that gives none.
    intrinsic_stress_GPa: float = None


@dataclass
class AlloyMaterial:
    """A composition-dependent alloy, resolved on demand into a
    MaterialProperties via Vegard's-law linear mixing of its two
    end-members (plus an optional quadratic bowing correction on Eg).

    Deliberately NOT a MATERIALS dict entry itself, and NOT the same
    mechanism as material_db.derive() (discrete copy-and-override of one
    fixed entry): an alloy is a *parametric* family (a continuum of
    possible compositions), while derive() produces one new *discrete*
    named entry. Kept as a separate concept so neither mechanism has to
    understand the other's math.
    """
    name: str
    end_member_a: str    # material_db key, composition fraction x=1
    end_member_b: str    # material_db key, composition fraction x=0
    bowing_eV: float = 0.0

    def resolve(self, x: float) -> MaterialProperties:
        """x = fraction of end_member_a, 0 <= x <= 1. Returns a new
        MaterialProperties with every numeric field linearly interpolated
        between the two end-members (None if either end-member's field is
        None), plus the standard quadratic bowing correction on Eg_eV_300K:
        Eg(x) = x*Eg_a + (1-x)*Eg_b - bowing*x*(1-x).
        """
        from core import material_db

        a = material_db.get(self.end_member_a)
        b = material_db.get(self.end_member_b)

        def mix(field_name):
            va, vb = getattr(a, field_name), getattr(b, field_name)
            if va is None or vb is None:
                return None
            return x * va + (1 - x) * vb

        eg = mix("Eg_eV_300K")
        if eg is not None:
            eg = eg - self.bowing_eV * x * (1 - x)

        return MaterialProperties(
            name=f"Si{x:.2f}Ge{1 - x:.2f}" if self.name == "SixGe1-x" else f"{self.name}(x={x:.2f})",
            category="semiconductor",
            eps_r=mix("eps_r"),
            chi_eV=mix("chi_eV"),
            Eg_eV_300K=eg,
            Nc_300K=mix("Nc_300K"),
            Nv_300K=mix("Nv_300K"),
            mu_n=mix("mu_n"),
            mu_p=mix("mu_p"),
            tau_n=mix("tau_n"),
            tau_p=mix("tau_p"),
            varshni_alpha_eV_per_K=mix("varshni_alpha_eV_per_K"),
            varshni_beta_K=mix("varshni_beta_K"),
            youngs_modulus_GPa=mix("youngs_modulus_GPa"),
            poisson_ratio=mix("poisson_ratio"),
            c11_GPa=mix("c11_GPa"),
            c12_GPa=mix("c12_GPa"),
            c44_GPa=mix("c44_GPa"),
            lattice_A=mix("lattice_A"),
        )

    def resolve_strained(self, x: float, substrate: str = "Silicon",
                          x_substrate_Ge: float = 0.0) -> MaterialProperties:
        """Compressively strained Si(1-x_Ge)Ge(x_Ge)-grown-on-`substrate`
        MaterialProperties (see strained_sige_on_si_offsets()'s docstring
        for the physics) - a SEPARATE method from resolve() (not a flag),
        so the relaxed and strained compositions stay both directly
        available/comparable, matching this project's existing swappable-
        model pattern (Hurkx vs. Schenk TAT, the three Kane P-variants).

        Starts from the SAME Vegard-mixed MaterialProperties resolve(x)
        already produces: eps_r, Nc_300K, Nv_300K, mu_n, mu_p, tau_n, tau_p
        are UNCHANGED from the relaxed case - explicitly scoped out, not a
        silent gap (see the harmonic-snuggling-puddle plan's "what this
        does NOT attempt" section: no valley-splitting DOS correction, no
        strain-enhanced mobility, no critical-thickness/relaxation check).
        Only Eg_eV_300K/chi_eV are overridden, via the delta_Ec/delta_Ev
        shifts relative to the SUBSTRATE material's own Eg_eV_300K/chi_eV:
            Eg_strained  = Eg_substrate + delta_Ec - delta_Ev
            chi_strained = chi_substrate - delta_Ec
        (Ec_substrate is the zero reference; Ev = Ec - Eg, so shifting Ec up
        by delta_Ec and Ev down by delta_Ev - equivalently Ec up by
        delta_Ec - widens the gap by delta_Ec on the conduction side and
        narrows it by delta_Ev on the valence side, net Eg change =
        delta_Ec - delta_Ev; chi = Evac - Ec, so raising Ec by delta_Ec
        lowers chi by the same amount.)

        x_Ge is resolved from THIS AlloyMaterial's own end-member identity
        (whichever of end_member_a/end_member_b is "Germanium"), so the
        SAME `x` convention resolve() uses (fraction of end_member_a)
        still applies here - only the physics used to set Eg_eV_300K/
        chi_eV changes; eps_r/Nc_300K/etc. still come from resolve(x) at
        this identical x."""
        from core import material_db

        relaxed = self.resolve(x)

        if self.end_member_a == "Germanium":
            x_Ge = x
        elif self.end_member_b == "Germanium":
            x_Ge = 1.0 - x
        else:
            raise ValueError(
                "resolve_strained only supports a Silicon/Germanium AlloyMaterial "
                f"pair (got end_member_a={self.end_member_a!r}, "
                f"end_member_b={self.end_member_b!r}) - the People & Bean strain "
                "formula is Si/Ge-specific.")

        substrate_props = material_db.get(substrate)
        delta_Ec, delta_Ev = strained_sige_on_si_offsets(x_Ge, x_substrate_Ge)

        return dataclasses.replace(
            relaxed,
            Eg_eV_300K=substrate_props.Eg_eV_300K + delta_Ec - delta_Ev,
            chi_eV=substrate_props.chi_eV - delta_Ec,
        )


def strained_sige_on_si_offsets(x_Ge: float, x_Ge_substrate: float = 0.0):
    """Compressively strained Si(1-x)Ge(x)-on-relaxed-Si(1-x')Ge(x') band
    offsets (x = x_Ge, x' = x_Ge_substrate) - People & Bean, Appl. Phys.
    Lett. 48, 538 (1986); consistent with Van de Walle & Martin, Phys. Rev.
    B 34, 5621 (1986) (confirmed via web search). Under biaxial compressive
    strain (SiGe's larger relaxed lattice constant compressed in-plane to
    match a lower-Ge-fraction/pure-Si substrate), almost the ENTIRE bandgap
    reduction from Ge alloying lands in the VALENCE band - qualitatively
    different from the relaxed-alloy Vegard-mixed model (AlloyMaterial.
    resolve()), which implicitly splits the bandgap difference between
    conduction and valence bands however linear-mixed chi_eV/Nc/Nv happen
    to place it:

        delta_Ev(x) = (0.74 - 0.53*x') * x   eV
        delta_Ec(x) ~= 0                     eV   (near-zero, "a few meV"
            in the literature for this composition range - a standard
            Type-I device-modeling simplification, NOT claimed to be
            exactly zero; see the harmonic-snuggling-puddle plan's "what
            this does NOT attempt" section for what else strain is known
            to affect but isn't modeled here)

    x_Ge_substrate=0.0 (growth on pure Si, this project's only case so
    far) gives delta_Ev(x) = 0.74*x eV exactly - e.g. x=0.4 (Si0.6Ge0.4)
    gives delta_Ev=0.296 eV. Returns (delta_Ec_eV, delta_Ev_eV), both >= 0
    for x_Ge >= x_Ge_substrate (i.e. growing a HIGHER-Ge-fraction layer on
    a lower-Ge-fraction/pure-Si substrate - the compressive-strain regime
    this formula is fit to; growing a LOWER-Ge-fraction layer on a higher-
    Ge-fraction substrate would be tensile strain, a different regime not
    covered by this formula)."""
    delta_Ev = (0.74 - 0.53 * x_Ge_substrate) * x_Ge
    delta_Ec = 0.0
    return delta_Ec, delta_Ev


# ---- Temperature-dependent property formulas ----
# T=300.0 (K) below is the reference temperature every MaterialProperties
# entry's *_300K fields are quoted at, not a magic number.
_T_REF = 300.0


def eg_at_T(props: MaterialProperties, T: float) -> float:
    """Varshni equation, Eg(T) = Eg(0) - alpha*T^2/(T+beta), re-based so it
    returns exactly Eg_eV_300K at T=_T_REF without needing a separate Eg(0)
    field:
        Eg(T) = Eg_eV_300K - alpha*(T^2/(T+beta) - _T_REF^2/(_T_REF+beta))
    Materials with no Varshni fit (insulators, metals) hold Eg flat at
    Eg_eV_300K - there's no scaling data to apply.
    """
    if props.Eg_eV_300K is None:
        return None
    if props.varshni_alpha_eV_per_K is None or props.varshni_beta_K is None:
        return props.Eg_eV_300K
    a, b = props.varshni_alpha_eV_per_K, props.varshni_beta_K
    shift = a * (T ** 2 / (T + b) - _T_REF ** 2 / (_T_REF + b))
    return props.Eg_eV_300K - shift


def nc_at_T(props: MaterialProperties, T: float) -> float:
    """Effective conduction-band DOS, Nc(T) = Nc_300K * (T/300)^1.5."""
    if props.Nc_300K is None:
        return None
    return props.Nc_300K * (T / _T_REF) ** 1.5


def nv_at_T(props: MaterialProperties, T: float) -> float:
    """Effective valence-band DOS, Nv(T) = Nv_300K * (T/300)^1.5."""
    if props.Nv_300K is None:
        return None
    return props.Nv_300K * (T / _T_REF) ** 1.5


def ni_at_T(props: MaterialProperties, T: float) -> float:
    """Intrinsic carrier concentration, ni(T) = sqrt(Nc(T)*Nv(T)) *
    exp(-Eg(T) / (2*Vt(T))), the standard Boltzmann-statistics formula."""
    nc, nv, eg = nc_at_T(props, T), nv_at_T(props, T), eg_at_T(props, T)
    if nc is None or nv is None or eg is None:
        raise ValueError(
            f"ni_at_T requires Nc_300K, Nv_300K, and Eg_eV_300K on {props.name!r}"
        )
    Vt = KB * T / Q
    return (nc * nv) ** 0.5 * np.exp(-eg / (2 * Vt))


def Xi(mat: Material) -> float:
    """Intrinsic-level-to-vacuum-level reference, Xi = chi + Vt*ln(Nc/ni)
    (electron affinity plus how far Ei sits below Ec) - the material-only
    (no psi, no mesh) quantity whose DIFFERENCE between two materials gives
    the Anderson's-rule/electron-affinity-rule heterojunction band-offset
    correction delta_Ei (see MaterialField's own docstring and
    delta_Ei_of() below). Pure scalar algebra on one Material - used both
    by MaterialField.from_regions() (mesh-aware, per-node/edge arrays) and
    core.analytic's closed-form Vbi_hetero (mesh-independent), so the two
    stay numerically identical by construction rather than by coincidence."""
    return mat.chi_eV + mat.Vt * np.log(mat.Nc / mat.ni)


def delta_Ei_of(mat: Material, mat_ref: Material) -> float:
    """delta_Ei(mat) relative to mat_ref - see Xi()'s docstring. 0.0 exactly
    when mat is mat_ref (or an identically-parameterized material), by
    construction (Xi(mat)-Xi(mat)=0)."""
    return Xi(mat) - Xi(mat_ref)


@dataclass
class MaterialField:
    """Per-node/per-edge material properties - the general heterojunction-
    capable sibling of a plain scalar Material, mirroring core.mesh's
    existing per-node/per-edge precedent (build_mos_grid's eps_edge,
    ni_arr, is_oxide). Every solver function that used to read a scalar
    mat.<attr> is generalized to read the equivalent array here;
    MaterialField.uniform() reproduces the old scalar behavior bit-for-bit
    (the same constant broadcast into every array element - elementwise
    multiplication by a repeated-constant array is bit-identical to
    multiplying by the bare scalar, no reduction/reordering involved), so a
    caller that never builds a heterojunction never needs to know this
    class exists.

    delta_Ei_arr is the Anderson's-rule/electron-affinity-rule band-offset
    correction to this project's existing intrinsic-level Boltzmann
    relations (see the harmonic-snuggling-puddle plan's physics section):
        n(x) = ni(x) * exp((psi(x) - phin(x) + delta_Ei(x)) / Vt)
        p(x) = ni(x) * exp((phip(x) - psi(x) - delta_Ei(x)) / Vt)
    identically 0 for a homogeneous device (recovering today's exact
    formula), and independent of every Newton unknown (psi, phin, phip),
    so it never touches an existing Jacobian entry - only the forward n/p
    evaluation (and the two places that invert it: equilibrium_bulk_potential
    and each solver's phin_bc/phip_bc-from-contact-density derivation) gain
    one added/subtracted term.
    """
    Vt: float                  # K, uniform across the device (single T assumed)
    eps_edge: np.ndarray       # len(x)-1
    mu_n_edge: np.ndarray      # len(x)-1
    mu_p_edge: np.ndarray      # len(x)-1
    ni_arr: np.ndarray         # len(x)
    tau_n_arr: np.ndarray      # len(x)
    tau_p_arr: np.ndarray      # len(x)
    delta_Ei_arr: np.ndarray   # len(x), 0 in the reference region

    @property
    def Dn_edge(self) -> np.ndarray:
        return self.mu_n_edge * self.Vt

    @property
    def Dp_edge(self) -> np.ndarray:
        return self.mu_p_edge * self.Vt

    @staticmethod
    def uniform(mat: Material, x) -> "MaterialField":
        """Constant arrays reproducing a single scalar Material everywhere -
        delta_Ei_arr = 0 identically (no reference-material ambiguity when
        there's only one material)."""
        N = len(x)
        return MaterialField(
            Vt=mat.Vt,
            eps_edge=np.full(N - 1, mat.eps),
            mu_n_edge=np.full(N - 1, mat.mu_n),
            mu_p_edge=np.full(N - 1, mat.mu_p),
            ni_arr=np.full(N, mat.ni),
            tau_n_arr=np.full(N, mat.tau_n),
            tau_p_arr=np.full(N, mat.tau_p),
            delta_Ei_arr=np.zeros(N),
        )

    @staticmethod
    def from_regions(mat_p: Material, mat_n: Material, x, junction_index,
                      ref: str = "n") -> "MaterialField":
        """Stepped per-node/per-edge arrays for a p-side/n-side heterojunction
        diode, matching core.mesh.build_diode_grid's own x<0=p-side, x>=0=
        n-side convention exactly (so the node/edge material split is
        position-based, consistent with how Cdop itself is assigned there -
        junction_index, the node nearest x=0, is accepted for API symmetry
        with that function but the actual split uses the same x>=0 test
        Cdop uses, not the index, since the two can differ by one node when
        the nearest-to-zero node happens to be on the negative side).

        delta_Ei(x) = Xi(x) - Xi(reference material), Xi = chi + Vt*ln(Nc/ni)
        (see MaterialField's own docstring) - ref selects which material's
        Xi is the zero reference ("n", the default, matches the ohmic
        n-side contact convention core.solver already uses as its Va=0
        ground reference).

        mat_p and mat_n must share the same T (Vt) - this project's
        MaterialField only carries one Vt for the whole device (uniform-
        temperature assumption); raises ValueError otherwise rather than
        silently picking one.
        """
        if not np.isclose(mat_p.Vt, mat_n.Vt):
            raise ValueError(
                "MaterialField.from_regions requires mat_p and mat_n at the same "
                f"temperature (mat_p.T={mat_p.T}K, mat_n.T={mat_n.T}K)")
        if ref not in ("n", "p"):
            raise ValueError(f"ref must be 'n' or 'p', got {ref!r}")

        x = np.asarray(x, dtype=float)
        N = len(x)

        if ref == "n":
            delta_Ei_p, delta_Ei_n = delta_Ei_of(mat_p, mat_n), 0.0
        else:
            delta_Ei_p, delta_Ei_n = 0.0, delta_Ei_of(mat_n, mat_p)

        is_n_node = x >= 0.0
        x_mid = (x[:-1] + x[1:]) / 2.0
        is_n_edge = x_mid >= 0.0

        return MaterialField(
            Vt=mat_n.Vt,
            eps_edge=np.where(is_n_edge, mat_n.eps, mat_p.eps),
            mu_n_edge=np.where(is_n_edge, mat_n.mu_n, mat_p.mu_n),
            mu_p_edge=np.where(is_n_edge, mat_n.mu_p, mat_p.mu_p),
            ni_arr=np.where(is_n_node, mat_n.ni, mat_p.ni),
            tau_n_arr=np.where(is_n_node, mat_n.tau_n, mat_p.tau_n),
            tau_p_arr=np.where(is_n_node, mat_n.tau_p, mat_p.tau_p),
            delta_Ei_arr=np.where(is_n_node, delta_Ei_n, delta_Ei_p),
        )


def resolve_material(props: MaterialProperties, T: float = 300.0) -> Material:
    """Bridge from a catalog entry to a live, solver-ready core.params.Material
    with every T-dependent field actually computed at T (ni, Eg, Nc, Nv);
    mu_n/mu_p/tau_n/tau_p carry straight through unchanged (constant-mobility
    model, no T-dependence implemented for those yet)."""
    return Material(
        T=T,
        eps_r=props.eps_r,
        ni=ni_at_T(props, T),
        mu_n=props.mu_n,
        mu_p=props.mu_p,
        tau_n=props.tau_n,
        tau_p=props.tau_p,
        chi_eV=props.chi_eV,
        Eg_eV=eg_at_T(props, T),
        Nc=nc_at_T(props, T),
        Nv=nv_at_T(props, T),
    )
