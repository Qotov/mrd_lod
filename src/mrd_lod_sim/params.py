"""Parameter objects, physical constants, and the VAF <-> TF converters.

Two conventions are load-bearing throughout the package (BUILD_SPEC 2.2):

- The model works internally in **per-site VAF** (variant allele fraction).
- Users reason in **tumour fraction (TF)**. For a clonal heterozygous SNV,
  ``VAF = TF / 2``.

Confusing the two is a factor-of-two error at ppm scale, so conversion is
funnelled through the two explicit functions below -- never open-coded.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Mass of DNA per haploid genome equivalent.
#:
#: Order-of-magnitude prior: a diploid human genome is ~6.6 pg, so one haploid
#: genome equivalent is ~3.3 pg = 0.0033 ng. This is the conversion between an
#: input cfDNA mass and a molecule count. Configurable, not a magic number in
#: logic. (Ref: standard human genome mass, e.g. Dolezel et al. 2003.)
PG_PER_GENOME_EQUIVALENT: float = 0.0033  # ng per haploid genome equivalent


def vaf_from_tf(tf: float, *, copies_altered: int = 1, ploidy: int = 2) -> float:
    """Per-site VAF from tumour fraction.

    For a clonal heterozygous SNV (one altered copy out of two), ``VAF = TF/2``.
    The general relationship for a variant present on ``copies_altered`` of
    ``ploidy`` copies is ``VAF = TF * copies_altered / ploidy``.

    Args:
        tf: tumour fraction (fraction of molecules of tumour origin).
        copies_altered: altered copies carrying the variant (default 1).
        ploidy: total copies at the locus (default 2, diploid).

    Returns:
        Per-site VAF.
    """
    if ploidy <= 0:
        raise ValueError("ploidy must be positive")
    return tf * copies_altered / ploidy


def tf_from_vaf(vaf: float, *, copies_altered: int = 1, ploidy: int = 2) -> float:
    """Tumour fraction from per-site VAF -- the inverse of :func:`vaf_from_tf`.

    For a clonal heterozygous SNV, ``TF = 2 * VAF``.
    """
    if copies_altered <= 0:
        raise ValueError("copies_altered must be positive")
    return vaf * ploidy / copies_altered


@dataclass(frozen=True, slots=True)
class MoleculeParams:
    """Inputs to the molecule budget (BUILD_SPEC 2.1).

    Attributes:
        input_ng: cfDNA input mass in nanograms.
        conversion_efficiency: fraction of input molecules recovered as usable
            consensus families (typical 0.2-0.6). The most commonly
            underestimated parameter and a frequent cause of disappointing
            real-world LoD.
        strand_recovery: additional multiplicative penalty when duplex
            consensus is required (both strands must be recovered). 1.0 for
            single-strand consensus, ~0.5 for duplex. Kept separate from
            ``conversion_efficiency`` so the duplex trade-off (lower error,
            fewer usable molecules) stays explicit.
        pg_per_genome_equivalent: mass per haploid genome equivalent (ng).
            Exposed as a parameter rather than hard-coded in logic.
    """

    input_ng: float
    conversion_efficiency: float
    strand_recovery: float = 1.0
    pg_per_genome_equivalent: float = PG_PER_GENOME_EQUIVALENT

    def __post_init__(self) -> None:
        if self.input_ng <= 0:
            raise ValueError("input_ng must be positive")
        if not 0.0 < self.conversion_efficiency <= 1.0:
            raise ValueError("conversion_efficiency must be in (0, 1]")
        if not 0.0 < self.strand_recovery <= 1.0:
            raise ValueError("strand_recovery must be in (0, 1]")
        if self.pg_per_genome_equivalent <= 0:
            raise ValueError("pg_per_genome_equivalent must be positive")
