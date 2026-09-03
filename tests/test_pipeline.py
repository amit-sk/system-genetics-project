"""Sanity tests for the analysis pipeline of system_genetics_project.ipynb.

The functions are loaded from the notebook itself (imports, function definitions and
constants only - none of the heavy pipeline cells are executed), and each is checked
on small synthetic inputs whose correct output is known.

Run from the project root:  python3 tests/test_pipeline.py
Not part of the assignment; not referenced by the report.
"""
import ast
import json
import sys
import tempfile
import traceback
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from scipy.stats import linregress, norm, poisson

NOTEBOOK = 'system_genetics_project.ipynb'


def load_notebook_functions(path=NOTEBOOK):
    """Execute only the imports, function definitions and UPPER_CASE constants of the notebook's code cells."""
    notebook = json.load(open(path))
    namespace = {}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            is_constant = (isinstance(node, ast.Assign)
                           and all(isinstance(t, ast.Name) and t.id == t.id.upper() for t in node.targets))
            if isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom)) or is_constant:
                try:
                    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), namespace)
                except Exception:            # constants that depend on pipeline state (e.g. GENOTYPE_MATRIX)
                    pass
    return namespace


ns = load_notebook_functions()
rng = np.random.default_rng(0)


# ---------------------------------------------------------------- Section 2

def test_f_test_matches_scipy():
    """The vectorised F-test P-value equals scipy.stats.linregress on informative SNPs."""
    trait = rng.normal(size=30)
    genotype_matrix = rng.choice([0.0, 1.0, 2.0], size=(5, 30))
    p_values, n_strains = ns['f_test_all_snps'](genotype_matrix, trait)
    for snp in range(5):
        expected = linregress(genotype_matrix[snp], trait).pvalue
        assert np.isclose(p_values[snp], expected), (p_values[snp], expected)
        assert n_strains[snp] == 30


def test_f_test_detects_true_association():
    """A SNP that determines the trait gets a tiny P-value; an unrelated SNP does not."""
    causal = rng.choice([0.0, 2.0], size=40)
    unrelated = rng.choice([0.0, 2.0], size=40)
    trait = 3 * causal + rng.normal(scale=0.1, size=40)
    p_values, _ = ns['f_test_all_snps'](np.vstack([causal, unrelated]), trait)
    assert p_values[0] < 1e-20
    assert p_values[1] > 1e-3


def test_f_test_monomorphic_snps_are_nan():
    """Monomorphic SNPs (no genotype variance) give NaN."""
    trait = rng.normal(size=20)
    monomorphic = np.zeros(20)
    p_values, _ = ns['f_test_all_snps'](monomorphic[None, :], trait)
    assert np.isnan(p_values).all()


def test_f_test_ignores_unknown_genotypes():
    """NaN genotype entries are excluded per SNP, matching scipy on the informative subset."""
    trait = rng.normal(size=25)
    genotype = rng.choice([0.0, 2.0], size=25)
    genotype[[3, 7, 11]] = np.nan
    p_values, n_strains = ns['f_test_all_snps'](genotype[None, :], trait)
    known = ~np.isnan(genotype)
    assert n_strains[0] == known.sum()
    assert np.isclose(p_values[0], linregress(genotype[known], trait[known]).pvalue)


def test_bh_fdr_hand_computed():
    """BH-adjusted values match a hand-computed example; NaNs pass through."""
    p_values = np.array([0.01, 0.04, 0.03, 0.005, np.nan])
    expected = np.array([0.02, 0.04, 0.04, 0.02, np.nan])       # m=4, q_i = min_{j>=i} p_(j) * m / j
    q_values = ns['bh_fdr'](p_values)
    assert np.allclose(q_values[:4], expected[:4]) and np.isnan(q_values[4])
    assert (q_values[:4] >= p_values[:4]).all()


def test_encode_genotypes():
    table = pd.DataFrame({'S1': ['B', 'H'], 'S2': ['D', 'U'], 'S3': ['H', 'B']})
    encoded = ns['encode_genotypes'](table, ['S1', 'S2', 'S3'])
    assert np.array_equal(encoded[0], [0, 2, 1])
    assert encoded[1, 0] == 1 and np.isnan(encoded[1, 1]) and encoded[1, 2] == 0


def test_get_phenotype_vector():
    table = pd.DataFrame({'ID_FOR_CHECK': [7], 'Phenotype': ['toy trait'], 'S1': [1.5], 'S3': [2.5]})
    values, name = ns['get_phenotype_vector'](table, 7, ['S1', 'S2', 'S3'])
    assert name == 'toy trait'
    assert values[0] == 1.5 and np.isnan(values[1]) and values[2] == 2.5


def _toy_qtl_world(trait_values, n_snps=30, n_strains=20, seed=5):
    """Install a small synthetic genotype/phenotype world into the notebook namespace."""
    generator = np.random.default_rng(seed)
    strains = [f'S{i}' for i in range(n_strains)]
    calls = generator.choice(['B', 'D'], size=(n_snps, n_strains))
    ns['STRAINS'] = strains
    ns['genotypes'] = pd.DataFrame({'Locus': [f'rs{i}' for i in range(n_snps)],
                                    'Chr': [1] * n_snps, 'Position': np.arange(n_snps) * 1000,
                                    **{s: calls[:, j] for j, s in enumerate(strains)}})
    ns['GENOTYPE_MATRIX'] = ns['encode_genotypes'](ns['genotypes'], strains)
    ns['phenotypes'] = pd.DataFrame({'ID_FOR_CHECK': [1], 'Phenotype': ['toy'],
                                     **{s: [v] for s, v in zip(strains, trait_values)}})
    ns['RESULTS_DIR'] = tempfile.mkdtemp()
    return strains


def test_run_qtl_pipeline_fallback_and_significant():
    """No significant SNP -> the 15 lowest-P SNPs are written; a planted QTL -> the significant SNPs are written."""
    generator = np.random.default_rng(6)
    _toy_qtl_world(generator.normal(size=20))                    # pure noise phenotype
    results = ns['run_qtl_pipeline'](1)
    written = pd.read_csv(f"{ns['RESULTS_DIR']}/qtl_1.csv")
    assert not results['significant'].any()
    assert len(written) == ns['N_FALLBACK_SNPS']
    assert list(written['adjusted_p_value']) == sorted(written['adjusted_p_value'])

    strains = _toy_qtl_world(np.zeros(20))                       # phenotype driven by SNP 0
    causal = ns['GENOTYPE_MATRIX'][0]
    trait = 5 * causal + np.random.default_rng(7).normal(scale=0.1, size=20)
    ns['phenotypes'].loc[0, strains] = trait
    results = ns['run_qtl_pipeline'](1)
    written = pd.read_csv(f"{ns['RESULTS_DIR']}/qtl_1.csv")
    assert results['significant'].any()
    assert len(written) == int(results['significant'].sum())
    assert results.loc[0, 'Locus'] == 'rs0'                      # the planted SNP has the lowest P


def test_expression_matrix_parsing():
    """GSM tables and 'strain:' characteristics are assembled into a probes x arrays matrix."""
    def fake_sample(strain, values):
        return SimpleNamespace(metadata={'characteristics_ch1': ['tissue: Toy', f'strain: {strain}', 'Sex: F']},
                               table=pd.DataFrame({'ID_REF': [10, 20], 'VALUE': values}))

    series = SimpleNamespace(gsms={'gsm1': fake_sample('BXD1', [1.0, 2.0]),
                                   'gsm2': fake_sample('BXD2', [3.0, 4.0])})
    expression, strain_per_array = ns['expression_matrix'](series)
    assert list(expression.index) == ['10', '20']                # probe ids as strings
    assert expression.loc['10', 'gsm1'] == 1.0 and expression.loc['20', 'gsm2'] == 4.0
    assert strain_per_array.to_dict() == {'gsm1': 'BXD1', 'gsm2': 'BXD2'}


def test_keep_genotyped_strains():
    ns['STRAINS'] = ['BXD1', 'BXD2']
    expression = pd.DataFrame({'a1': [1.0], 'a2': [2.0], 'a3': [3.0]}, index=['g'])
    strain_per_array = pd.Series({'a1': 'BXD1', 'a2': 'C57BL/6J', 'a3': 'BXD2'})
    kept, kept_arrays = ns['keep_genotyped_strains'](expression, strain_per_array)
    assert list(kept.columns) == ['a1', 'a3']                    # the parental strain's array is dropped
    assert set(kept_arrays) == {'BXD1', 'BXD2'}


def test_eqtl_counts():
    pairs = pd.DataFrame({'snp': ['s1', 's1', 's2', 's3'],
                          'gene': ['g1', 'g2', 'g3', 'g1'],
                          'type': ['cis', 'trans', 'trans', 'cis']})
    counts = ns['eqtl_counts'](pairs)
    assert counts['eQTLs (SNP-gene pairs)'] == 4
    assert counts['  cis'] == 2 and counts['  trans'] == 2 and counts['  unknown (gene without position)'] == 0
    assert counts['distinct eQTL SNPs'] == 3
    assert counts['  cis only'] == 1 and counts['  trans only'] == 1 and counts['  both cis and trans'] == 1
    assert counts['genes with an eQTL'] == 3
    assert counts['  with a cis eQTL'] == 1 and counts['  with a trans eQTL'] == 2


# ---------------------------------------------------------------- Section 3.1

def test_chromosome_number():
    f = ns['chromosome_number']
    assert f('chr5') == 5 and f('chr19') == 19 and f('chrX') == ns['X_CHROMOSOME']
    assert np.isnan(f('chrUn')) and np.isnan(f('chr4_random')) and np.isnan(f('chrY'))


def test_annotate_gpl6246_symbol_parsing():
    table = pd.DataFrame({
        'ID': [1, 2, 3, 4],
        'category': ['main', 'main', 'main', 'control->bgp'],
        'gene_assignment': ['NM_1 // Abc1 // desc // 1 A // 11 /// NM_2 // Xyz // d // 1 // 12',
                            '---', np.nan, 'NM_9 // Ctrl // d // 1 // 9'],
        'seqname': ['chr2', 'chrX', 'chrUn', 'chr1'],
        'RANGE_START': [100, 200, 300, 400]})
    annotation = ns['annotate_gpl6246'](table)
    assert list(annotation.index) == ['1', '2', '3']            # control row excluded
    assert annotation.loc['1', 'symbol'] == 'Abc1'              # first gene of the primary entry
    assert annotation.loc['2', 'symbol'] is None and annotation.loc['3', 'symbol'] is None
    assert annotation.loc['1', 'chr'] == 2 and annotation.loc['2', 'chr'] == ns['X_CHROMOSOME']


def test_annotate_gpl6466_location_parsing():
    table = pd.DataFrame({
        'ID': ['a', 'b', 'c'],
        'CONTROL_TYPE': ['FALSE', 'FALSE', 'pos'],
        'GENE_SYMBOL': ['Gene1', np.nan, 'Spike'],
        'CHROMOSOMAL_LOCATION': ['chr3:1000-1060', 'chrX:500-560', 'chr1:1-60']})
    annotation = ns['annotate_gpl6466'](table)
    assert list(annotation.index) == ['a', 'b']                 # control probe excluded
    assert annotation.loc['a', 'chr'] == 3 and annotation.loc['a', 'pos'] == 1000
    assert annotation.loc['b', 'chr'] == ns['X_CHROMOSOME']


def test_liftover_known_coordinate():
    """Actb start, mm10 chr5:142,903,115 -> mm9 chr5:143,664,794 (uses the local chain file)."""
    annotation = pd.DataFrame({'symbol': ['Actb'], 'chr': [5.0], 'pos': [142903115.0]}, index=['p'])
    lifted = ns['liftover_mm10_to_mm9'](annotation)
    assert lifted.loc['p', 'pos'] == 143664794


def test_filtering_steps():
    expression = pd.DataFrame(
        {'a1': [1.0, 5.0, 5.0, 9.0], 'a2': [1.1, 5.2, 6.0, 1.0]},
        index=['p1', 'p2', 'p3', 'p4'])
    annotation = pd.DataFrame({'symbol': ['G1', None, 'G2', 'G2']}, index=expression.index)

    with_symbol = ns['drop_without_symbol'](expression, annotation)
    assert list(with_symbol.index) == ['p1', 'p3', 'p4']

    kept, max_per_probe, threshold = ns['drop_low_max'](expression, 0.5)
    assert threshold == expression.max(axis=1).quantile(0.5)
    assert (kept.max(axis=1) >= threshold).all()

    kept, variance_per_probe, threshold = ns['drop_low_variance'](expression, 0.25)
    assert (kept.var(axis=1) >= threshold).all()

    genes, gene_annotation = ns['one_probe_per_gene'](expression.loc[['p3', 'p4']],
                                                      annotation.loc[['p3', 'p4']])
    assert list(genes.index) == ['G2']
    assert genes.loc['G2', 'a1'] == 9.0                        # p4 has the higher variance


def test_average_per_strain():
    expression = pd.DataFrame({'a1': [1.0], 'a2': [3.0], 'a3': [10.0]}, index=['g'])
    strain_per_array = pd.Series({'a1': 'S1', 'a2': 'S1', 'a3': 'S2'})
    means = ns['average_per_strain'](expression, strain_per_array)
    assert means.loc['g', 'S1'] == 2.0 and means.loc['g', 'S2'] == 10.0


def test_representative_snps():
    ns['STRAINS'] = ['S1', 'S2', 'S3']
    ns['genotypes'] = pd.DataFrame({
        'Locus': ['r1', 'r2', 'r3', 'r4', 'r5'],
        'Chr': [1, 1, 1, 1, 2],
        'Position': [100, 200, 300, 400, 100],
        'S1': ['B', 'B', 'B', 'B', 'B'],
        'S2': ['B', 'B', 'D', 'D', 'D'],
        'S3': ['D', 'D', 'D', 'U', 'D']})
    snp_table, snp_matrix, strain_order, representative_of = ns['representative_snps'](['S1', 'S2', 'S3'])
    assert list(snp_table['Locus']) == ['r1', 'r3', 'r4', 'r5']  # r2 identical to r1; U distinct; new chromosome resets
    assert snp_table.loc[0, 'RunEnd'] == 200 and snp_table.loc[0, 'RunSize'] == 2
    assert representative_of['r2'] == 'r1' and representative_of['r5'] == 'r5'
    assert np.array_equal(snp_matrix[0], [0, 0, 2])


# ---------------------------------------------------------------- Sections 3.2-3.3

def test_eqtl_scan_and_select():
    # deterministic, mutually orthogonal patterns: no assertion depends on chance correlations
    generator = np.random.default_rng(42)
    causal_snp = np.array([0.0, 2.0] * 10)                      # alternates every strain
    other_snp = np.array([0.0, 0.0, 2.0, 2.0] * 5)              # alternates every 2 -> orthogonal to causal_snp
    unrelated_gene = np.array([1.0, -1.0, -1.0, 1.0] * 5)       # orthogonal to both SNPs
    strains = [f'S{i}' for i in range(20)]
    expression = pd.DataFrame(
        {s: v for s, v in zip(strains, np.vstack([5 * causal_snp + generator.normal(scale=0.1, size=20),
                                                  unrelated_gene + generator.normal(scale=0.001, size=20)]).T)},
        index=['GeneA', 'GeneB'])
    p_values = ns['eqtl_scan'](expression, np.vstack([causal_snp, other_snp]), ['snpC', 'snpO'])
    assert p_values.loc['GeneA', 'snpC'] < 1e-20 and p_values.loc['GeneB', 'snpC'] > 1e-3

    q_values = pd.DataFrame(ns['bh_fdr'](p_values.to_numpy().ravel()).reshape(p_values.shape),
                            index=p_values.index, columns=p_values.columns)
    snp_table = pd.DataFrame({'Locus': ['snpC', 'snpO'], 'Chr': [1, 2], 'Position': [1000, 2000],
                              'RunEnd': [1500, 2000], 'RunSize': [2, 1]})
    annotation = pd.DataFrame({'symbol': ['GeneA', 'GeneB'], 'chr': [1.0, 2.0], 'pos': [1200.0, 5000.0]},
                              index=['GeneA', 'GeneB'])
    eqtls = ns['select_eqtls'](p_values, q_values, snp_table, annotation, alpha=0.05)
    top = eqtls.iloc[0]                                         # sorted by P-value
    assert (top.gene, top.snp, top.snp_pos, top.run_end) == ('GeneA', 'snpC', 1000, 1500)
    assert 'GeneB' not in set(eqtls['gene'])                    # the unassociated gene has no eQTL
    assert (eqtls['q_value'] < 0.05).all()


def test_classify_cis_trans():
    window = ns['CIS_WINDOW']
    pairs = pd.DataFrame({
        'snp_chr': [1, 1, 1, 1, 2],
        'snp_pos': [10_000_000] * 5,
        'run_end': [12_000_000] * 5,
        'gene_chr': [1, 1, 1, 1, 1],
        'gene_pos': [11_000_000,                    # inside the run           -> cis, distance 0
                     10_000_000 - window,           # exactly at the window    -> cis
                     10_000_000 - window - 1,       # just beyond              -> trans
                     np.nan,                        # no position              -> unknown
                     11_000_000]})                  # other chromosome         -> trans
    classified = ns['classify_cis_trans'](pairs)
    assert list(classified['type']) == ['cis', 'cis', 'trans', 'unknown', 'trans']
    assert classified.loc[0, 'distance_bp'] == 0 and classified.loc[1, 'distance_bp'] == window


def test_hotspot_threshold_and_regions():
    counts = pd.Series(rng.poisson(3.0, size=2000).astype(float))
    threshold = ns['hotspot_threshold'](counts)
    lam = counts.mean()
    assert poisson.sf(threshold, lam) < ns['HOTSPOT_P'] <= poisson.sf(threshold - 1, lam)

    hotspot_snps = pd.DataFrame({
        'Locus': ['h1', 'h2', 'h3', 'h4'],
        'Chr': [1, 1, 1, 2],
        'Position': [1_000_000, 3_000_000, 20_000_000, 1_000_000],   # h1+h2 within 5 Mb -> one region
        'trans_genes': [10, 25, 12, 11]})
    regions = ns['hotspot_regions'](hotspot_snps)
    assert len(regions) == 3
    top = regions.iloc[0]
    assert (top.chr, top.start, top.end, top.n_snps, top.max_trans_genes, top.peak_snp) == (1, 1_000_000, 3_000_000, 2, 25, 'h2')


# ---------------------------------------------------------------- Section 5

def test_gaussian_log_likelihood_matches_scipy():
    values = rng.normal(size=15)
    assert np.isclose(ns['gaussian_log_likelihood'](values, 0.3, 2.0),
                      norm.logpdf(values, loc=0.3, scale=np.sqrt(2.0)).sum())


def test_fit_genotype():
    log_likelihood, n_parameters = ns['fit_genotype'](np.array([0, 0, 2.0]))
    assert np.isclose(log_likelihood, 2 * np.log(2 / 3) + np.log(1 / 3)) and n_parameters == 1


def test_fit_given_genotype_and_continuous():
    genotype = np.array([0.0] * 10 + [2.0] * 10)
    values = rng.normal(size=20) + genotype
    log_likelihood, n_parameters = ns['fit_given_genotype'](values, genotype)
    expected = sum(norm.logpdf(values[genotype == g],
                               values[genotype == g].mean(),
                               values[genotype == g].std()).sum() for g in (0.0, 2.0))
    assert np.isclose(log_likelihood, expected) and n_parameters == 4

    predictor = rng.normal(size=20)
    values = 2 * predictor + 1 + rng.normal(scale=0.5, size=20)
    log_likelihood, n_parameters = ns['fit_given_continuous'](values, predictor)
    slope, intercept = np.polyfit(predictor, values, 1)
    residuals = values - (slope * predictor + intercept)
    assert np.isclose(log_likelihood, norm.logpdf(residuals, 0, residuals.std()).sum()) and n_parameters == 3


def test_drop_small_genotype_groups():
    genotype = np.array([0.0] * 5 + [1.0] * 2 + [2.0] * 5)
    kept_genotype, kept_r, kept_c = ns['drop_small_genotype_groups'](genotype, genotype * 10, genotype * 100)
    assert 1.0 not in kept_genotype and len(kept_genotype) == 10
    assert np.array_equal(kept_r, kept_genotype * 10)


def _synthetic_triplet(truth, n=300, seed=1):
    """Generate (genotype, expression, phenotype) from a known causal model."""
    generator = np.random.default_rng(seed)
    L = generator.choice([0.0, 2.0], size=n)
    noise = lambda: generator.normal(scale=1.0, size=n)
    if truth == 'M1':
        R = 2 * L + noise(); C = 3 * R + noise()
    elif truth == 'M2':
        C = 2 * L + noise(); R = 3 * C + noise()
    else:                                   # M3: independent effects
        R = 2 * L + noise(); C = -1.5 * L + noise()
    return L, R, C


def test_mediation_recovers_true_model():
    for truth in ['M1', 'M2', 'M3']:
        fits = ns['mediation_test'](*_synthetic_triplet(truth))
        best = min(fits, key=lambda name: fits[name]['AIC'])
        assert best.startswith(truth), (truth, best, {k: round(v['AIC'], 1) for k, v in fits.items()})


def test_permute_within_genotype_groups_preserves_groups():
    genotype = np.array([0.0] * 6 + [2.0] * 6)
    values = np.arange(12.0)
    permuted = ns['permute_within_genotype_groups'](values, genotype, np.random.default_rng(2))
    for group in (0.0, 2.0):
        assert sorted(permuted[genotype == group]) == sorted(values[genotype == group])
    assert not np.array_equal(permuted, values)


def test_permutation_test_significance():
    """A true mediation triplet gets a small P-value; an independent-effects triplet does not."""
    _, p_mediated = ns['permutation_test'](*_synthetic_triplet('M1', n=100), n_permutations=200, seed=3)
    _, p_independent = ns['permutation_test'](*_synthetic_triplet('M3', n=100), n_permutations=200, seed=3)
    assert p_mediated <= 0.01
    assert p_independent > 0.1


def test_collect_triplets_window():
    ns['qtl_results'] = {99: pd.DataFrame({'Locus': ['q1'], 'Chr': [1], 'Position': [1_000_000],
                                           'significant': [True]})}
    window = ns['NEARBY_WINDOW']
    ns['eqtls'] = {'toy': pd.DataFrame({
        'gene': ['Near', 'Far', 'OtherChr'],
        'gene_chr': [1.0, 1.0, 2.0], 'gene_pos': [1.0, 1.0, 1.0],
        'snp': ['s1', 's2', 's3'], 'snp_chr': [1, 1, 2],
        'snp_pos': [1_000_000 + window, 1_000_000 + window + 1, 1_000_000],
        'run_end': [1_000_000 + window, 1_000_000 + window + 1, 1_000_000],
        'type': ['cis', 'cis', 'cis'], 'p_value': [1e-9, 1e-9, 1e-9], 'q_value': [1e-3, 1e-3, 1e-3]})}
    triplets = ns['collect_triplets'](99, 'toy')
    assert list(triplets['gene']) == ['Near']                   # exactly at the window; the others excluded


# ---------------------------------------------------------------- runner

def main():
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith('test_') and callable(obj)]
    failed = 0
    for name, test in tests:
        try:
            test()
            print(f'PASS  {name}')
        except Exception:
            failed += 1
            print(f'FAIL  {name}')
            traceback.print_exc()
    print(f'\n{len(tests) - failed} / {len(tests)} tests passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
