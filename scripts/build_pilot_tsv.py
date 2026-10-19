"""Write pilot_decidability.tsv (one row per pilot benchmark, coded 2026-10-15).

Every number below comes from a script in this folder or a quoted source:
  dti_support.py      BioSNAP / Human / BindingDB release statistics
  beeline_scope.py    -> results/beeline_scope_500_v2.tsv (denominators corrected 2026-10-18)
  dream5_scope.py     -> results/dream5_scope_v2.tsv (positives subtracted 2026-10-18)
  BioSNAP target-side W: release genes in UniProt human reviewed (2,173 / 20,431)
  Human eligibility W: Liu et al. 2015 counts, 1 - (23,392 x 10,757) / (367,142 x 19,342)

Usage: python -I build_pilot_tsv.py OUT.tsv
"""
import csv
import sys

COLS = [
    "benchmark", "domain", "sources_checked", "negative_source",
    "rule_stated_in_paper", "rule_found_in_code_or_release", "text_sufficient",
    "entity", "candidate", "depth", "threshold_T", "declared_universe",
    "decidability", "verification", "support_at_evaluated_unit",
    "W_pilot", "W_target_used", "needed_for_full_W", "notes",
]

ROWS = [
    {
        "benchmark": "BioSNAP (MolTrans; reused by DrugBAN)",
        "domain": "drug-target interaction",
        "sources_checked": "Huang et al. 2021 Bioinformatics (arXiv 2004.11424, Sec. 4.1); github kexinhuang12345/MolTrans dataset/BIOSNAP/full_data; github peizhenbai/DrugBAN datasets/biosnap/full.csv; SNAP ChG-Miner_miner-chem-gene.tsv.gz",
        "negative_source": "constructed (random unlabelled pairs)",
        "rule_stated_in_paper": "\"For negative pairs, we sample from the unseen pairs\" of the ChG-Miner DrugBank graph; balanced 1:1; drugs/targets without SMILES/sequence removed",
        "rule_found_in_code_or_release": "all 13,640 negatives lie inside ChG-Miner drugs x ChG-Miner genes; 0 collide with ChG-Miner positives; negatives per drug 3.02-3.05 in every positive-count bin (uniform pair sampling inside the grid); 1 drug and 61 genes appear only as negatives, all with ChG-Miner degree >= 1",
        "text_sufficient": "no: the grid (entities with >= 1 known interaction) is implicit in 'unseen pairs'; the release decides it",
        "entity": "drug; target protein",
        "candidate": "(drug, protein) pair",
        "depth": "number of DrugBank interactions in ChG-Miner",
        "threshold_T": "1 on both drug and protein",
        "declared_universe": "none declared",
        "decidability": "verified",
        "verification": "release negatives checked pair for pair against the source graph",
        "support_at_evaluated_unit": "violated for any target that includes drugs or proteins with no recorded DrugBank interaction",
        "W_pilot": "0.894 (protein-coverage proxy over the declared Cartesian candidate grid; not a measured-negative mass fraction)",
        "W_target_used": "release drugs x UniProt human reviewed proteome (20,431); 2,173 of 2,182 release genes are human reviewed",
        "needed_for_full_W": "a declared drug universe (e.g. DrugBank small molecules with structures; DrugBank download needs a free account)",
        "notes": "the drug side can only raise exclusion of the Cartesian candidate grid; the final negative denominator and drug universe remain undeclared, so this proxy is not numerically interchangeable with BindingDB W",
    },
    {
        "benchmark": "Human (Liu et al. 2015; Tsubaki et al. 2019 release; reused by TransformerCPI and DrugBAN)",
        "domain": "drug-target interaction",
        "sources_checked": "Liu et al. 2015 Bioinformatics 31:i221 (PMC4765858, Secs 3.2-3.3, 4.1-4.3); Tsubaki et al. 2019 Bioinformatics 35:309; github lifanchen-simm/transformerCPI Human,C.elegans/dataset/human_data.txt; DrugBAN datasets/human/full.csv",
        "negative_source": "constructed (in silico screen)",
        "rule_stated_in_paper": "negative candidates require compound AND protein each to have more than 15 validated/predicted interactions in the DrugBank 4.1 + Matador + STITCH 4.0 assembly (23,392 of 367,142 compounds; 10,757 of 19,342 proteins pass); then score d >= 0.9 and a chi-squared feature filter give 384,916 candidates; Tsubaki 2019 takes negatives 'from the top candidates' at 1:1",
        "rule_found_in_code_or_release": "3,364 positive and 3,364 negative rows; 2,633 unique positive pairs (731 duplicated rows); only 17 of 2,726 compounds appear with both labels; 1,547 compounds appear only as negatives; of compounds with exactly one positive, 98.5% have no negative",
        "text_sufficient": "partly: the eligibility threshold is stated, the subset taken from the 384,916 candidates is not",
        "entity": "compound; protein",
        "candidate": "(compound, protein) pair",
        "depth": "validated + predicted interaction count in the assembly",
        "threshold_T": "16 (> 15) on both compound and protein, plus a similarity screen",
        "declared_universe": "STITCH 4.0 human assembly (367,142 compounds, 19,342 proteins), per the paper",
        "decidability": "partial",
        "verification": "not checkable: depths need the STITCH 4.0 assembly and the release carries SMILES/sequences, not identifiers, so membership in the 384,916 candidates cannot be tested directly",
        "support_at_evaluated_unit": "violated by the stated T > 15 rule; inside the benchmark, compounds with a known positive almost never receive a negative (consistent with the hidden ligand bias reported by Chen et al. 2020)",
        "W_pilot": ">= 0.965 (candidate-pair exclusion bound before the similarity screen; not an identified negative-mass fraction)",
        "W_target_used": "all compound x protein pairs of the paper's human assembly, before the score screen: 1 - (23,392 x 10,757)/(367,142 x 19,342)",
        "needed_for_full_W": "STITCH 4.0 per-entity interaction counts and the d >= 0.9 screen",
        "notes": "the depth counts predicted interactions, so the threshold also conditions on prior predictors",
    },
    {
        "benchmark": "BindingDB (DrugBAN low-bias version, Bai et al. 2021)",
        "domain": "drug-target interaction",
        "sources_checked": "Bai et al. 2023 Nat Mach Intell (arXiv 2208.02194, Methods 'Datasets' and Supplementary S2); DrugBAN datasets/bindingdb/full.csv",
        "negative_source": "measured (IC50 > 10,000 nM)",
        "rule_stated_in_paper": "(i) positive if IC50 < 100 nM, negative if IC50 > 10,000 nM; (ii) \"We removed all DTI pairs where the drugs only had one type of pairs (positive or negative)\"",
        "rule_found_in_code_or_release": "all 14,643 drugs carry >= 1 positive and >= 1 negative; 950 of 2,623 targets appear only as negatives (no target-side filter)",
        "text_sufficient": "yes for rule (ii)",
        "entity": "drug",
        "candidate": "measured (drug, target) pair",
        "depth": "number of measured positives (IC50 < 100 nM) on the drug",
        "threshold_T": "1 on the drug (rule ii); none on the target",
        "declared_universe": "BindingDB IC50 records, implicitly",
        "decidability": "verified",
        "verification": "rule (ii) holds for every drug in the release; rule (i) unchecked (the release omits IC50 values)",
        "support_at_evaluated_unit": "violated for a target of all measured inactive pairs: drugs with no measured active are excluded; same move as a donor threshold, made to remove a shortcut",
        "W_pilot": "0.784492 = 215245 / 274375 in the cleaned, connectivity-separated declared population; 0.784646 = 215441 / 274571 before the additional block-overlap exclusion",
        "W_target_used": "E2 cleaned BindingDB_All 2026-10 measured negatives on the declared proteins, uniform pair weights; primary W uses the population after removing 196 overlapping-connectivity U negatives",
        "needed_for_full_W": "Bai et al.'s BindingDB release date (current dump used instead)",
        "notes": "Historical raw-input values were 0.771 on benchmark proteins and 0.794 over all single-chain targets; these use different cleaning and identities and are not the primary manuscript W. The released benchmark does not retain IC50 values, so its original label construction is not independently verified. Primary measurements use results/hrp_e2 and results/records.tsv",
    },
    {
        "benchmark": "BEELINE scRNA-seq (Pratapa et al. 2020)",
        "domain": "gene regulatory network",
        "sources_checked": "Pratapa et al. 2020 Nat Methods (PMC7098173) and Supplementary Tables (MOESM5, MOESM6); github Murali-group/Beeline at 16f6c68 (2020-02-24) BLEval/computeAUC.py, computeEarlyPrec.py, and utils/generateExpInputs.py (4b302c3); Zenodo 3701939 BEELINE-data.zip and BEELINE-Networks.zip",
        "negative_source": "implicit (all unlisted pairs in the scored grid)",
        "rule_stated_in_paper": "ground truth intersected with the gene set (significant TFs + top 500/1000 variable genes); \"we only considered interactions outgoing from a TF\"; random-predictor precision = network density",
        "rule_found_in_code_or_release": "AUROC/AUPRC grid = ordered pairs among genes touching >= 1 reference edge; EPR grid = reference-edge sources x reference-edge genes; input TFs and genes with no reference edge are dropped",
        "text_sufficient": "no: the paper does not say that genes or TFs without a reference edge leave the candidate set; the code does",
        "entity": "regulator (TF); gene",
        "candidate": "(TF, gene) pair",
        "depth": "number of reference edges (out-degree for EPR)",
        "threshold_T": "1",
        "declared_universe": "none declared beyond the input gene set",
        "decidability": "decidable, unverified",
        "verification": "no negatives are released; reconstruction reproduces #genes for all 7 datasets and #TFs for the 5 mouse datasets exactly; human TF counts do not reproduce with the released human-tfs.csv (hESC 410 vs 330, hHep 448 vs 311); cell-type ChIP densities 0.085-0.578 match the paper's '0.08 to 0.58'",
        "support_at_evaluated_unit": "violated for the TF-to-gene target: TFs without ChIP/reference targets are never scored",
        "W_pilot": "negative mass, TF pairs, TFs+500: cell-type ChIP 0.869-0.972 (hESC 0.937, hHep 0.961, mDC 0.972, mESC 0.921, mHSC-E 0.938, mHSC-GM 0.937, mHSC-L 0.869). The candidate-pair fraction reported earlier is lower (0.766-0.969) because it counted positives and did not intersect the universe's sources with the input TF set",
        "W_target_used": "input TFs x input genes, non-self, uniform, POSITIVES SUBTRACTED so this is a negative-mass share as in Paper 2; the EPR universe is intersected with the input TF set, since 2 (mDC) to 7 (mESC) reference-edge sources lie outside it",
        "needed_for_full_W": "the TF list used for the human datasets",
        "notes": "for ChIP references depth 0 can mean not assayed; corrected TF-to-gene negative-mass values are in results/beeline_scope_500_v2.tsv. They concern the EPR-style TF grid, not the all-gene AUROC grid. Human TF counts remain unreproduced; legacy _500.tsv and _1000.tsv use earlier denominators",
    },
    {
        "benchmark": "DREAM5 network inference (Marbach et al. 2012)",
        "domain": "gene regulatory network",
        "sources_checked": "Marbach et al. 2012 Nat Methods (PMC3512113, Online Methods); Supplementary Data 1 (MOESM584): Evaluation scripts/matlab/edgelist_2_D5C4_gs.m and DREAM5_Challenge4_Evaluation.m, gold_standard_edges_only, gold_standard_complete, NetworkN input data",
        "negative_source": "implicit grid, released explicitly as 'gold_standard_complete'",
        "rule_stated_in_paper": "\"the remaining pairs are considered negatives\"; \"predictions for genes that are not part of the gold standard, i.e., for which no experimentally supported interactions exist, were ignored\"",
        "rule_found_in_code_or_release": "negatives = regulators with >= 1 gold edge x genes touching >= 1 gold edge, minus positives and self-pairs; other predictions removed before scoring",
        "text_sufficient": "partly: gene exclusion stated; regulator exclusion (TFs with no gold target) only in the scorer",
        "entity": "regulator (TF); gene",
        "candidate": "(TF, gene) pair",
        "depth": "number of gold-standard edges",
        "threshold_T": "1",
        "declared_universe": "input TF list x expression genes (the prediction task)",
        "decidability": "verified",
        "verification": "rebuilt negatives equal the released complete gold standard pair for pair in networks 1, 3, 4 (274,380 / 150,214 / 223,262)",
        "support_at_evaluated_unit": "violated for the prediction task's own TF x gene space in E. coli and yeast; nearly full in silico",
        "W_pilot": "negative mass: E. coli 0.9001; S. cerevisiae 0.8871; in silico 0.1322 (candidate-pair fraction 0.8989 / 0.8853 / 0.1305)",
        "W_target_used": "input TFs x expression genes, non-self, uniform, positives subtracted so this is a negative-mass share as in Paper 2",
        "needed_for_full_W": "",
        "notes": "the in silico network is a built-in contrast: when depth reflects the true network rather than study effort, little is excluded; S. aureus (network 2) has no gold standard",
    },
]


def main(out):
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in ROWS:
            assert set(r) == set(COLS), set(COLS) ^ set(r)
            assert not any("\t" in v or "\n" in v for v in r.values())
            w.writerow(r)


if __name__ == "__main__":
    main(sys.argv[1])
