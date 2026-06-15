# Snakefile for ceballos_neuropeptide-organization Snakemake pipeline

# Expected final outputs (figures and key results)
expected_figs = [
    # 1_plot_brainmaps.py
    "figs/ADIPOR2_brainmap.pdf",
    "figs/GRPR_brainmap.pdf",
    "figs/CALCRL_brainmap.pdf",
    "figs/CCKBR_brainmap.pdf",
    "figs/EDNRB_brainmap.pdf",
    "figs/NPY1R_brainmap.pdf",
    "figs/GALR1_brainmap.pdf",
    "figs/VIPR1_brainmap.pdf",
    "figs/RXFP1_brainmap.pdf",
    "figs/NTSR1_brainmap.pdf",
    "figs/NPR2_brainmap.pdf",
    "figs/OPRK1_brainmap.pdf",
    "figs/SSTR1_brainmap.pdf",
    "figs/OXTR_brainmap.pdf",
    "figs/GHR_brainmap.pdf",
    # 2_data_showcase.py
    "figs/genes_network_clustermap.pdf",
    "figs/expression_per_structure.pdf",
    "figs/median_expression_trace.pdf",
    # 3_hypothalamus.py
    "figs/hth_nuclei_receptor_expression.pdf",
    "figs/hth_nuclei_receptor_expression_heatmap.pdf",
    # 4_nt_profiling.py
    "figs/colocalization_nt_peptides.pdf",
    "figs/ionotropic_metabotropic_receptors.pdf",
    "figs/kappa_opioid_receptor_comparison.pdf",
    "figs/mu_opioid_receptor_comparison.pdf",
    # 5_pls.py
    "figs/pls_cov_exp.pdf",
    "figs/scores.pdf",
    "figs/pls_cv_score_correlation.pdf",
    "figs/receptor_loadings.pdf",
    "figs/term_loadings.pdf",
    "results/pls_result_Schaefer400_TianS4_HTH.npy",
    # 6-2_summarize_absrel_results.py
    "figs/dn_ds_boxplot.pdf",
    "figs/median_dn_ds_heatmap.pdf",
    # S-1_ahba-rna_gradient.py
    "figs/rnaseq_genes_heatmap.pdf",
    "figs/rnaseq_microarray_receptor_genes_pca_comparison.pdf",
    "figs/rnaseq_microarray_receptor_genes_pls_weights.pdf",
    "figs/rnaseq_microarray_receptor_genes_pls_scores.pdf",
    "figs/rna_microarray_pls_cv.pdf",
    "figs/rna_microarray_pls_scores.pdf",
    # S-1_rna_nt_metabotropic_colocalization.py
    "figs/rna_ionotropic_metabotropic_receptors.pdf",
    # S-2_compare_ahba_hpa.py
    "figs/hpa_ahba_scatterplot.pdf",
    # S-3_data_showcase_cammoun.py
    "figs/genes_heatmap_cammoun.pdf",
    "figs/cammoun_schaefer_genes_pca_loadings_correlation.pdf",
    "figs/cammoun_schaefer_pls_weights_terms.pdf",
    "figs/cammoun_schaefer_pls_weights_receptor.pdf",
    # S-4_sex_differences.py
    "figs/sexdiff_networks.pdf",
    # S-5_blood_perfusion.py
    "figs/endothelin_receptors_cbf.pdf",
    # S-6_feeding_receptors.py
    "figs/feeding_receptor_loadings.pdf",
    # compare_pls_variants.py
    "figs/pls_comparison_cv.pdf",
    "figs/pls_comparison_loadings.pdf",
    # ensemble_pls.py
    "figs/ensemble_comparison_cv.pdf",
    # test_ensemble_vs_gpls_brainsmash.py
    "figs/ensemble_vs_gpls_brainsmash.pdf",
    # ensemble_gpls_hybrids.py
    "figs/gpls_hybrid_ensemble_cv.pdf",
    # ensemble_dynamic_gating.py
    "figs/gpls_dynamic_gating_cv.pdf",
    # plot_all_ensemble_brainmaps.py
    "figs/APLNR_ensemble_brainmap.pdf",
    "figs/Adipose_ensemble_brainmap.pdf",
    "figs/Bombesin_like_ensemble_brainmap.pdf",
    "figs/CCK_gastrin_ensemble_brainmap.pdf",
    "figs/Calcitonin_ensemble_brainmap.pdf",
    "figs/Endothelin_ensemble_brainmap.pdf",
    "figs/F_Y_amide_ensemble_brainmap.pdf",
    "figs/GHR_ensemble_brainmap.pdf",
    "figs/Galanin_ensemble_brainmap.pdf",
    "figs/Glucagon_secretin_ensemble_brainmap.pdf",
    "figs/HCRTR1_ensemble_brainmap.pdf",
    "figs/Insulin_ensemble_brainmap.pdf",
    "figs/Kinin_tensin_ensemble_brainmap.pdf",
    "figs/MCHR1_ensemble_brainmap.pdf",
    "figs/MCHR2_ensemble_brainmap.pdf",
    "figs/Natriuretic_factor_ensemble_brainmap.pdf",
    "figs/Opioid_ensemble_brainmap.pdf",
    "figs/Somatostatin_ensemble_brainmap.pdf",
    "figs/Vasopressin_oxytocin_ensemble_brainmap.pdf",
    "figs/combined_ensemble_brainmaps.pdf",
    "figs/brainsmash_p_values_comparison.pdf",
    "figs/ensemble_accuracy_correlation.pdf",
    "figs/ensemble_accuracy_error.pdf",
    "figs/ensemble_auc_comparison.pdf",
    "figs/ensemble_roc_curves.pdf",
    "figs/ensemble_pr_comparison.pdf",
    "figs/ensemble_pr_curves.pdf"
]

rule all:
    input:
        expected_figs

rule plot_brainmaps:
    input:
        script="scripts/1_plot_brainmaps.py",
        gene_expr="data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv",
        overview="data/receptor_overview.csv"
    output:
        "figs/ADIPOR2_brainmap.pdf",
        "figs/GRPR_brainmap.pdf",
        "figs/CALCRL_brainmap.pdf",
        "figs/CCKBR_brainmap.pdf",
        "figs/EDNRB_brainmap.pdf",
        "figs/NPY1R_brainmap.pdf",
        "figs/GALR1_brainmap.pdf",
        "figs/VIPR1_brainmap.pdf",
        "figs/RXFP1_brainmap.pdf",
        "figs/NTSR1_brainmap.pdf",
        "figs/NPR2_brainmap.pdf",
        "figs/OPRK1_brainmap.pdf",
        "figs/SSTR1_brainmap.pdf",
        "figs/OXTR_brainmap.pdf",
        "figs/GHR_brainmap.pdf"
    shell:
        "python {input.script}"

rule data_showcase:
    input:
        script="scripts/2_data_showcase.py",
        gene_expr="data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv",
        overview="data/receptor_overview.csv",
        lut="data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv"
    output:
        "figs/genes_network_clustermap.pdf",
        "figs/expression_per_structure.pdf",
        "figs/median_expression_trace.pdf"
    shell:
        "python {input.script}"

rule hypothalamus:
    input:
        script="scripts/3_hypothalamus.py",
        receptors="data/receptor_filtered.csv",
        order="results/gene_expression_cluster_order.npy"
    output:
        "figs/hth_nuclei_receptor_expression.pdf",
        "figs/hth_nuclei_receptor_expression_heatmap.pdf"
    shell:
        "python {input.script}"

rule nt_profiling:
    input:
        script="scripts/4_nt_profiling.py",
        gene_expr="data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv",
        densities="data/annotations/nt_receptor_densities_Schaefer400_TianS4_HTH.csv",
        classes="data/annotations/nt_receptor_classes.csv",
        overview="data/receptor_overview.csv"
    output:
        "figs/colocalization_nt_peptides.pdf",
        "figs/ionotropic_metabotropic_receptors.pdf",
        "figs/kappa_opioid_receptor_comparison.pdf",
        "figs/mu_opioid_receptor_comparison.pdf",
        "results/da_nt_peptides_total_dominance.npy",
        "results/da_nt_nulls_peptides_total_dominance_100.npy"
    shell:
        "python {input.script}"

rule pls:
    input:
        script="scripts/5_pls.py",
        gene_expr="data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv",
        lut="data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv",
        neurosynth="data/neurosynth_Schaefer400_TianS4.csv",
        nulls="data/receptor_spatial_nulls_Schaefer400_TianS4_HTH.npy"
    output:
        "figs/pls_cov_exp.pdf",
        "figs/scores.pdf",
        "figs/pls_cv_score_correlation.pdf",
        "figs/receptor_loadings.pdf",
        "figs/term_loadings.pdf",
        "results/pls_result_Schaefer400_TianS4_HTH.npy"
    shell:
        "python {input.script}"

rule positive_selection:
    input:
        script="scripts/6-1_positive_selection.sh",
        align="data/evo/01proseqs_align",
        ordered="data/evo/02proseqs_ordered",
        codalign="data/evo/03codalign",
        timetree="data/evo/timetree_bi.nwk"
    output:
        directory("data/evo/06absrel_results")
    shell:
        "bash {input.script}"

rule summarize_absrel:
    input:
        script="scripts/6-2_summarize_absrel_results.py",
        results="data/evo/06absrel_results"
    output:
        "figs/dn_ds_boxplot.pdf",
        "figs/median_dn_ds_heatmap.pdf"
    shell:
        "python {input.script}"

rule ahba_rna_interpolation:
    input:
        script="scripts/S-1_ahba-rna_interpolation.py",
        lut="data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv"
    output:
        "results/abagen_rnaseq_interpolation.csv",
        "results/abagen_rnaseq_interpolation_normalized.csv"
    shell:
        "python {input.script}"

rule ahba_rna_gradient:
    input:
        script="scripts/S-1_ahba-rna_gradient.py",
        interpolated="results/abagen_rnaseq_interpolation.csv",
        gene_expr="data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv",
        lut="data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv"
    output:
        "figs/rnaseq_genes_heatmap.pdf",
        "figs/rnaseq_microarray_receptor_genes_pca_comparison.pdf",
        "figs/rnaseq_microarray_receptor_genes_pls_weights.pdf",
        "figs/rnaseq_microarray_receptor_genes_pls_scores.pdf",
        "figs/rna_microarray_pls_cv.pdf",
        "figs/rna_microarray_pls_scores.pdf"
    shell:
        "python {input.script}"

rule rna_nt_metabotropic_colocalization:
    input:
        script="scripts/S-1_rna_nt_metabotropic_colocalization.py",
        interpolated="results/abagen_rnaseq_interpolation_normalized.csv",
        receptors="data/receptor_filtered.csv",
        densities="data/annotations/nt_receptor_densities_Schaefer400_TianS4_HTH.csv"
    output:
        "figs/rna_ionotropic_metabotropic_receptors.pdf",
        "results/rna_da_nt_peptides_total_dominance.npy"
    shell:
        "python {input.script}"

rule compare_ahba_hpa:
    input:
        script="scripts/S-2_compare_ahba_hpa.py",
        hpa="data/hpa_whole-brain.tsv",
        genes="data/gene_list.csv",
        destrieux="data/parcellations/destrieux_labels.csv",
        hpa_map="data/parcellations/hpa_destrieux_map.csv",
        abagen="data/abagen_genes_Destrieux.csv"
    output:
        "figs/hpa_ahba_scatterplot.pdf"
    shell:
        "python {input.script}"

rule data_showcase_cammoun:
    input:
        script="scripts/S-3_data_showcase_cammoun.py",
        gene_expr="data/receptor_gene_expression_Cammoun2012_250_7N_Freesurfer_Subcortex.csv",
        overview="data/receptor_overview.csv",
        lut="data/parcellations/Cammoun2012_7N_Freesurfer_Subcortex_LUT.csv",
        neurosynth="data/neurosynth/derivatives/Cammoun2012_7N_Freesurfer_Subcortex_neurosynth.csv",
        order="results/gene_expression_cluster_order.npy"
    output:
        "figs/genes_heatmap_cammoun.pdf",
        "figs/cammoun_schaefer_genes_pca_loadings_correlation.pdf",
        "figs/cammoun_schaefer_pls_weights_terms.pdf",
        "figs/cammoun_schaefer_pls_weights_receptor.pdf"
    shell:
        "python {input.script}"

rule sex_differences:
    input:
        script="scripts/S-4_sex_differences.py",
        lut="data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv",
        gene_expr="data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv"
    output:
        "figs/sexdiff_networks.pdf"
    shell:
        "python {input.script}"

rule blood_perfusion:
    input:
        script="scripts/S-5_blood_perfusion.py",
        gene_expr="data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv",
        nulls="data/receptor_spatial_nulls_Schaefer400_TianS4.npy",
        cbf="data/annotations/cbf_hcpavg_Schaefer400_TianS4.npy"
    output:
        "figs/endothelin_receptors_cbf.pdf"
    shell:
        "python {input.script}"

rule feeding_receptors:
    input:
        script="scripts/S-6_feeding_receptors.py",
        gene_expr="data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv",
        eating="data/eating_term.nii.gz",
        food="data/food_term.nii.gz",
        nulls="data/gene_null_sets_Schaefer400_TianS4_HTH.npy"
    output:
        "figs/feeding_receptor_loadings.pdf"
    shell:
        "python {input.script}"

rule compare_pls_variants:
    input:
        script="scripts/compare_pls_variants.py",
        neurosynth="data/neurosynth_Schaefer400_TianS4.csv",
        receptor_genes="data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv",
        centroids="data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz"
    output:
        cv_plot="figs/pls_comparison_cv.pdf",
        loadings_plot="figs/pls_comparison_loadings.pdf"
    shell:
        "python {input.script}"

rule ensemble_pls:
    input:
        script="scripts/ensemble_pls.py",
        neurosynth="data/neurosynth_Schaefer400_TianS4.csv",
        receptor_genes="data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv"
    output:
        plot="figs/ensemble_comparison_cv.pdf"
    shell:
        "python {input.script}"

rule test_ensemble_vs_gpls_brainsmash:
    input:
        script="scripts/test_ensemble_vs_gpls_brainsmash.py",
        neurosynth="data/neurosynth_Schaefer400_TianS4.csv",
        receptor_genes="data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv",
        centroids="data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz",
        nulls="data/receptor_spatial_nulls_Schaefer400_TianS4_HTH.npy"
    output:
        plot="figs/ensemble_vs_gpls_brainsmash.pdf"
    shell:
        "python {input.script}"

rule ensemble_gpls_hybrids:
    input:
        script="scripts/ensemble_gpls_hybrids.py",
        neurosynth="data/neurosynth_Schaefer400_TianS4.csv",
        receptor_genes="data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv",
        centroids="data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz"
    output:
        plot="figs/gpls_hybrid_ensemble_cv.pdf"
    shell:
        "python {input.script}"

rule ensemble_dynamic_gating:
    input:
        script="scripts/ensemble_dynamic_gating.py",
        neurosynth="data/neurosynth_Schaefer400_TianS4.csv",
        receptor_genes="data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv",
        centroids="data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz"
    output:
        plot="figs/gpls_dynamic_gating_cv.pdf"
    shell:
        "python {input.script}"

rule plot_all_ensemble_brainmaps:
    input:
        script="scripts/plot_all_ensemble_brainmaps.py",
        gene_expr="data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv",
        overview="data/receptor_overview.csv",
        neurosynth="data/neurosynth_Schaefer400_TianS4.csv",
        lut="data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv"
    output:
        "figs/APLNR_ensemble_brainmap.pdf",
        "figs/Adipose_ensemble_brainmap.pdf",
        "figs/Bombesin_like_ensemble_brainmap.pdf",
        "figs/CCK_gastrin_ensemble_brainmap.pdf",
        "figs/Calcitonin_ensemble_brainmap.pdf",
        "figs/Endothelin_ensemble_brainmap.pdf",
        "figs/F_Y_amide_ensemble_brainmap.pdf",
        "figs/GHR_ensemble_brainmap.pdf",
        "figs/Galanin_ensemble_brainmap.pdf",
        "figs/Glucagon_secretin_ensemble_brainmap.pdf",
        "figs/HCRTR1_ensemble_brainmap.pdf",
        "figs/Insulin_ensemble_brainmap.pdf",
        "figs/Kinin_tensin_ensemble_brainmap.pdf",
        "figs/MCHR1_ensemble_brainmap.pdf",
        "figs/MCHR2_ensemble_brainmap.pdf",
        "figs/Natriuretic_factor_ensemble_brainmap.pdf",
        "figs/Opioid_ensemble_brainmap.pdf",
        "figs/Somatostatin_ensemble_brainmap.pdf",
        "figs/Vasopressin_oxytocin_ensemble_brainmap.pdf"
    shell:
        "python {input.script}"

rule plot_combined_brainmaps:
    input:
        script="scripts/plot_combined_brainmaps.py",
        gene_expr="data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv",
        overview="data/receptor_overview.csv",
        neurosynth="data/neurosynth_Schaefer400_TianS4.csv",
        lut="data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv"
    output:
        "figs/combined_ensemble_brainmaps.pdf",
        "figs/combined_ensemble_brainmaps.png"
    shell:
        "python {input.script}"

rule compare_brainsmash_p_values:
    input:
        script="scripts/compare_brainsmash_p_values.py",
        neurosynth="data/neurosynth_Schaefer400_TianS4.csv",
        receptor_genes="data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv",
        centroids="data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz",
        nulls="data/receptor_spatial_nulls_Schaefer400_TianS4_HTH.npy"
    output:
        csv="results/brainsmash_p_values_comparison.csv",
        plot_pdf="figs/brainsmash_p_values_comparison.pdf",
        plot_png="figs/brainsmash_p_values_comparison.png"
    shell:
        "python {input.script}"

rule compare_ensemble_accuracy:
    input:
        script="scripts/compare_ensemble_accuracy.py",
        neurosynth="data/neurosynth_Schaefer400_TianS4.csv",
        receptor_genes="data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv",
        centroids="data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz"
    output:
        csv="results/ensemble_accuracy_comparison.csv",
        plot_corr="figs/ensemble_accuracy_correlation.pdf",
        plot_err="figs/ensemble_accuracy_error.pdf"
    shell:
        "python {input.script}"

rule compare_ensemble_auc:
    input:
        script="scripts/compare_ensemble_auc.py",
        neurosynth="data/neurosynth_Schaefer400_TianS4.csv",
        receptor_genes="data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv",
        centroids="data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz"
    output:
        csv="results/ensemble_auc_comparison.csv",
        plot_auc="figs/ensemble_auc_comparison.pdf",
        plot_roc="figs/ensemble_roc_curves.pdf",
        plot_pr="figs/ensemble_pr_comparison.pdf",
        plot_pr_curve="figs/ensemble_pr_curves.pdf"
    shell:
        "python {input.script}"
