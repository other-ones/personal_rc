python plot_reduction.py \
--prior_path='saved_models/analysis/ti_plot_noprior/cat/embeds.pt' \
--mlm_path='saved_models/analysis/ti_plot_noprior/cat1_mlm0001/embeds.pt' \
--nomlm_path='saved_models/analysis/ti_plot_noprior/cat1_nomlm/embeds.pt' \
--dst_dir='plots_pca/cat1_noprior' --perplexity=0 --reducer=pca \
# --other1_path='saved_models/analysis/ti_plot_noprior/pot_prior/embeds.pt' \
# --other2_path='saved_models/analysis/ti_plot_noprior/vase_prior/embeds.pt' 


python plot_results.py \
--prior_path='saved_models/analysis/ti_plot_noprior/cat/embeds.pt' \
--mlm_path='saved_models/analysis/ti_plot_noprior/cat1_mlm0001/embeds.pt' \
--nomlm_path='saved_models/analysis/ti_plot_noprior/cat1_nomlm/embeds.pt' \
--other1_path='saved_models/analysis/ti_plot_noprior/pot_prior/embeds.pt' \
--other2_path='saved_models/analysis/ti_plot_noprior/vase_prior/embeds.pt' \
--dst_dir='plots/cat1_noprior'

python plot_reduction.py \
--prior_path='saved_models/analysis2/ti_plot_noprior/cat/embeds.pt' \
--mlm_path='saved_models/analysis2/ti_plot_noprior/cat1_mlm0001/embeds.pt' \
--nomlm_path='saved_models/analysis2/ti_plot_noprior/cat1_nomlm/embeds.pt' \
--dst_dir='plots_pca/cat1_noprior' --perplexity=0 --reducer=pca \
# --other1_path='saved_models/analysis/ti_plot_noprior/pot_prior/embeds.pt' \
# --other2_path='saved_models/analysis/ti_plot_noprior/vase_prior/embeds.pt' 