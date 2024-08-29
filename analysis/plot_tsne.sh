python plot_reduction.py \
--prior_path='saved_models/analysis2/ti_plot_wprior/cat/prior_embeds.pt' \
--mlm_path='saved_models/analysis2/ti_plot_wprior/cat1_mlm0001/learned_embeds_incprior_1_target_prior.pt' \
--nomlm_path='saved_models/analysis2/ti_plot_wprior/cat1_nomlm/learned_embeds_incprior_1_target_prior.pt' \
--dst_dir='plots_tsne_new/cat1_prior' --perplexity=15 --reducer=tsne 
# --other1_path='saved_models/analysis/ti_plot_noprior/pot_prior/embeds.pt' \
# --other2_path='saved_models/analysis/ti_plot_noprior/vase_prior/embeds.pt' 
# --other1_path='saved_models/analysis/ti_plot_noprior/dog/embeds.pt' \
# --other2_path='saved_models/analysis/ti_plot_noprior/dog6_mlm0001/embeds.pt' 


python plot_results.py \
--prior_path='saved_models/analysis/ti_plot_noprior/cat/embeds.pt' \
--mlm_path='saved_models/analysis/ti_plot_noprior/cat1_mlm0001/embeds.pt' \
--nomlm_path='saved_models/analysis/ti_plot_noprior/cat1_nomlm/embeds.pt' \
--other1_path='saved_models/analysis/ti_plot_noprior/pot_prior/embeds.pt' \
--other2_path='saved_models/analysis/ti_plot_noprior/vase_prior/embeds.pt' \
--dst_dir='plots/cat1_noprior'

python plot_reduction.py \
--prior_path='saved_models/analysis2/ti_plot_noprior/cat/prior_embeds.pt' \
--mlm_path='saved_models/analysis2/ti_plot_noprior/cat1_mlm0001/learned_embeds_incprior_0_target_keyword.pt' \
--nomlm_path='saved_models/analysis2/ti_plot_noprior/cat1_nomlm/learned_embeds_incprior_0_target_keyword.pt' \
--dst_dir='plots_tsne4/cat1_noprior' --perplexity=25 --reducer=tsne \
--other1_path='saved_models/analysis/ti_plot_noprior/pot_prior/embeds.pt' \
--other2_path='saved_models/analysis/ti_plot_noprior/vase_prior/embeds.pt'  