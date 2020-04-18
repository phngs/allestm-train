ls ~/proj/allestm-train/data | grep fasta | cut -f 1 -d "." | xargs -P 8 -I {} python store_other_methods.py ~/proj/allestm-train/data.db spot1d_phi continuous.PhiAngles ~/proj/allestm-train/data/{}.fasta ~/proj/allestm-train/data/{}.spot1d
ls ~/proj/allestm-train/data | grep fasta | cut -f 1 -d "." | xargs -P 8 -I {} python store_other_methods.py ~/proj/allestm-train/data.db spot1d_psi continuous.PsiAngles ~/proj/allestm-train/data/{}.fasta ~/proj/allestm-train/data/{}.spot1d

ls ~/proj/allestm-train/data | grep fasta | cut -f 1 -d "." | xargs -P 8 -I {} python store_other_methods.py ~/proj/allestm-train/data.db spot1d_sec categorical.SecStruc ~/proj/allestm-train/data/{}.fasta ~/proj/allestm-train/data/{}.spot1d

ls ~/proj/allestm-train/data | grep fasta | cut -f 1 -d "." | xargs -P 8 -I {} python store_other_methods.py ~/proj/allestm-train/data.db spot1d_rsa continuous.RsaComplex ~/proj/allestm-train/data/{}.fasta ~/proj/allestm-train/data/{}.spot1d
ls ~/proj/allestm-train/data | grep fasta | cut -f 1 -d "." | xargs -P 8 -I {} python store_other_methods.py ~/proj/allestm-train/data.db spot1d_rsa continuous.RsaChain ~/proj/allestm-train/data/{}.fasta ~/proj/allestm-train/data/{}.spot1d
