#!/usr/bin/env bash
# Download stuff alpha_bitopic and alpha_polytopic
# Download up topology info

mkdir data
cat alpha_* | xargs -P 4 -I {} wget -O data/{}.opm http://opm.phar.umich.edu/pdb/{}.pdb
cat alpha_* | xargs -P 4 -I {} wget -O data/{}.pdb https://files.rcsb.org/download/{}.pdb
rm data/*.1

python prepare_data.py ~/databases/depth/data/ ~/databases/depth/data.db ~/databases/depth/alpha_*

python create_datasets.py ~/databases/depth/data.db depth


# HHblits DB copy
ssh lofn cp -rvu /home/users/hoenigschmid/databases/hhblits/uniclust30_2017_10/ /localscratch/
ssh gefjun cp -rvu /home/users/hoenigschmid/databases/hhblits/uniclust30_2017_10/ /localscratch/

# HHblits for features
sqlite3 ~/databases/depth/data.db "select distinct id from datasets where name='depth'" | xargs -I {} qsub -q lofn.q -l m_core=20 -b y /home/users/hoenigschmid/miniconda3/envs/depth/bin/python /home/users/hoenigschmid/projects/depth/run_hhblits.py -c 20 {} /home/users/hoenigschmid/databases/depth/data.db

### Other tools
sqlite3 ~/databases/depth/data.db "select distinct id, sequence from proteins join datasets using (id) where name='depth'" | sed -e 's/\(......\)|\(.*\)/>\1\n\2/' > ~/databases/depth/all.fasta
~/scripts/splitFasta.pl ~/databases/depth/all.fasta ~/databases/depth/data/

sqlite3 ~/databases/depth/data.db "select distinct id from datasets where name='depth'" | xargs -I {} bash -c "sqlite3 ~/databases/depth/data.db 'select msa from alignments where id=\"{}\"' > ~/databases/depth/data/{}.a3m"
find /home/users/hoenigschmid/databases/depth/data/ -name "*.a3m" | cut -f 1 -d "." | xargs -I {} ~/scripts/ifne.pl {}.fas ~/scripts/reformat.pl a3m fas {}.a3m {}.fas

# Update alignments in db.
sqlite3 ~/databases/depth/data.db "select distinct id from datasets where name='depth'" | xargs -I {} python store_msa.py ~/databases/depth/data.db {} ~/databases/depth/data/{}.fas

# Create clean alignments
ls ~/databases/depth/data | grep "fasta$" | cut -f 1 -d "." | xargs -I {} python ~/scripts/fas2validaa.py {}.fas {}.cleanfas

# Polyphobius
ls ~/databases/depth/data | grep "fasta$" | cut -f 1 -d "." | xargs -I {} bash -c "~/programs/phobius/jphobius -poly ~/databases/depth/data/{}.cleanfas > ~/databases/depth/data/{}.polyphobius"

# Be in root database dir!
# ANGLOR
ls data | grep fasta | cut -f 1 -d "." | xargs -I {} mkdir data/{}
ls data | grep fasta | cut -f 1 -d "." | xargs -I {} cp data/{}.fasta data/{}/seq.txt
ls data | grep fasta | cut -f 1 -d "." | xargs -P 4 -I {} ./tools/ANGLOR_source/ANGLOR/ANGLOR.pl {}
ls data | grep fasta | cut -f 1 -d "." | xargs -I {} cp data/{}/phi.txt data/{}.anglorphi
ls data | grep fasta | cut -f 1 -d "." | xargs -I {} cp data/{}/psi.txt data/{}.anglorpsi
ls data | grep fasta | cut -f 1 -d "." | xargs -I {} rm -rf data/{}

# PredyFlexy
ls data | grep fasta | cut -f 1 -d "." | xargs -I {} mkdir data/{}
cd tools/PredyFlexy/
ls ../../data | grep fasta | cut -f 1 -d "." | xargs -I {} python2.7 pred.py -f ../../data/{}.fasta -D ../../data/{} --confidence --flex
cd ../../
ls data | grep fasta | cut -f 1 -d "." | xargs -I {} cp data/{}/PRED-FINAL/predictions.txt data/{}.predyflexy
ls data | grep fasta | cut -f 1 -d "." | xargs -I {} rm -rf data/{}

# memsat-svm
ls data | grep fasta | cut -f 1 -d "." | xargs -I {} mkdir data/{}
ls data | grep fasta | cut -f 1 -d "." | xargs -I {} mkdir data/{}/input
ls data | grep fasta | cut -f 1 -d "." | xargs -I {} mkdir data/{}/output
cd tools/memsat-svm
ls ../../data | grep fasta | cut -f 1 -d "." | xargs -I {} ./run_memsat-svm.pl ../../data/{}.fasta -d swissprot/swissprot -i ../../data/{}/input/ -j ../../data/{}/output
cd ../../
ls data | grep fasta | cut -f 1 -d "." | xargs -I {} cp data/{}/output/{}.memsat_svm data/
ls data | grep fasta | cut -f 1 -d "." | xargs -I {} rm -rf data/{}

# SpineX
mkdir /localscratch/nr
cp -uv tools/spineXpublic/nr.* /localscratch/nr/
ls data | grep fasta | cut -f 1 -d "." | xargs -I {} mkdir data/{}
ls data | grep fasta | cut -f 1 -d "." | xargs -I {} cp data/{}.fasta data/{}/{}
ls data | grep fasta | cut -f 1 -d "." | xargs -I {} bash -c "echo {} > data/{}/id"
export spineXblast=/home/software/share/ncbi-blast2/2.2.21/x64/
ls data | grep fasta | cut -f 1 -d "." | xargs -P 4 -I {} ~/scripts/ifne.pl data/{}/tmp ./tools/spineXpublic/spX.pl data/{}/id data/{}/ data/{}/
ls data | grep fasta | cut -f 1 -d "." | xargs -I {} cp data/{}/{}.spXout data/
ls data | grep fasta | cut -f 1 -d "." | xargs -I {} rm -rf data/{}
rm -rf spxtemp*
rm -rf /localscratch/nr

# profbval
# -> VM

# Insert predictions into db.
ls ~/databases/depth/data | grep fasta | cut -f 1 -d "." | xargs -P 8 -I {} python store_other_methods.py ~/databases/depth/data.db anglor_phi continuous.PhiAngles ~/databases/depth/data/{}.fasta ~/databases/depth/data/{}.anglorphi
ls ~/databases/depth/data | grep fasta | cut -f 1 -d "." | xargs -P 8 -I {} python store_other_methods.py ~/databases/depth/data.db anglor_psi continuous.PsiAngles ~/databases/depth/data/{}.fasta ~/databases/depth/data/{}.anglorpsi
ls ~/databases/depth/data | grep fasta | cut -f 1 -d "." | xargs -P 8 -I {} python store_other_methods.py ~/databases/depth/data.db spinex_phi continuous.PhiAngles ~/databases/depth/data/{}.fasta ~/databases/depth/data/{}.spXout
ls ~/databases/depth/data | grep fasta | cut -f 1 -d "." | xargs -P 8 -I {} python store_other_methods.py ~/databases/depth/data.db spinex_psi continuous.PsiAngles ~/databases/depth/data/{}.fasta ~/databases/depth/data/{}.spXout

ls ~/databases/depth/data | grep fasta | cut -f 1 -d "." | xargs -P 8 -I {} python store_other_methods.py ~/databases/depth/data.db spinex_sec categorical.SecStruc ~/databases/depth/data/{}.fasta ~/databases/depth/data/{}.spXout

ls data | grep fasta | cut -f 1 -d "." | xargs -P 8 -I {} python store_other_methods.py data.db prof categorical.SecStruc data/{}.fasta data/{}.profRdb
ls data | grep fasta | cut -f 1 -d "." | xargs -P 8 -I {} python store_other_methods.py data.db psipred categorical.SecStruc data/{}.fasta data/{}.horiz

ls ~/databases/depth/data | grep fasta | cut -f 1 -d "." | xargs -P 8 -I {} python store_other_methods.py ~/databases/depth/data.db spinex_rsa continuous.RsaComplex ~/databases/depth/data/{}.fasta ~/databases/depth/data/{}.spXout
ls ~/databases/depth/data | grep fasta | cut -f 1 -d "." | xargs -P 8 -I {} python store_other_methods.py ~/databases/depth/data.db spinex_rsa continuous.RsaChain ~/databases/depth/data/{}.fasta ~/databases/depth/data/{}.spXout

ls ~/databases/depth/data | grep fasta | cut -f 1 -d "." | xargs -P 8 -I {} python store_other_methods.py ~/databases/depth/data.db predyflexy continuous.Bfactors ~/databases/depth/data/{}.fasta ~/databases/depth/data/{}.predyflexy
ls ~/databases/depth/data | grep fasta | cut -f 1 -d "." | xargs -P 8 -I {} python store_other_methods.py ~/databases/depth/data.db profbval_bnorm continuous.Bfactors ~/databases/depth/data/{}.fasta ~/databases/depth/data/{}.profbval

ls ~/databases/depth/data | grep fasta | cut -f 1 -d "." | xargs -P 8 -I {} python store_other_methods.py ~/databases/depth/data.db profbval_strict binary.Bfactors ~/databases/depth/data/{}.fasta ~/databases/depth/data/{}.profbval

ls ~/databases/depth/data | grep fasta | cut -f 1 -d "." | xargs -P 8 -I {} python store_other_methods.py ~/databases/depth/data.db memsat_svm categorical.Topology ~/databases/depth/data/{}.fasta ~/databases/depth/data/{}.memsat_svm
ls ~/databases/depth/data | grep fasta | cut -f 1 -d "." | xargs -P 8 -I {} python store_other_methods.py ~/databases/depth/data.db polyphobius categorical.Topology ~/databases/depth/data/{}.fasta ~/databases/depth/data/{}.polyphobius

# ML



