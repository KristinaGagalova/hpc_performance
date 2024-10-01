#!/bin/bash --login

#SBATCH --job-name=run_pggb
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00  # Adjust this as necessary
#SBATCH --account=y95  # Specify your account
#SBATCH --partition=work # Specify the partition
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load python/3.11.6

python3 get_su_threads_cpus.py 2,4,8,16,32 "module load singularity/4.1.0-slurm && \
	singularity run pggb:0.6.0--hdfd78af_0 \
	pggb -i /scratch/y95/kgagalova/benchmarking/genomes/arabidopsis/ara_pangenome.fasta \
	-o output_{cpus} \
	-t {cores} \
	-p 90 \
	-s 5k \
	-V 'ref:1000'" \
	y95 work
