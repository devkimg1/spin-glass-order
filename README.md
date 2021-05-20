# spin-glass-order
Quantifying order in spin glasses via computable information density
numba_ized_split_1024_1 is an example of how the rejection free works. To run this model on slurm, move the entire folder into your msi directory. Then, move all of the .txt files into your directory.
sbatch instant.txt

wait a few minutes for the models to instantiate

sbatch n_choice_ising.txt

wait around 8 hours for the simulation to complete

sbatch reader.txt

the above creates results
