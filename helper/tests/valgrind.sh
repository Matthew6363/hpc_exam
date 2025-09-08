#!/bin/bash
#SBATCH --job-name=valgrind_stencil
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4         # single process, multi-thread with OpenMP
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --hint=nomultithread
#SBATCH --exclusive

module load openMPI/5.0.5

BIN=stencil_parallel
SRC=../src/stencil_template_parallel.c
OUTDIR=outputs_valgrind
mkdir -p $OUTDIR

# Build with debug symbols and OpenMP support
mpicc -D_XOPEN_SOURCE=700 -O3 -march=native -std=c17 -g -fno-omit-frame-pointer -fopenmp -Iinclude "$SRC" -o "$BIN"

# Application arguments - use a small test case for profiling (adjust as needed)
APP_ARGS="-x 512 -y 512 -o 0 -v 0 -n 100"

# Set OpenMP environment variables to pin threads to physical cores for performance stability
export OMP_NUM_THREADS=4
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Cache configuration parameters typical for AMD EPYC Zen architecture
# Per-core caches (bytes)
I1="32768,8,64"
D1="32768,8,64"
L2="524288,8,64"
# Shared LLC slice (bytes)
L3="16777216,16,64"

echo "[*] Running Cachegrind L2 cache view (LL = L2 = $L2)"
srun -n1 -c1 --cpu-bind=cores \
  valgrind --tool=cachegrind --cache-sim=yes --branch-sim=yes \
           --I1=$I1 --D1=$D1 --LL=$L2 \
           --log-file=$OUTDIR/cachegrind_L2.log \
           --cachegrind-out-file=$OUTDIR/cachegrind_L2.out \
           ./$BIN $APP_ARGS

cg_annotate --auto=yes --show=Dr,D1mr,DLmr,Bc,Bcm  $OUTDIR/cachegrind_L2.out > $OUTDIR/cachegrind_L2_annotate.txt

echo "[*] Running Cachegrind L3 cache view (LL = L3 = $L3)"
srun -n1 -c1 --cpu-bind=cores \
  valgrind --tool=cachegrind --cache-sim=yes --branch-sim=yes \
           --I1=$I1 --D1=$D1 --LL=$L3 \
           --log-file=$OUTDIR/cachegrind_L3.log \
           --cachegrind-out-file=$OUTDIR/cachegrind_L3.out \
           ./$BIN $APP_ARGS

cg_annotate --auto=yes --show=Dr,D1mr,DLmr,Bc,Bcm  $OUTDIR/cachegrind_L3.out > $OUTDIR/cachegrind_L3_annotate.txt

echo "[*] Profiling complete. Annotated outputs saved to:"
echo "  $OUTDIR/cachegrind_L2_annotate.txt  (per-core L2 cache view)"
echo "  $OUTDIR/cachegrind_L3_annotate.txt  (shared LLC L3 cache view)"

