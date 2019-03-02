set -e
for t in t1 t2
do
sed -e "s/{T}/$t/g" intratrial_reliability.batch | sbatch -p russpold
done
