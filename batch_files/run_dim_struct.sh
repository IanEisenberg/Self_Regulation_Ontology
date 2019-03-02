
flag=$1

for subset in task survey
do
    if [ ${flag}=="NONE" ]; then
        # first run the subset without classification
        sed  -e "s/{SUBSET}/$subset/g" -e "s/{CLASSIFIER}/NONE/g" dimensional_structure.batch | sbatch
    else
        # then run each classification
        for classifier in lasso ridge rf svm
        do
            sed  -e "s/{SUBSET}/$subset/g" -e "s/{CLASSIFIER}/$classifier/g" dimensional_structure.batch | sbatch
        done
    fi
done
