#/bin/bash

server="localhost";  #where the mysql database is

user="root";    #the root user of mysql 

password="checkmate#$%"; #the password of the root user in mysql

database="Flixter"; #the database to pick

#ALGORITHMS="aleco_mem alecosocialmf_full_mem batchbp_mem pmf_mem regsvd_mem svd++_mem"
ALGORITHMS="regsvd_mem_print_bad pmf_mem_print_bad"

rm log.txt #remove any previous log

for features in 5 10

do

echo "##############################################################" >> log.txt
echo "Number Of Features $features" >> log.txt
echo "##############################################################" >> log.txt

for p in 01 02 03 04 05 

do

echo "Fold $p" >> log.txt
echo "##############################################################" >> log.txt
echo -e "Algorithm\t\tRMSE\t\tIterations\t\tAverage Iter\t\tTime\n" >> log.txt

for i in $ALGORITHMS
do

./$i $server $user $password $database$p $features 3400

done

done

done 

echo

echo "##############################################################" >> log.txt

echo

echo -e "Algorithm\tAverage RMSE\tAverage Iterations\tAverage Iter\tAverage Time\n" >> log.txt

for i in $ALGORITHMS
do

avg_rmse=`grep $i log.txt | awk '{s+=$2}END{print s/NR}'`
avg_iter=`grep $i log.txt | awk '{s+=$3}END{print s/NR}'`
avg_iter_time=`grep $i log.txt | awk '{s+=$4}END{print s/NR}'`
avg_time=`grep $i log.txt | awk '{s+=$6}END{print s/NR}'`

echo -e "$i\t\t$avg_rmse\t\t$avg_iter\t\t$avg_iter_time\t\t\t$avg_time\n" >> log.txt

done

echo "##############################################################" >> log.txt


