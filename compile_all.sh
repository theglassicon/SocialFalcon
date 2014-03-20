#/bin/bash

ALGORITHMS="pmf regsvd svd++ socialmf socialfalcon prepare_db"


echo "##############################################################"

for i in $ALGORITHMS

do

icc -xT -O3 -ip -parallel $i.c -o $i -I/usr/include/mysql -L/usr/lib/mysql -lmysqlclient
#gcc pmf.c -o pmf -I/usr/include/mysql -L/usr/lib/mysql -lmysqlclient

echo -e "Done $i\n"

done

echo "##############################################################"


