#!/bash

#
# $time sh LSW_RUN_burn.sh
#

#cmd="time ./gpu_burn 60"
cmd="./gpu_burn 60"
#for i in {1..10} ;  do ${cmd} ; done
for i in {1..10}
do
       echo "now on " ${i}	
	${cmd}
done
