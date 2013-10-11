#!/bin/bash
for i in output*.dat
do
   echo "diff ../Sequential/output.dat $i"
   diff ../Sequential/output.dat $i
done
