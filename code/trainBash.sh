#!/bin/bash -l

#PBS -N LSTMTrainingScript

#PBS -q default

#PBS -l nodes=2:ppn=1

#PBS -l walltime=00:15:00

#PBS -m bea

#PBS -M aaron.m.karp.gr@dartmouth.edu

cd $PBS_O_WORKDIR

./trainLSTMScript.py