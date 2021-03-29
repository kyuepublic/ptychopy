/*
 * testEPIE.cpp
 *
 *  Created on: Aug 7, 2014
 *      Author: ynashed
 */

#include "MPIPhaser.h"
#include "Parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <iostream>

int main(int argc, char *argv[])
{

	Timer wallTimer;
	wallTimer.start();
	CXParams::getInstance()->parseFromCommandLine(argc, argv);

#ifdef HAVE_MPI
	IPhaser* phaser = new MPIPhaser;
#else
	IPhaser* phaser = new IPhaser;
#endif
	if(phaser->init())
	{
		phaser->addPhasingMethod( 	CXParams::getInstance()->getReconstructionParams()->algorithm.c_str(),
									CXParams::getInstance()->getReconstructionParams()->iterations);
		phaser->phase();
		wallTimer.start();
		phaser->writeResultsToDisk();
		fprintf(stderr,"Walltime:%f\n", wallTimer.getElapsedTimeInSec());
	}
	delete phaser;

	return 0;
}
