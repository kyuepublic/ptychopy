/*
 * optimizeEPIE.cpp
 *
 *  Created on: Aug 12, 2014
 *      Author: ynashed
 */

#ifdef HAVE_MPI
#include "IPhaser.h"
#include "Parameters.h"
#include "utilities.h"

#include <PSO_Optimizer.h>
#include <DE_Optimizer.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>

#include <mpi.h>

using namespace std;
using namespace CudaOptimize;

static IPhaser* phaser = 0;
static unsigned int generation = 0;

float evalSolution(const float* params)
{
	CXParams::getInstance()->setDrift(make_float2(params[0], params[1]));
	phaser->phase();
//	printf("Tried (%e,%e) with error (%f)\n", eParams.drift.x, eParams.drift.y, error);
	return phaser->getPhaseErrors().back();
}

void optimizeEPIE(const SolutionSet* parameters, FitnessSet* fitnesses, dim3 fitnessGrid, dim3 fitnessBlock)
{
	float* errors_h = new float[parameters->getSolutionNumber()];

	for(unsigned int s=1; s<parameters->getSolutionNumber(); ++s)
		MPI_Send((void*)parameters->getHostPositionsConst(0, s), 2, MPI_FLOAT, s, 999, MPI_COMM_WORLD);

	float myError = evalSolution(parameters->getHostPositionsConst(0, 0));
	MPI_Gather( &myError, 1, MPI_FLOAT, errors_h, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	fitnesses->set(errors_h);
	delete [] errors_h;

	printf("### END GENERATION [%d] ###\n", ++generation);
}

typedef unsigned long long timestamp_t;

static timestamp_t get_timestamp ()
{
	struct timeval now;
	gettimeofday (&now, NULL);
	return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

int main(int argc, char *argv[])
{
	CXParams::getInstance()->parseFromCommandLine(argc, argv);
	const ExperimentParams* eParams = CXParams::getInstance()->getExperimentParams();
	const ReconstructionParams* rParams = CXParams::getInstance()->getReconstructionParams();


	int procs, myrank, GPUNum;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	GPUNum = GPUQuery::getInstance()->getDeviceCount();
	cudaSetDevice(myrank%GPUNum);

	phaser = new IPhaser;
	phaser->addPhasingMethod("ePIE", 30);

	if(myrank == 0)
	{
		PSO_Optimizer somOptimizer(optimizeEPIE, 2, 1, procs);
		somOptimizer.setTerminationFlags(TERMINATE_GENS);
		somOptimizer.setGenerations(rParams->iterations);
		somOptimizer.setHostFitnessEvaluation(true);
		somOptimizer.setTopology(GBEST);

		somOptimizer.setBounds(0, 0, make_float2(eParams->stepSize.x, 0.0f));
		somOptimizer.setBounds(0, 1, make_float2(eParams->stepSize.y, 0.0f));

		timestamp_t t0 = get_timestamp();
		somOptimizer.optimize();
		timestamp_t t1 = get_timestamp();
		cout << "Optimization done... [" << (t1 - t0) / 1000000.0L << "]" << endl;
		const float* finalDriftValues = somOptimizer.getBestSolution(0);
		printf("Found drifts (%e,%e)\n", finalDriftValues[0], finalDriftValues[1]);

	}
	else
	{
		MPI_Status status;
		float drifts[2];

		for(unsigned int i=0; i<rParams->iterations; ++i)
		{
			MPI_Recv(&drifts, 2, MPI_FLOAT, 0, 999, MPI_COMM_WORLD, &status);
			float myError = evalSolution(drifts);
			MPI_Gather( &myError, 1, MPI_FLOAT, 0, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
		}

	}

	delete phaser;
	MPI_Finalize();

	return 0;
}
#else
int main(int argc, char *argv[])
{return 0;}
#endif
