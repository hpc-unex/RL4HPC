//
//  main.cpp
//  test_colls
//
//  Created by jarico on 20/4/17.
//  Copyright © 2017 jarico. All rights reserved.
//

#include "taulop_kernel.hpp"
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <iomanip>
#include <unistd.h>
#include <string>
#include <sstream>
#include <map>
using namespace std;


// Algorithms and collectives supported
typedef enum {BINOMIAL, LINEAR, ADAPTIVE} algorithms_t;

map<string, const int> algorithms = {
	{"Binomial",  BINOMIAL},
	{"Linear",    LINEAR},
	{"Adaptive",  ADAPTIVE}
};

// Parameters structure (to be read from file)
struct params {
	int      P;
	int      S;
	int      m;
	int      M;
	int     *nodes;
	string   net;
	string   collective;
	string   algorithm;
	string   platform;
};
typedef struct params params;


// Helper function to get nodes as a vector
void str_to_vector (int P, int *nodes, string str) {

	std::stringstream ss(str);
	int idx = 0;

	for (int i; ss >> i;) {
		nodes[idx] = i;
		idx++;
		if ((ss.peek() == ',') || (ss.peek() == ' '))
			ss.ignore();
	}
}



///////////////////////////////////
// Algorithms implementation.
///////////////////////////////////


// Binomial broadcast
double binomial (int m, Communicator *w) {

	cerr << "BCAST - Binomial" << endl;

	Collective *bcast  = new BcastBinomial();
	double t = 0.0;

	int size = m;
	TauLopCost *tcoll = bcast->evaluate(w, &size);
	t = tcoll->getTime();

	delete tcoll;

	return t;
}




// Main:
// DESC: read parameters from python generated file and call algorithm.
// Return: (cout the time, then Python invoker captures the output)
int main (int argc, char * argv[]) {

	int    opt;
	string bcast_file;

	/*
	cerr << "Parameters: " << argc << endl;
	for (int i = 0; i < argc; i++) {
		cerr << argv[i] << endl;
	}
	*/


	while((opt = getopt(argc, argv, "f:")) != -1)
	{
		switch(opt)
		{
			case 'f':
			  bcast_file = string(optarg);
				break;
			case ':':
				cerr << "option needs a value" << endl;
				break;
			case '?':
				cerr << "unknown option: " << optopt << endl;
				break;
		}
	}

  // TODO: read parameters from file
	params pm;
	pm.P = 8;
	pm.m = 1024;
	pm.net = "IB";
	pm.collective = "MPI_Bcast";
	pm.algorithm = "Binomial";
	pm.platform = "CIEMAT";
	pm.nodes = new int [pm.P];
	for (int i = 0; i < pm.P; i++) {
		pm.nodes[i] = 0;
	}

	// Show parameters. Use only stderr because stdout is for sending the result.
	cerr << "BCAST Options FILE    " << bcast_file    << endl;
	cerr << "Number of processes:  " << pm.P          << endl;
	cerr << "Message size:         " << pm.m          << endl;
	cerr << "Network type:         " << pm.net        << endl;
	cerr << "Collective operation: " << pm.collective << endl;
	cerr << "Algorithm:            " << pm.algorithm  << endl;
	cerr << "HPC platform:         " << pm.platform   << endl;


	double t      = 0.0;

	// Network parameters
	TauLopParam::setInstance(pm.net.c_str());

	// Communicator
	Communicator *world = new Communicator (pm.P);

	// Mapping (default: Sequential)
	int *nodes = new int [pm.P];
	// str_to_vector(pm.P, pm.nodes, M_str);

	Mapping *map = new Mapping (pm.P, pm.nodes);
	world->map(map);

	switch (algorithms[pm.algorithm]) {
		case BINOMIAL:
		  cerr << "Here you have, the Binomial broadcast." << endl;
			t = binomial(pm.m, world);
			break;
		case ADAPTIVE:
		  t = binomial(pm.m, world);
			cerr << "I know you want to run an adaptive algorithm, but you must to WAIT." << endl;
			break;
		default:
			cerr << "ERROR: collective " << pm.algorithm << " not supported." << endl;
	}

	delete [] pm.nodes;
	delete map;
	delete world;

	// Return value
	cout << t << endl;
	// cout << flush;

	return 0;
}
