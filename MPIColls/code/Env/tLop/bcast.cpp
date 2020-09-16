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
#include <fstream>
#include <map>
using namespace std;


// Algorithms and collectives supported
typedef enum {BINOMIAL, LINEAR, ADAPTIVE} algorithms_t;
map<string, const int> algorithms = {
	{"Binomial",  BINOMIAL},
	{"Linear",    LINEAR},
	{"Adaptive",  ADAPTIVE}
};

typedef enum {NUM_PROCS,
	            ROOT,
							MSG_SIZE,
							SEGMENT_SIZE,
							NUM_NODES,
							NODES,
							MAPPING,
							NETWORK,
							PLATFORM,
							COLLECTIVE,
							ALGORITHM,
							N_ITER,
							GRAPH
						} filecontent_t;

map<string, const int> filecontent = {
	{"# P",          NUM_PROCS},
	{"# Root",       ROOT},
	{"# m",          MSG_SIZE},
	{"# S",          SEGMENT_SIZE},
	{"# M",          NUM_NODES},
	{"# Nodes",      NODES},
	{"# Mapping",    MAPPING},
	{"# Network",    NETWORK},
	{"# Platform",   PLATFORM},
	{"# Collective", COLLECTIVE},
	{"# Algorithm",  ALGORITHM},
	{"# n_iter",     N_ITER},
	{"# Graph",      GRAPH}
};


// Parameters structure (to be read from file)
struct params {
	int      P;
	int      root;
	int      S;
	int      m;
	int      M;
	int     *nodes;
	string   net;
	string   collective;
	string   algorithm;
	string   platform;
	int     *mapping;
	int      n_iter;
};
typedef struct params params;



// Helper function to get nodes/ranks as a vector
void str_to_vector (int *vec, string str) {

	std::stringstream ss(str);
	int idx = 0;

	for (int i; ;) {
		if ( (ss.peek() == ',') || (ss.peek() == ' ') ||
		     (ss.peek() == '[') || (ss.peek() == ']')   ) {
			ss.ignore();
		} else {
			if (!(ss >> i))
			  break;
			vec[idx] = i;
			idx++;
		}
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


// Adaptive broadcast
double adaptive (int m, Communicator *w) {

	cerr << "BCAST - Adaptive" << endl;

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
	double t = 0.0;
	Graph  g;

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
	if (bcast_file.empty()) {
		cerr << "ERROR File not found: " << bcast_file << endl;
		cout << t << endl;
		return (-1);
	}

	ifstream bfile;
  string str;
	params pm;

  bfile.open(bcast_file);
	if (bfile.fail()) {
		cerr << "ERROR openning file: " << bcast_file << endl;
		cout << t << endl;
		return (-1);
	}

	while (! bfile.eof()) {
		getline(bfile, str);
		if (bfile.eof())
		  break;

		switch(filecontent[str]) {
			case NUM_PROCS:
			  getline(bfile, str);
				pm.P = stoi(str);
			  break;
			case ROOT:
			  getline(bfile, str);
				pm.root = stoi(str);
			  break;
			case MSG_SIZE:
			  getline(bfile, str);
				pm.m = stoi(str);
				break;
			case SEGMENT_SIZE:
			  getline(bfile, str);
				pm.S = stoi(str);
			  break;
			case NUM_NODES:
			  getline(bfile, str);
				pm.M = stoi(str);
			  break;
			case NODES:
			  getline(bfile, str);
				pm.nodes = new int [pm.M];
				for (int i = 0; i < pm.M; i++) pm.nodes[i] = i;
			  break;
			case MAPPING:
			  getline(bfile, str);
				pm.mapping = new int [pm.P];
				str_to_vector(pm.mapping, str);
			  break;
			case NETWORK:
			  getline(bfile, str);
				pm.net = str;
			  break;
			case PLATFORM:
			  getline(bfile, str);
				pm.platform = str;
			  break;
  		case COLLECTIVE:
	  	  getline(bfile, str);
				pm.collective = str;
			  break;
			case ALGORITHM:
			  getline(bfile, str);
				pm.algorithm = str;
			  break;
			case N_ITER:
			  getline(bfile, str);
				pm.n_iter = stoi(str);
			  break;
			case GRAPH:
			  getline(bfile, str);
				cerr << "TBD: Graph (example): " << str << endl;
				g.insert(0,0,1);
				g.insert(0,1,2);
				g.insert(0,2,3);
				g.insert(1,3,3);
				g.insert(0,4,4);
				g.insert(1,5,4);
				g.insert(2,6,4);
				g.insert(3,7,4);
			  break;
			default:
			  cerr << "ERROR: unknown option in file " << str << endl;
			  break;
		}
	}

  bfile.close();


	// Show parameters. Use only stderr because stdout is for sending the result.
	cerr << "BCAST Options FILE    " << bcast_file    << endl;
	cerr << "Number of processes:  " << pm.P          << endl;
	cerr << "Root:                 " << pm.root       << endl;
	cerr << "Number of nodes:      " << pm.M          << endl;
	cerr << "Segment size:         " << pm.S          << endl;
	cerr << "Message size:         " << pm.m          << endl;
	cerr << "Network type:         " << pm.net        << endl;
	cerr << "Collective operation: " << pm.collective << endl;
	cerr << "Algorithm:            " << pm.algorithm  << endl;
	cerr << "HPC platform:         " << pm.platform   << endl;
	cerr << "Num. iterations:      " << pm.n_iter     << endl;
	cerr << "Nodes:                "                  << endl;
	for (int i = 0; i < pm.M; i++) {
		cerr << " " << pm.nodes[i];
	}
	cerr << endl;
	cerr << "Mapping:              "                  << endl;
	for (int i = 0; i < pm.P; i++) {
		cerr << " " << pm.mapping[i];
	}
	cerr << endl;


	// Network parameters
	TauLopParam::setInstance(pm.net.c_str());

	// Communicator
	Communicator *world = new Communicator (pm.P);

  // Mapping
	Mapping *map = new Mapping (pm.P, pm.mapping);
	world->map(map);

	switch (algorithms[pm.algorithm]) {
		case BINOMIAL:
		  cerr << "Here you have, the Binomial broadcast." << endl;
			t = binomial(pm.m, world);
			break;
		case ADAPTIVE:
		  t = adaptive(pm.m, world);
			cerr << "I know you want to run an adaptive algorithm, but you must to WAIT." << endl;
			break;
		default:
			cerr << "ERROR: collective " << pm.algorithm << " not supported." << endl;
	}

	delete [] pm.nodes;
	delete [] pm.mapping;
	delete map;
	delete world;

	// Return value
	cout << t << endl;

	return 0;
}
