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
typedef enum {BINOMIAL, LINEAR} algorithms_t;

map<string, const int> algorithms = {
	{"binomial", BINOMIAL},
	{"linear",   LINEAR}
};


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
// DESC: read parameters from python invocation and call algorithm.
// Return: (cout the time, then in Python capture the output)
int main (int argc, char * argv[]) {
	
	int opt;
	
	int P;
	int m;
	string M_str;
	string c;
	string a;
	string n;
	string s;

	
	/*
	cerr << "Parameters: " << argc << endl;
	for (int i = 0; i < argc; i++) {
		cerr << argv[i] << endl;
	}
	*/
	
	
	while((opt = getopt(argc, argv, "P:m:M:c:a:n:s:")) != -1)
	{
		switch(opt)
		{
			case 'P': // Number of processes
				P = stoi(optarg);
				break;
			case 'm': // Message size
				m = stoi(optarg);
				break;
			case 'M': // Node vector
				M_str = string(optarg);
				break;
			case 'c': // Collective operation
				c = string(optarg);
				break;
			case 'a': // Algorithm
				a = string(optarg);
				break;
			case 'n': // Network type
				n = string(optarg);
				break;
			case 's': // HPC platform
				s = string(optarg);
				break;
			case ':':
				cerr << "option needs a value" << endl;
				break;
			case '?':
				cerr << "unknown option: " << optopt << endl;
				break;
		}
	}
	
	
	// PRINT VALUES:
	cerr << "Number of processes:  " << P << endl;
	cerr << "Message size:         " << m << endl;
	cerr << "Node vector:          " << M_str << endl;
	cerr << "Collective operation: " << c << endl;
	cerr << "Algorithm:            " << a << endl;
	cerr << "Network type:         " << n << endl;
	cerr << "HPC platform:         " << s << endl;
	
	
	
	double t      = 0.0;
	
	// Network parameters
	TauLopParam::setInstance(n.c_str());
	
	// Communicator
	Communicator *world = new Communicator (P);
	
	// Mapping (default: Sequential)
	int *nodes = new int [P];
	str_to_vector(P, nodes, M_str);

	
	Mapping *map = new Mapping (P, nodes);
	world->map(map);
	
	
	switch (algorithms[a]) {
		case BINOMIAL:
			t = binomial(m, world);
			break;
		default:
			cerr << "ERROR: collective " << a << " not supported." << endl;
	}

	
	delete [] nodes;
	delete map;
	delete world;

	// Return value
	cout << t << endl;
	// cout << flush;
	
	return 0;
}



