#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cassert>
#include <omp.h>
#include <time.h>


constexpr int CHECK_CYCLES = 3;
constexpr int INSERT_EASY_CYCLES = 4;
constexpr int INSERT_HARD_CYCLES = 5;

typedef std::vector<float> corpus_vector;
typedef std::vector<std::pair<double, int>> topk_list; // one query, one NMA


struct offload_context{
	int total_vectors;
	int* vectors_per_iks;
	float** corpus_vectors;
	float** query_vectors;
	int query_batch_size;
	int** vectors_per_iks_nma; // [num_iks, num_nma]
	topk_list*** top_k_lists;  // [num_iks,num_nma, num_pe]

	offload_context(int total_vectors, int num_nma, int num_pe, int num_iks, int query_batch_size = 1, bool detailed=false) : total_vectors(total_vectors), query_batch_size(query_batch_size) {
		corpus_vectors = new float*[total_vectors];
		std::mt19937 rng(std::random_device{}());
		std::uniform_real_distribution<double> dist(0.0, 1.0);
		if (detailed) {
#pragma omp parallel for
		for (int i = 0; i < total_vectors; i++) {
			corpus_vectors[i] = new float[768];
			for (int j = 0; j < 768; j++) {
				corpus_vectors[i][j] = dist(rng);
			}
		}
		query_vectors = new float*[query_batch_size];
		for (int i = 0; i < query_batch_size; i++) {
			query_vectors[i] = new float[768];
			for (int j = 0; j < 768; j++) {
				query_vectors[i][j] = dist(rng);
			}
		}
		}
		else {
			corpus_vectors = nullptr;
			query_vectors = nullptr;
		}
		vectors_per_iks = new int[num_iks];
		vectors_per_iks_nma = new int*[num_nma];
		for (int i = 0; i < num_nma; i++) {
			vectors_per_iks_nma[i] = new int[num_nma];
		}
		top_k_lists = new topk_list**[num_iks];
		for (int i = 0; i < num_iks; i++) {
			top_k_lists[i] = new topk_list*[num_nma];
			for (int j = 0; j < num_nma; j++) {
				top_k_lists[i][j] = new topk_list[num_pe];
			}
		}
	}


};

struct TopK {
	int busy_cycles;
	std::vector<std::pair<double, int>> queue;

	TopK() : busy_cycles(0) {}

	// Gather timing and insert the score into the queue
	int check(double score, int id) {
		int out = busy_cycles;
		if (queue.size() < 32) {
			busy_cycles = INSERT_EASY_CYCLES;
			queue.emplace_back(score, id);
			std::sort(queue.begin(), queue.end());
		} else if (score > queue[0].first) {
			busy_cycles = INSERT_HARD_CYCLES;
			queue[0] = {score, id};
			std::sort(queue.begin(), queue.end());
		} else {
			busy_cycles = CHECK_CYCLES;
		}
		return out;
	}
};

struct PE {
	TopK topk;
	int updating_cycles;
	int d;
	int id_pe;
	int id_nma;
	int id_iks;
	int mac_units;


	PE(int d, int mac_units, int id_pe, int id_nma, int id_iks) : d(d), mac_units(mac_units), id_pe(id_pe), id_nma(id_nma), id_iks(id_iks) {
		updating_cycles = 0;
	}

	std::pair<int, int> run_one_batch(int offset, int corpus_batch_size, offload_context* offload_context){ 
		if (corpus_batch_size == 0) corpus_batch_size = mac_units;
		assert(corpus_batch_size <= mac_units);
		int base_cycles = d;
		int top_k_cycles = 0;
		int prev_stall = updating_cycles;
		std::mt19937 rng(std::random_device{}());
		std::uniform_real_distribution<double> dist(0.0, 1.0);

		for (int m = 0; m < corpus_batch_size; ++m) {
			int id = offset + m;



		   float dot_product = 0;
		   if (offload_context->corpus_vectors != nullptr && offload_context->query_vectors != nullptr) {
			   dot_product = 0;
		   for (int i = 0; i < d; i++) {
			   dot_product += offload_context->corpus_vectors[id][i] * offload_context->query_vectors[id_pe][i];
		   }
			top_k_cycles += topk.check(dot_product, id);
		   }
		   else {
			   top_k_cycles += topk.check(dist(rng), id);
		   }

		}

		updating_cycles = top_k_cycles;
		offload_context->top_k_lists[id_iks][id_nma][id_pe] = topk.queue;
		if (prev_stall > base_cycles) {
			return {prev_stall - base_cycles, base_cycles};
		} else {
			return {0, base_cycles};
		}
	}
};

struct NMA {
	std::vector<PE> pe_array;
	int d;
	int mac_units;
	int num_vectors;
	int id_nma;
	int id_iks;

	NMA(int d, int mac_units, int num_pe, int id_nma, int id_iks) : d(d), mac_units(mac_units), id_nma(id_nma), id_iks(id_iks) {
		pe_array.reserve(num_pe);
		for (int i = 0; i < num_pe; ++i) {
			pe_array.emplace_back(d, mac_units, i, id_nma, id_iks);
		}
	}

	std::pair<int, int> run_one_batch(int offset, int corpus_batch_size, offload_context* offload_context) {
		int cycles = 0;
		int stall_cycles = 0;

		for (size_t i = 0; i < offload_context->query_batch_size && i < pe_array.size(); ++i) {
			auto [stall, useful] = pe_array[i].run_one_batch(offset, corpus_batch_size, offload_context);
			cycles = useful;
			if (stall > stall_cycles) {
				stall_cycles = stall;
			}
		}
		return {stall_cycles, cycles};
	}

	std::pair<int, int> run_batches(int iks_offset, offload_context* offload_context) {
		int total_vectors = offload_context->vectors_per_iks_nma[id_iks][id_nma];
		int num_full_batches = total_vectors / mac_units;
		int remainder = total_vectors % mac_units;
		int cycles = 0;
		int stall_cycles = 0;
		float** query_major_result = new float*[offload_context->query_batch_size];
		for (int i = 0; i < offload_context->query_batch_size; i++) {
			query_major_result[i] = new float[total_vectors];
		}

		int offset = iks_offset;

		for (int i = 0; i < id_nma; i++) {
			offset += offload_context->vectors_per_iks_nma[id_iks][i];
		}



		for (int i = 0; i < num_full_batches; ++i) {
			auto [stall, useful] = run_one_batch(offset, mac_units, offload_context);
			cycles += useful;
			stall_cycles += stall;
			offset += mac_units;
		}

		if (remainder > 0) {
			auto [stall, useful] = run_one_batch(offset, remainder, offload_context);
			cycles += useful;
			stall_cycles += stall;
		}

		return {stall_cycles, cycles};
	}

	void store_vectors(int num_vectors) {
		this->num_vectors = num_vectors;
	}

};



	


struct IKS {
	std::vector<NMA> nma_array;
	int d;
	int mac_units;
	int id_iks;

	IKS(int d, int mac_units, int num_pe, int num_nma, int id_iks) : d(d), mac_units(mac_units), id_iks(id_iks) {
		nma_array.reserve(num_nma);
		for (int i = 0; i < num_nma; ++i) {
			nma_array.emplace_back(d, mac_units, num_pe, i, id_iks);
		}
	}

	// Chooses the number of vectors for each NMA
	void store_vectors(int num_vectors, offload_context* offload_context) {
		int num_nma = nma_array.size();
		std::vector<int> vectors_per_nma(num_nma, num_vectors / num_nma);
		int remainder = num_vectors % num_nma;
		for (int i = 0; i < remainder; ++i) {
			vectors_per_nma[i] += 1;

		}
		for (size_t i = 0; i < nma_array.size(); ++i) {
			nma_array[i].store_vectors(vectors_per_nma[i]);
			offload_context->vectors_per_iks_nma[id_iks][i] = vectors_per_nma[i];
		}
	}


	std::pair<int, int> run_search(offload_context* offload_context) {
		store_vectors(offload_context->vectors_per_iks[id_iks], offload_context);



		int stall = 0;
		int useful = 0;

		std::pair<int, int>* stall_useful_cycles = new std::pair<int, int>[nma_array.size()];

		omp_set_num_threads(nma_array.size()+1 );

		#pragma omp parallel
		{
			int thread_id = omp_get_thread_num();
			if (thread_id == 0) {
				#pragma omp master
				{
					std::cerr << "Printing to waste time in master thread (See README.md)" << std::endl;
				}
			}

				else{
				#pragma omp single
				{
					for (int i = 0; i < nma_array.size(); i++) {
						#pragma omp task
						{
							int iks_offset = 0;
							for (int j = 0; j < id_iks; j++) {
								iks_offset += offload_context->vectors_per_iks[j];
							}
							auto [s, u] = nma_array[i].run_batches(iks_offset, offload_context);
							stall_useful_cycles[i] = {s, u};
						}
					}
				};

			}
		};


		for (int i = 0; i < nma_array.size(); i++) {
			if (stall_useful_cycles[i].first > stall) {
				stall = stall_useful_cycles[i].first;
			}
			if (stall_useful_cycles[i].second > useful) {
				useful = stall_useful_cycles[i].second;
			}
		}
		return {stall, useful};
	}
};

struct MultiIKS {
	std::vector<IKS> iks_array;
	int d;
	int mac_units;
	
	MultiIKS(int d, int mac_units, int num_pe, int num_nma, int num_iks) : d(d), mac_units(mac_units) {
		iks_array.reserve(num_iks);
		for (int i = 0; i < num_iks; ++i) {
			iks_array.emplace_back(d, mac_units, num_pe, num_nma, i);
		}
	}

	// Chooses the number of vectors to store in each IKS
	void store_vectors(int num_vectors, offload_context* offload_context) {

		int num_iks = iks_array.size();
		std::vector<int> vectors_per_iks(num_iks, num_vectors / num_iks);

		int remainder = num_vectors % num_iks;
		for (int i = 0; i < remainder; ++i) {
			vectors_per_iks[i] += 1;
		}

		for (size_t i = 0; i < iks_array.size(); ++i) {
			offload_context->vectors_per_iks[i] = vectors_per_iks[i];
		}

	}

	std::pair<int, int> run_search(offload_context* offload_context) {
		store_vectors(offload_context->total_vectors, offload_context);
		int stall = 0;
		int useful = 0;
		for (size_t i = 0; i < iks_array.size(); ++i) {
			auto [s, u] = iks_array[i].run_search(offload_context);

			if (s > stall) {
				stall = s;
			}
			if (u > useful) {
				useful = u;
			}
		}
		return {stall, useful};
	}

};




int main(int argc, char** argv) {
	int d, mac_units, num_pe, num_nma, num_iks, corpus_size, batch_size;
	if (argc == 8) {
		d = std::stoi(argv[1]);
		mac_units = std::stoi(argv[2]);
		num_pe = std::stoi(argv[3]);
		num_nma = std::stoi(argv[4]);
		num_iks = std::stoi(argv[5]);
		corpus_size = std::stoi(argv[6]);
		batch_size = std::stoi(argv[7]);
	} else {
		std::cerr << "Usage: " << argv[0] << " d mac_units num_pe num_nma num_iks corpus_size batch_size" << std::endl;
		return 1;
	}
	offload_context offload_context(corpus_size, num_nma, num_pe, num_iks, batch_size);



	MultiIKS multiiks(d, mac_units, num_pe, num_nma, num_iks);


	std::pair<int, int> multi_iks_cycles = multiiks.run_search(&offload_context);

	timespec start_time, end_time;

	clock_gettime(CLOCK_MONOTONIC, &start_time);
	topk_list* final_top_k_list = new topk_list[batch_size];
	for (int i = 0; i < batch_size; i++) {
		std::vector<std::pair<double, int>> final_top_k_list_i = {};
		for (int j = 0; j < num_iks; j++) {

			for (int k = 0; k < num_nma; k++) {
				final_top_k_list_i.insert(final_top_k_list_i.end(), offload_context.top_k_lists[j][k][i].begin(), offload_context.top_k_lists[j][k][i].end());

			}
		}
		std::sort(final_top_k_list_i.begin(), final_top_k_list_i.end(), std::greater<std::pair<double, int>>());
		if (final_top_k_list_i.size() > 32) {
			final_top_k_list_i.resize(32);
		}
		final_top_k_list[i] = final_top_k_list_i;
	}
	clock_gettime(CLOCK_MONOTONIC, &end_time);

	int top_k_ns = end_time.tv_nsec-start_time.tv_nsec;

	std::cout << "Stall cycles: " << multi_iks_cycles.first << std::endl;
	std::cout << "Useful cycles: " << multi_iks_cycles.second << std::endl;
	std::cout << "Top-k time: " << top_k_ns/1e3 << " us" << std::endl;
	std::cout << "Total time: " << (multi_iks_cycles.first + multi_iks_cycles.second + top_k_ns)/1e6 << " ms" << std::endl;









	return 0;
}

