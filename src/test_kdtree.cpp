#include <iostream>
#include <time.h>
#include <fstream>
#include "KDTree.h"

#define VISUALIZATION_KDTREE

const int RAND_NUM = 10000;
const int SEARCH_NUM = 10;
const int TREE_DIM = 2;

using dataset = std::vector<std::pair<Point<TREE_DIM>, int>>;

void Search(int num, Point<TREE_DIM>& pos_source, std::vector<Point<TREE_DIM>>& pos_list, dataset& all_pos);

void KDTreeTest()
{
	int count = 0;
	dataset data;
	for (int it = 0; it < RAND_NUM; ++it) {
		std::pair<Point<TREE_DIM>, int> pt;
		pt.second = count++;
		pt.first[0] = std::rand() % int(KD_TREE_MAX_X - KD_TREE_MIN_X) + KD_TREE_MIN_X + 1;
		pt.first[1] = std::rand() % int(KD_TREE_MAX_Y - KD_TREE_MIN_Y) + KD_TREE_MIN_Y + 1;
		data.push_back(pt);
	}

	clock_t start_build_tree = clock();
	KDTree<TREE_DIM, int> kd(data);
	clock_t end_build_tree = clock();
	std::cout << "Finished building KD-Tree! Time:"
		<< double(end_build_tree - start_build_tree) / CLOCKS_PER_SEC
		<< "ms" << std::endl;

	clock_t start_search_tree = clock();
	Point<TREE_DIM> target;
	target[0] = 1600;
	target[1] = 1600;
	kd.kNNValue(target, 100);
	clock_t end_search_tree = clock();
	std::cout << "KD-Tree Search Time:"
		<< double(end_search_tree - start_search_tree) / CLOCKS_PER_SEC 
		<< "ms" << std::endl;

	clock_t start_search_normal = clock();
	std::vector<Point<TREE_DIM>> pos_list;
	Search(100, target, pos_list, data);
	clock_t end_search_normal = clock();
	std::cout << "Normal Search Time:"
		<< double(end_search_normal - start_search_normal) / CLOCKS_PER_SEC
		<< "ms" << std::endl;
	for (int it = 0; it < pos_list.size(); ++it) {
		for (int dim = 0; dim < TREE_DIM; ++dim) {
			std::cout << pos_list[it][dim] << ",";
		}
	}
	std::cout << std::endl;

#ifdef SANITY_CHECK
	// Sanity Check With KDTree
	clock_t start_sanity_check = clock();
	bool sanityPass = true;
	for (int it = 0; it < RAND_NUM; ++it) {
		if (!kd.contains(data[it].first) || kd.kNNValue(data[it].first, 1) != data[it].second) {
			sanityPass = false;
			break;
		}
	}
	clock_t end_sanity_check = clock();
	if (sanityPass) std::cout << "Sanity Check With KDTree PASSED! Time:";
	else std::cout << "Sanity Check With KDTree FAILED!" << std::endl;
	std::cout << double(end_sanity_check - start_sanity_check) / CLOCKS_PER_SEC << std::endl;

	// Sanity Check Normal
	clock_t start_sanity_check_normal = clock();
	sanityPass = true;
	for (int it = 0; it < RAND_NUM; ++it) {
		std::vector<Point<TREE_DIM>> pos_list;
		Search(1, data[it].first, pos_list, data);
		if (pos_list[0][0] != data[it].first[0] || pos_list[0][1] != data[it].first[1]) {
			sanityPass = false;
			break;
		}
	}
	clock_t end_sanity_check_normal = clock();
	if (sanityPass) std::cout << "Sanity Check With Normal PASSED! Time:";
	else std::cout << "Sanity Check With Normal FAILED!" << std::endl;
	std::cout << double(end_sanity_check_normal - start_sanity_check_normal) / CLOCKS_PER_SEC << std::endl;
#endif	
}

void Search(int num, Point<TREE_DIM>& pos_source, std::vector<Point<TREE_DIM>>& pos_list, dataset& all_pos)
{
	struct Compare {
		bool operator() (const std::pair<float, Point<TREE_DIM>>& p1,
			const std::pair<float, Point<TREE_DIM>>& p2) {
			return p1.first < p2.first;
		}
	} comp_pair;

	std::vector<std::pair<float, Point<TREE_DIM>>> distance;
	for (int it = 0; it < all_pos.size(); ++it) {
		std::pair<float, Point<TREE_DIM>> tmp;
		tmp.second = all_pos[it].first;
		tmp.first = std::powf(pos_source[0] - all_pos[it].first[0], 2) +
			std::powf(pos_source[1] - all_pos[it].first[1], 2);
		distance.push_back(tmp);
	}

	std::sort(distance.begin(), distance.end(), comp_pair);

	num = std::min(num, (int)distance.size());
	for (int it = 0; it < num; ++it) {
		pos_list.push_back(distance[it].second);
	}
}

int main(int argc, char** argv)
{
	KDTreeTest();
	return 0;
}