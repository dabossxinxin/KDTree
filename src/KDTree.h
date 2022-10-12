#pragma once

//#define VISUALIZATION_KDTREE

#include "Point.h"
#include "BoundedPQueue.h"
#include <stdexcept>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <utility>
#include <algorithm>

#ifdef VISUALIZATION_KDTREE
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif

const double KD_TREE_MIN_X = 0.;
const double KD_TREE_MAX_X = 8000.;
const double KD_TREE_MIN_Y = 0.;
const double KD_TREE_MAX_Y = 8000.;

#ifdef VISUALIZATION_KDTREE
const int rows = KD_TREE_MAX_Y - KD_TREE_MIN_Y + 1;
const int cols = KD_TREE_MAX_X - KD_TREE_MIN_X + 1;
cv::Mat visualization_kdtree = cv::Mat(rows, cols, CV_8UC3, cv::Scalar(255, 255, 255));
#endif

template <std::size_t N, typename ElemType>
class KDTree {
public:
	// 默认构造函数
    KDTree();

	// 使用特定结构初始化KD-Tree
    KDTree(std::vector<std::pair<Point<N>, ElemType>>& points);

    // 析构KD-Tree内存空间
    ~KDTree();

    // KD-Tree复制构造和拷贝构造函数
    KDTree(const KDTree& rhs);
    KDTree& operator=(const KDTree& rhs);

    // 当前树节点的维度
    std::size_t dimension() const;

    // 当前树中节点的个数
    std::size_t size() const;
    bool empty() const;

    // 当前树中是否包含特定点pt
    bool contains(const Point<N>& pt) const;

    // 在当前树中插入特定节点，若已存在，则覆盖
    void insert(const Point<N>& pt, const ElemType& value=ElemType());

	// 返回特定节点对应的关键值，若无该节点则插入一个
    ElemType& operator[](const Point<N>& pt);

	// 返回特定节点对应的关键值，若无该节点则抛出异常
    ElemType& at(const Point<N>& pt);
    const ElemType& at(const Point<N>& pt) const;
    Point<N>& at(const ElemType& elem);
    const Point<N>& at(const ElemType& elem) const;

    // 获取给定点位的k个最近邻节点，并返回出现次数最多的关键值
    ElemType kNNValue(const Point<N>& key, std::size_t k) const;

private:
    struct Node {
		Point<N>	point;
        Node		*left;      // 左节点
		Node		*right;     // 右节点
        Node		*parent;    // 父节点
        int			level;      // 树深度
        ElemType	value;		// 关键字

#ifdef VISUALIZATION_KDTREE
        double min_x = KD_TREE_MIN_X;
        double min_y = KD_TREE_MIN_Y;
        double max_x = KD_TREE_MAX_X;
        double max_y = KD_TREE_MAX_Y;
#endif

        Node(const Point<N>& _pt, int _level, const ElemType& _value=ElemType()):
            point(_pt), left(NULL), right(NULL), parent(NULL), level(_level), value(_value) {}
    };

    // 给定特定结构创建树
    Node* buildTree(typename std::vector<std::pair<Point<N>, ElemType>>::iterator start,
                    typename std::vector<std::pair<Point<N>, ElemType>>::iterator end, int currLevel);

    // 给定特定关键字或节点返回树节点指针
    Node* findNode(Node* currNode, const Point<N>& pt) const;
    Node* findNode(Node* currNode, const ElemType& elem) const;

    // 迭代求解设定点位在树中的k近邻
    void nearestNeighborRecurse(const Node* currNode, const Point<N>& key, BoundedPQueue<ElemType>& pQueue) const;

    // 深拷贝树节点
    Node* deepcopyTree(Node* root);

    // 析构树节点
    void freeResource(Node* currNode);

    // 将KD-Tree的分布情况绘制在图像中
    void drawKDTree(Node*& root);

private:
    Node            *root_;			// 树根节点
    std::size_t     size_;			// 树中元素数量
    Axis_Enum       axis_enum_;
};


template <std::size_t N, typename ElemType>
KDTree<N, ElemType>::KDTree() :
    root_(NULL), size_(0) {}

template <std::size_t N, typename ElemType>
typename KDTree<N, ElemType>::Node* KDTree<N, ElemType>::deepcopyTree(typename KDTree<N, ElemType>::Node* root) 
{
    if (root == NULL) return NULL;
    Node* newRoot = new Node(*root);
    newRoot->left = deepcopyTree(root->left);
    newRoot->right = deepcopyTree(root->right);
    return newRoot;
}

template <std::size_t N, typename ElemType>
typename KDTree<N, ElemType>::Node* KDTree<N, ElemType>::buildTree(typename std::vector<std::pair<Point<N>, ElemType>>::iterator start,
                                                                   typename std::vector<std::pair<Point<N>, ElemType>>::iterator end, int currLevel)
{
    if (start >= end) return NULL;

    int axis = currLevel % N;
    auto cmp = [axis](const std::pair<Point<N>, ElemType>& p1, const std::pair<Point<N>, ElemType>& p2) {
        return p1.first[axis] < p2.first[axis];
    };

	// 获取当前维度的中值
    std::size_t len = end - start;
    auto mid = start + len / 2;
    std::nth_element(start, mid, end, cmp);
    while (mid > start && (mid - 1)->first[axis] == mid->first[axis]) {
        --mid;
    }

    Node* newNode = new Node(mid->first, currLevel, mid->second);
    newNode->left = buildTree(start, mid, currLevel + 1); 
    if (newNode->left != NULL) newNode->left->parent = newNode;
    newNode->right = buildTree(mid + 1, end, currLevel + 1);
    if (newNode->right != NULL) newNode->right->parent = newNode;
    return newNode;
}

template <std::size_t N, typename ElemType>
void KDTree<N, ElemType>::drawKDTree(KDTree::Node*& root)
{
    if (root == NULL || (root->left == NULL && root->right)) return;

    cv::circle(visualization_kdtree, cv::Point(root->point[0], root->point[1]), 1, cv::Scalar(0, 0, 255));
    
    if (root->level != 0) {
        int axis = root->parent->level % N;
        if (axis == 0) {
            root->min_y = root->parent->min_y;
            root->max_y = root->parent->max_y;
            if (root->point[axis] < root->parent->point[axis]) {
                root->min_x = root->parent->min_x;
                root->max_x = root->parent->point[axis];
            }
            else {
                root->min_x = root->parent->point[axis];
                root->max_x = root->parent->max_x;
            }
        }
        else if (axis == 1) {
            root->min_x = root->parent->min_x;
            root->max_x = root->parent->max_x;
            if (root->point[axis] < root->parent->point[axis]) {
                root->min_y = root->parent->min_y;
                root->max_y = root->parent->point[axis];
            }
            else {
                root->min_y = root->parent->point[axis];
                root->max_y = root->parent->max_y;
            }
        }
    }

    cv::line(visualization_kdtree, cv::Point(root->min_x, root->min_y), cv::Point(root->min_x, root->max_y), cv::Scalar(0, 0, 0));
    cv::line(visualization_kdtree, cv::Point(root->min_x, root->max_y), cv::Point(root->max_x, root->max_y), cv::Scalar(0, 0, 0));
    cv::line(visualization_kdtree, cv::Point(root->max_x, root->max_y), cv::Point(root->max_x, root->min_y), cv::Scalar(0, 0, 0));
    cv::line(visualization_kdtree, cv::Point(root->max_x, root->min_y), cv::Point(root->min_x, root->min_y), cv::Scalar(0, 0, 0));

    drawKDTree(root->left);
    drawKDTree(root->right);
}

template <std::size_t N, typename ElemType>
KDTree<N, ElemType>::KDTree(std::vector<std::pair<Point<N>, ElemType>>& points) 
{
    root_ = buildTree(points.begin(), points.end(), 0);
    size_ = points.size();
#ifdef VISUALIZATION_KDTREE
    drawKDTree(root_);
#endif
}

template <std::size_t N, typename ElemType>
KDTree<N, ElemType>::KDTree(const KDTree& rhs) 
{
    root_ = deepcopyTree(rhs.root_);
    size_ = rhs.size_;
}

template <std::size_t N, typename ElemType>
KDTree<N, ElemType>& KDTree<N, ElemType>::operator=(const KDTree& rhs)
{
    if (this != &rhs) {
        freeResource(root_);
        root_ = deepcopyTree(rhs.root_);
        size_ = rhs.size_;
    }
    return *this;
}

template <std::size_t N, typename ElemType>
void KDTree<N, ElemType>::freeResource(typename KDTree<N, ElemType>::Node* currNode) 
{
    if (currNode == NULL) return;
    freeResource(currNode->left);
    freeResource(currNode->right);
    delete currNode;
    currNode = nullptr;
}

template <std::size_t N, typename ElemType>
KDTree<N, ElemType>::~KDTree()
{
    freeResource(root_);
    size_ = 0;
}

template <std::size_t N, typename ElemType>
std::size_t KDTree<N, ElemType>::dimension() const 
{
    return N;
}

template <std::size_t N, typename ElemType>
std::size_t KDTree<N, ElemType>::size() const 
{
    return size_;
}

template <std::size_t N, typename ElemType>
bool KDTree<N, ElemType>::empty() const 
{
    return size_ == 0;
}

template <std::size_t N, typename ElemType>
typename KDTree<N, ElemType>::Node* KDTree<N, ElemType>::findNode(typename KDTree<N, ElemType>::Node* currNode, const Point<N>& pt) const 
{
    if (currNode == NULL || currNode->point == pt) return currNode;

    const Point<N>& currPoint = currNode->point;
    int currLevel = currNode->level;
    if (pt[currLevel%N] < currPoint[currLevel%N]) {
        return currNode->left == NULL ? currNode : findNode(currNode->left, pt);
    } else {
        return currNode->right == NULL ? currNode : findNode(currNode->right, pt);
    }
}

// 这里使用哈希表实现会有更低的时间复杂度
template <std::size_t N, typename ElemType>
typename KDTree<N, ElemType>::Node* KDTree<N,ElemType>::findNode(typename KDTree<N, ElemType>::Node* currNode, const ElemType& elem) const
{
    if (currNode == NULL || currNode->value == elem) return currNode;
    
    const ElemType& type = currNode->value;
    Node* leftNode = findNode(currNode->left, elem);
    Node* rightNode = findNode(currNode->right, elem);

    if (leftNode == NULL && rightNode == NULL) return NULL;
    else if (leftNode != NULL) return leftNode;
    else if (rightNode != NULL) return rightNode;
    else return NULL;
}

template <std::size_t N, typename ElemType>
bool KDTree<N, ElemType>::contains(const Point<N>& pt) const 
{
    auto node = findNode(root_, pt);
    return node != NULL && node->point == pt;
}

template <std::size_t N, typename ElemType>
void KDTree<N, ElemType>::insert(const Point<N>& pt, const ElemType& value) 
{
    auto targetNode = findNode(root_, pt);

    // 树为空
    if (targetNode == NULL) {
        root_ = new Node(pt, 0, value);
        size_ = 1;
    } else {
        // 当前树中已经存在该节点，更新该节点的值
        if (targetNode->point == pt) {
            targetNode->value = value;
        // 当前树中不存在该节点，将该节点插入当前树
        } else {
            int currLevel = targetNode->level;
            Node* newNode = new Node(pt, currLevel + 1, value);
            if (pt[currLevel%N] < targetNode->point[currLevel%N]) {
                targetNode->left = newNode;
            } else {
                targetNode->right = newNode;
            }
            ++size_;
        }
    }
}

template <std::size_t N, typename ElemType>
const ElemType& KDTree<N, ElemType>::at(const Point<N>& pt) const
{
    auto node = findNode(root_, pt);
    if (node == NULL || node->point != pt) {
        throw std::out_of_range("Point not found in the KD-Tree");
    } else {
        return node->value;
    }
}

template <std::size_t N, typename ElemType>
ElemType& KDTree<N, ElemType>::at(const Point<N>& pt)
{
    const KDTree<N, ElemType>& constThis = *this;
    return const_cast<ElemType&>(constThis.at(pt));
}

template <std::size_t N, typename ElemType>
Point<N>& KDTree<N, ElemType>::at(const ElemType& elem)
{
    const KDTree<N, ElemType>& constThis = *this;
    return const_cast<Point<N>&>(constThis.at(elem));
}

template <std::size_t N, typename ElemType>
const Point<N>& KDTree<N, ElemType>::at(const ElemType& elem) const
{
    auto node = findNode(root_, elem);
    if (node == NULL || node->value != elem) {
        throw std::out_of_range("Point not found in the KD-Tree");
    }
    else {
        return node->point;
    }
}

template <std::size_t N, typename ElemType>
ElemType& KDTree<N, ElemType>::operator[](const Point<N>& pt) 
{
    auto node = findNode(root_, pt);
	// 当前树中具有节点pt
    if (node != NULL && node->point == pt) {
        return node->value;
	// 当前树中没有节点pt,使用默认elem插入该节点
    } else { 
        insert(pt);
        if (node == NULL) return root_->value;
        else return (node->left != NULL && node->left->point == pt) ? node->left->value: node->right->value;
    }
}

template <std::size_t N, typename ElemType>
void KDTree<N, ElemType>::nearestNeighborRecurse(const typename KDTree<N, ElemType>::Node* currNode, 
	const Point<N>& key, BoundedPQueue<ElemType>& pQueue) const
{
    if (currNode == NULL) return;
    const Point<N>& currPoint = currNode->point;
    pQueue.enqueue(currNode->value, Distance(currPoint, key));

    // 在树的一侧分支中搜索最邻近的节点
    int currLevel = currNode->level;
    bool isLeftTree;
    if (key[currLevel%N] < currPoint[currLevel%N]) {
        nearestNeighborRecurse(currNode->left, key, pQueue);
        isLeftTree = true;
    } else {
        nearestNeighborRecurse(currNode->right, key, pQueue);
        isLeftTree = false;
    }

    // 若搜索数量不满足或优先级不满足，则在树的另一分支继续搜索
    if (pQueue.size() < pQueue.maxSize() || fabs(key[currLevel%N] - currPoint[currLevel%N]) < pQueue.worst()) {
        if (isLeftTree) nearestNeighborRecurse(currNode->right, key, pQueue);
        else nearestNeighborRecurse(currNode->left, key, pQueue);
    }
}

template <std::size_t N, typename ElemType>
ElemType KDTree<N, ElemType>::kNNValue(const Point<N>& key, std::size_t k) const 
{
#ifdef VISUALIZATION_KDTREE
    cv::circle(visualization_kdtree, cv::Point(key[0], key[1]), 4, cv::Scalar(0, 255, 0));
#endif

    BoundedPQueue<ElemType> pQueue(k);
    if (empty()) return ElemType();

    nearestNeighborRecurse(root_, key, pQueue);

#ifdef VISUALIZATION_KDTREE
    while (!pQueue.empty()) {
        ElemType elem = pQueue.dequeueMin();
        Point<N> pt = this->at(elem);
        cv::circle(visualization_kdtree, cv::Point(pt[0], pt[1]), 6, cv::Scalar(255, 0, 0));
		std::cout << pt[0] << "," << pt[1] << ",";
    }
	std::cout << std::endl;
#endif

    std::unordered_map<ElemType, int> counter;
    while (!pQueue.empty()) {
        ++counter[pQueue.dequeueMin()];
    }

    ElemType result;
    int cnt = -1;
    for (const auto &p : counter) {
        if (p.second > cnt) {
            result = p.first;
            cnt = p.second;
        }
    }
    return result;
}