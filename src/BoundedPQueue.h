#pragma once

#include <map>
#include <algorithm>
#include <limits>

template <typename T>
class BoundedPQueue {
public:
    
    explicit BoundedPQueue(std::size_t maxSize);

    void enqueue(const T& value, double priority);

    T dequeueMin();

    std::size_t size() const;
    bool empty() const;

    std::size_t maxSize() const;

    double best()  const;
    double worst() const;

private:
    std::multimap<double, T> elems;
    std::size_t maximumSize;
};

template <typename T>
BoundedPQueue<T>::BoundedPQueue(std::size_t maxSize) {
    maximumSize = maxSize;
}

template <typename T>
void BoundedPQueue<T>::enqueue(const T& value, double priority) 
{
    elems.insert(std::make_pair(priority, value));

    if (size() > maxSize()) {
        typename std::multimap<double, T>::iterator last = elems.end();
        --last;
        elems.erase(last);
    }
}

template <typename T>
T BoundedPQueue<T>::dequeueMin() 
{
    T result = elems.begin()->second;
    elems.erase(elems.begin());
    return result;
}

template <typename T>
std::size_t BoundedPQueue<T>::size() const {
    return elems.size();
}

template <typename T>
bool BoundedPQueue<T>::empty() const {
    return elems.empty();
}

template <typename T>
std::size_t BoundedPQueue<T>::maxSize() const 
{
    return maximumSize;
}

template <typename T>
double BoundedPQueue<T>::best() const {
    return empty()? std::numeric_limits<double>::infinity() : elems.begin()->first;
}

template <typename T>
double BoundedPQueue<T>::worst() const {
    return empty()? std::numeric_limits<double>::infinity() : elems.rbegin()->first;
}