#pragma once

#include <cmath>
#include <algorithm>

enum Axis_Enum {
    Axis_X = 0,
    Axis_Y,
    Axis_Z
};

template <std::size_t N>
class Point {
public:
    Point();

    typedef double* iterator;
    typedef const double* const_iterator;

    std::size_t size() const;

    double& operator[](std::size_t index);
    double operator[](std::size_t index) const;

    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;

private:
    double coords[N];
};

template <std::size_t N>
double SquaredDistance(const Point<N>& one, const Point<N>& two);

template <std::size_t N>
double Distance(const Point<N>& one, const Point<N>& two);

template <std::size_t N>
bool operator==(const Point<N>& one, const Point<N>& two);

template <std::size_t N>
bool operator!=(const Point<N>& one, const Point<N>& two);

template <std::size_t N>
Point<N>::Point(){
    for (int it = 0; it < N; ++it) {
        coords[it] = 0;
    }
}

template <std::size_t N>
std::size_t Point<N>::size() const {
    return N;
}

template <std::size_t N>
double& Point<N>::operator[] (std::size_t index) {
    return coords[index];
}

template <std::size_t N>
double Point<N>::operator[] (std::size_t index) const {
    return coords[index];
}

template <std::size_t N>
typename Point<N>::iterator Point<N>::begin() {
    return coords;
}

template <std::size_t N>
typename Point<N>::const_iterator Point<N>::begin() const {
    return coords;
}

template <std::size_t N>
typename Point<N>::iterator Point<N>::end() {
    return begin() + size();
}

template <std::size_t N>
typename Point<N>::const_iterator Point<N>::end() const {
    return begin() + size();
}

template <std::size_t N>
double Distance(const Point<N>& one, const Point<N>& two) {
    double result = 0.0;
    for (std::size_t i = 0; i < N; ++i)
        result += std::powf(one[i] - two[i], 2.0);
    return std::sqrt(result);
}

template <std::size_t N>
double SquaredDistance(const Point<N>& one, const Point<N>& two) {
    double result = 0.0;
    for (std::size_t i = 0; i < N; ++i)
        result += std::powf(one[i] - two[i], 2.0);
    return result;
}

template <std::size_t N>
bool operator==(const Point<N>& one, const Point<N>& two) {
    return std::equal(one.begin(), one.end(), two.begin());
}

template <std::size_t N>
bool operator!=(const Point<N>& one, const Point<N>& two) {
    return !(one == two);
}