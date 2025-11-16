/*
@File    :   parser.cpp
@Time    :   2025/11/16 09:12:44
@Author  :   Lin 
@Version :   1.0
@Desc    :   Parse the answers from the Json files.
copyright USTC
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <string>
#include <vector>

template <typename T> using vT = std::vector<T>;

vT<int> KMP(const std::string &s, const std::string &t) {
  int ns = s.size(), nt = t.size();
  vT<int> next(nt + 1); 
  vT<int> indexs;
  for (int i = 1, j = 0; i < nt; ++i) {
    while (j && t[i] != t[j])
      j = next[j];
    if (t[i] == t[j])
      ++j;
    next[i + 1] = j;
  }

  for (int i = 0, j = 0; i < ns; ++i) {
    while (j && s[i] != t[j])
      j = next[j];
    if (s[i] == t[j])
      ++j;
    if (j == nt) {
      indexs.push_back(i + 1);
      j = next[j];
    }
  }
  return indexs;
}

void KMP(const std::string &s, const std::string &t, vT<int> &indexs) {
  int ns = s.size(), nt = t.size();
  vT<int> next(nt + 1); 
  for (int i = 1, j = 0; i < nt; ++i) {
    while (j && t[i] != t[j])
      j = next[j];
    if (t[i] == t[j])
      ++j;
    next[i + 1] = j;
  }

  for (int i = 0, j = 0; i < ns; ++i) {
    while (j && s[i] != t[j])
      j = next[j];
    if (s[i] == t[j])
      ++j;
    if (j == nt) {
      indexs.push_back(i + 1);
      j = next[j];
    }
  }
}

vT<std::string> extract_Math_cpp(std::string &s) {
  vT<std::string> ans;
  vT<int> indexs = KMP(s, "boxed{");
  int sz = s.size();
  for (auto i : indexs) {
    for (int j = i, n = 0; j < sz; ++j) {
      if (s[j] == '{')
        ++n;
      else if (s[j] == '}') {
        --n;
        if (n < 0) {
          if (j + 1 < sz && s[j + 1] == '%')
            ans.push_back(s.substr(i, j - i + 1));
          else
            ans.push_back(s.substr(i, j - i));
          break;
        }
      }
    }
  }
  return ans;
}

vT<std::string_view> extract_Math_cpp_view(const std::string &s) {
  vT<std::string_view> ans;
  vT<int> indexs;
  int sz = s.size();
  KMP(s, "boxed{", indexs);

  for (int i : indexs) {
    for (int j = i, n = 0; j < sz; ++j) {
      if (s[j] == '{') {
        ++n;
      } else if (s[j] == '}') {
        --n;
        if (n < 0) {
          if (j + 1 < sz && s[j + 1] == '%') {
            ans.emplace_back(&s[i], j - i + 1);
          } else {
            ans.emplace_back(&s[i], j - i);
          }
          break;
        }
      }
    }
  }
  return ans;
}

PYBIND11_MODULE(EMath, m) {
  // m.def("extract_Math_cpp", &extract_Math_cpp, "support math dataset");
  m.def("extract_Math", &extract_Math_cpp_view,
        "support math dataset via c++20");
}