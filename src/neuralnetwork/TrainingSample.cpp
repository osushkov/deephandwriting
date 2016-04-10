

#include "TrainingSample.hpp"
#include <iostream>

std::ostream &operator<<(std::ostream &stream, const TrainingSample &ts) {
  stream << ts.input << " : " << ts.expectedOutput;
  return stream;
}
