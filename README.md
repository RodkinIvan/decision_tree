# Decision Tree

## Classifier

There is minimalistic realization of Decision Tree on C++ which can be also used as python module
-[ ] All the realization of tree in __src/Tree.*__

-[ ] The src/py_module is necessary for building python module from it using boost::python

-[ ] There are some tests in main, which can be ran be uncommenting add_executable in CMakeLists.txt

-[ ] Dynamic library module/decision_tree.so acts like a python module, so, for example, being in the same folder with it, you can import it using
```
import decision_tree as tree
```
or
```
from decision_tree import decision_tree_classifier
```
etc.

