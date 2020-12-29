# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.


## This was my first project with hands on development for building and training neural networks.

This project is well suited to show the force and possibilities of Classification supervised learning tasks and how such models can, and nowadays are regularly applied in products and productions. For a mainstream task as such direct classifications simpler solutions can be found for the time being, however for finding tailormade solutions of the mainline pytorch and tensorflow projects are the industry standard for some time to come.

### A note on the overall outcome of the project:

The sanity-check shows that there is an underlying issue in the data, for which the sanity-checks are actually performed.
There have been numerous competitions for model performance on platforms like Kaggle, where models where showing "too good to be true" accuracies on the testsets, while when brought to productions  those models showed failure by completely wrong prediciting objects and others. 
While mislabeling can be a source of failure, the primary cause in general is [data leakage](https://www.researchgate.net/publication/221653692_Leakage_in_Data_Mining_Formulation_Detection_and_Avoidance), in which amongs other reason - due to failures in dataset preparations - the model collected informations from the testset prior to training, which should never be the case, as this should be entirely unknown information to the model.






