

This is a common directory used by all models. Any common code/library that
is used by models belongs here.

External libraries can be added to the requirements.txt in this directory.
When executor service starts up, it executes setup.sh script. Which will
install the dependencies defined in requirement.txt.
