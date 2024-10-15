# NYU Course Notebooks

This repository contains the sample notebooks needed for the demonstrations used in the course. 
This contains of the following directories: 

- demos: contain notebooks which showcase various topics/code discussed in class
- demos/data: contains the data sets that are used in the class
- unsw: contains one day log from UNSW dataset which has been processed through Zeek preprocessing

- The UNSW dataset is referenced in the paper
- Sivanathan, A., et al.: Characterizing and classifying IoT traﬃc in smart cities and campuses. In: IEEE INFOCOM Workshop Smart Cities and Urban Computing (2017)

## Installation


Setup a virtual environment for python using the following commands
(You can use your own environment name instead of using the provided name of env_demos: 
Use python 3.9 or higher version 

```
$ python -m venv env_demos
$ source env_demos/bin/activate
```

Next, install the required packages:
```
$ pip install -r requirements.txt
```

## Execution 
After installation, you can open a terminal on your machine and start jupyter notebook in the demos directory

```
$ jupyter notebook
```

This should open the jupyter notebook in a browser and you can brosw and run the selected notebooks. 

