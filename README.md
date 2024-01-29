# Cloud Kitchen Delivery Network Optimization
**Streamlined food delivery networks, using Python and Linear Programming to minimize delivery travel distances.**

In this project, our team aims to establish a network of "cloud kitchens" in Portland, Oregon. The innovative business model involves deploying these kitchens in vacant commercial real estate and delivering meals to service stations using large drones. The primary objective is to efficiently guide the start-up in serving a selected city by leveraging Python programming and Linear Programming.

**_Task I: Data Gathering and Visualization_**

Gathering information on 25 business buildings suitable for cloud kitchen locations.

Coordinates of these locations are obtained using Python libraries, and the data is visualized on a map with optimal boundaries.

A sample of 50 service station locations is generated randomly for further analysis.

A Python function calculates Euclidean distances between cloud kitchens and service stations.

_**Task II: Optimization Model Formulation and Solution**_

Formulation of an assignment problem using the PuLP library with specified parameters and variables.

Solving the optimization model to determine the best assignment of cloud kitchens to service stations, minimizing total travel distance.

Storing the resulting assignments using a sparse data structure.

_**Task III: Analysis and Visualization**_

Construction of an "Origin and Destination (OD)" table showcasing selected pairs based on the optimization model.

Visualization of the assignment solution on the map.

Creation of a frequency graph for different distance ranges to analyze the distribution of Origin-Destination assignments.

_**Deliverables:**_

Well-documented Python program ("Codebase.py") generating necessary files.

Locations table ("Locations.txt") detailing cloud kitchen and service station information.

Map of locations ("Locations.jpeg") for visual reference.

Distances matrix ("Distances.csv") reflecting calculated Euclidean distances.

Instantiated formulation ("AP.mps") for the optimization model.

OD table ("OD.txt") representing selected Origin-Destination pairs.

Frequency graph ("Frequency.jpeg") illustrating distribution across distance ranges.

Solution map ("Solution.jpeg") for visualizing the optimized assignment.
