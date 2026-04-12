# tornotify-ml

## Inspiration
Back on September 24, 2001, a devastating tornado crossed over areas of North Campus. 2 students killed, 57 injured, 300 cars destroyed, damaged dorms and completely shredded trees. Since this incident, the University implemented tornado sirens to enhance early warning communications in the hopes to save more lives. However, tornado detection has been a significantly challenging task for forecasters at weather offices. 

Several years ago, the MIT Lincoln Laboratory published a benchmark dataset, allowing researchers utilize machine learning models to detect tornadic signatures in weather radar data.  We wanted to utilize this model to build an architecture that detects tornados with realtime weather radar data with the goal of finding earlier detection to give people more time to take shelter.

## What it does
The model ingests realtime and archived Next Generation Weather Radar data (NEXRAD) served on an AWS S3 Bucket. The latest frame from each radar site first has a filter that parses for high reflectivity, indicating present storm cells. If cells are present, the data is cropped and the cell gets extrapolated and passed into the next phase for tornado classification. After the cell is processed by the model, areas are plotted on a map with their respective probabilities of being a tornado. 

In order to ensure predictions are not anomalies, the program will aggregate future radar time steps to ensure consistency. If a tornado is likely, its path will be plotted on the map.

The user can also observe past weather events in the dashboard dating back to 2013 for all 50 states.

## How we built it
All of the system is written in Python. We utilized hugging face to import MIT model and connect it to our live data ingestion function that we wrote using the nexradaws library. All radar images with state and county borders were generated utilizing pyart and cartopy. The dashboard built to showcase the system in action was created with the streamlit library.

## Challenges we ran into
Single frames would sometimes return false positive indications for tornado signatures and we had to come up with a strategy to ensure only persistent detections were flagged. This required validation over multiple time steps in radar data. If the next time step showed a similar probability for a tornado in the same region that followed a computed storm track movement, the system would not consider the tornado event an anomaly and elevate detection to the user.


## Accomplishments that we're proud of
Training data spanned between 2013 and 2022. We wanted to see if the system would accurately determine tornado events beyond the training set. A key accomplishment was when we passed archived radar data from the 2023 Rolling Fork Tornado in Mississippi and had a accurate classification and storm path. We then tried this on other recent tornados outside of the dataset and saw similar success. 

## What we learned
Before undertaking this project, we did not know much about interpreting radar data and atmospheric science as a whole. Understanding the six components of modern radar including reflectivity, radial velocity, spectrum width, differential reflectivity, correlation coefficient, and specific differential phase helped us to understand what the model was processing and how to interpret performance on different severe weather examples.

## What's next for Tornotify - Storm Early Warning Systems
We want to go beyond standard mobile phone warning systems and integrate our detection system into smart home devices. This would include stereo systems playing warnings, lights changing colors, and other smart features that could warn people away from their phones as well as people with sensory disabilities. 
