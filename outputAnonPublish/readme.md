# Data files
## vpinfo.res 
csv file that contains metainformation about participants, columns are: 
* 1st participant id
* 2nd cohort (4, 7 or 10 month-old)
* 5th age in days
* 9th group id
* the remaining columns are irrelevant for this study
## log files in anonPublish
each csv file contains eye-tracking data. Filename consists of participant id, cohort and tobii or smi eye-tracking device indicator. The columns are: 
* 1 pc time (in seconds)
* 2 eye tracker time in microsecs
* 4 left eye horizontal gaze coordinate in degrees of visual angle assuming constant screen-to-eye distance of 70cm, origin at screen center,  positive values go in top-right direction from participants POV, Qpursuit class in Experiment.py includes details about the monitor geometry 
* 5 left eye vertical
* 6 right eye horizontal
* 7 right eye vertical
* 8-9 left and right eye pupil diameter, units as defined by the SMI API
* 10-12 left eye position in mm; 10=horizontal, 11=vertical and 12=depth coordinate, origin at eye tracker center
* 13-16 right eye position in mm 
