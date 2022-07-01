# ASReview_Language_Study
A study of the significance of language morphology in active learning aided systematic reviews using ASReview
To reproduce the study, perform the following steps:

1. pull the repository
2. navigate to asreview-0.19.2_for_language_study/ and install:
``` bash
pip install .
```
3. open the batch_builder.py script and configure which ASReview simulations you wish to run. The default settings may run more than a full day and create ~165 GB of files in output/
4. run batch_builder.py and check if the created simulation_batch.bat file is how you expected it
5. run simulation_batch.bat in your command prompt
6. change the options in calculate_metrics.py to match the batch(es) you ran
7. run calculate_metrics.py to create a table of WSS, RRF and ATD metrics
8. run visualise_metrics.py to create the recall curves
9. run ATD_plot.Rmd to create the ATD comparison plots

Vocabulary size and sparse words of all datasets can be calculated with vocabulary_size.py

Running repeated multilingual sbert simulations can take a long time. To run the feature extraction only once per setup, follow the instructions in batch_builder.py


