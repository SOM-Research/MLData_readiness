### This is the data supporting the study: "On the Readiness of Scientific Data for a Fair and Transparent Use in Machine learning"
[![DOI](https://zenodo.org/badge/692880864.svg)](https://zenodo.org/doi/10.5281/zenodo.10514145)


In this repository you will find:

1 - **Full Results**: The full results of the extraction process containing 4041 data papers annotated using the scripts in the root of this project

The *fullResults.xlsx* file contains the whole results of the extraction process, and the *ResultsSData.xlsx* and *ResultsDBrief.xlsx* contanins the results for each journal.

2 - **Analysis sheet**: The sheet with the charts, counts and analysis done to write the study

The *FullStudyAnalysis.xlsx* contains the full data, the charts, the topic analysis, and high-level insights of the data

3 - **Code**: The code used to extract the data. One for each journal. This will help into replicating the experiment.

*dataPaperScrapping.ipynb* notebook contains the code used to filter all the data papers type of both journals, and get the PDF (when possible). If you want to reproduce the experiment you may start by this notebook.

Once you have all the PDF of the journals, *SDataExtractor.py* and *DBriefExtractor.py* contains the code to perform the extraction for each journal. Note you will need and OpenAI ApiKey and a GROBID service running to execute the notebooks.




