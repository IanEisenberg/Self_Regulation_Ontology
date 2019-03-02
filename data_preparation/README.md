The code in this directory is used to prepare the data for processing.

* process_itemlevel.py: generate separate tsv and json files in the Derived_data
directory for each survey
* process_health.py: process k6_health data
* process_alcohol_drug.py: process data from drug/alcohol questionnaire
* process_demographics.py: process demographic data
* cleanup_items_for_mirt_cv.py: collapse survey responses that occur too infrequently

To run everything, you can use:

    make dataprep

### Metadata

To prepare the metadata, run

    make metadata

The metadata schema is aligned as closely as possible to the
(NIMH Data Archive Data Dictionary) [https://ndar.nih.gov/data_dictionary.html].
It contains the following entries:

For each instrument (task or survey):
* **title**: full text name of instrument
* **description**: detailed description of instrument
* **expFactoryName**: instrument name in Experiment Factory
* **shortName**: short name for instrument (from NDADD)
* **URL**:
  * **CognitiveAtlasURL**: URL for instrument in Cognitive Atlas
  * **NDAURL**: URL for instrument in NDA Data Dictionary
* **dataElements**: a list of data elements (e.g. items, variables)

For each data element within an instrument:
* **expFactoryName**: item identifier
* **NDAElementName**: identifier for item from NIH Data Archive DD
* **variableType**: one of ['string','integer','float','boolean']
* **variableScope**: one of ['item','subscale','instrument']
* **variableDefinition**: None, or a dictionary defining the ElementNames included in the subscale and their weights
* **variableUnits**: None, or one of ['ordinal','seconds','probability','percentage','rate','count','other']
* **variableClass**: one of ['surveyResponse','surveySummary',
'responseTime','accuracy','DDMDriftRate','DDMNondecisionTime',
'DDMThreshold','learningRate','load','discountRate','span']
* **size**: max size for strings,None otherwise
* **isRequired**: one of ['required','recommended']
* **description**: full text description of item (for survey items this is the item text)
* **valueRange**: range of possible values (defined as "min::max")
* **notes**: For survey items, including semicolon delimited list of
