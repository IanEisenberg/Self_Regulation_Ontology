#!/usr/bin/env python3
"""
validate metadata entry

see README.md for outline of each entry

using lowerCamelCase for consistency with NDA Data Dictionary

"""
from jsonschema import validate


def validate_exp(data):
    assert isinstance(data,dict)
    schema={}
    schema['Instrument']={
        "type" : "object",
        "properties" : {
            "title" : {"type" : "string"},
            "expFactoryName" : {"type" : "string"},
            "shortName" : {"type" : "string"},
            "dataElements" : {"type":"object"},
            "URL" : {"type" : "object"},
            "description" : {"type" : "string"},
            },
            "required": ["title", "expFactoryName",
                        "dataElements"]
        }
    schema['dataElement']={
        "type" : "object",
        "properties" : {
            "NDAElementName" : {"type" : "string"},
            "expFactoryName" : {"type" : "string"},
            "variableType" : {'enum':['string','integer',
                                'float','boolean']},
            "variableScope" : {'enum':['item','subscale',
                                        'instrument']},
            "variableDefinition" : {"type":"object"},
            "variableUnits" : {"enum" : ['ordinal','seconds',
                                        'milliseconds',
                                        'probability','percentage',
                                        'rate','count',
                                        'arbitrary','other']},
            "variableClass" : {'enum':['surveyResponse','surveySummary',
                                'responseTime','accuracy',
                                'DDMDriftRate','DDMNondecisionTime',
                                'DDMThreshold','learningRate',
                                'load','discountRate','span','other']},
            "variableModel": {'type':'string',
                            'description':'name of model used to generate variable (e.g. "HDDM")'},
            "dataElements" : {"type":"object"},
            "size" : {"type" : "integer"},
            "isRequired" : {'enum':['required','recommended']},
            "description" : {"type" : "string"},
            "valueRange" : {"type" : "string"},
            },
            "required": ["expFactoryName"]
        }
    validate(data,schema['Instrument'])
    for i in data['dataElements']:
        validate(i,schema['dataElement'])

if __name__=="__main__":
    foo={'dataElements': [],
        'expFactoryName': 'asdf',
        'shortName': 's',
        'title': 'foo'}
    foo['dataElements'].append({'expFactoryName':'item1',
                                'varType':'string'})
    validate_exp(foo)
