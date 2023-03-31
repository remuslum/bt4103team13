import json
import pickle
from functions import *
import numpy as np
import boto3

with open('skills_api.pkl', 'rb') as pickle_file:
    skills_api = pickle.load(pickle_file)

s3_client = boto3.client('s3')
s3_bucket = 'nus-sambaash'
s3_folderpath = 'skills-engine/resumes/'

print('done configuring')


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

print('start invoking lambda')

def lambda_handler(event, context):
    # if 'doc_name' not in event:
    #     return {
    #     'statusCode': 400,
    #     'body': json.dumps("'doc_name' not found in the keys of request")
    # }

    # doc_name = event['doc_name']
    # file_content = s3_client.get_object(Bucket=s3_bucket, Key=s3_folderpath+doc_name)["Body"].read()
    # print('done reading resume')
    # resume_string = read_resume(file_content)
    # print(resume_string)
    if 'doc' not in event:
        return {
        'statusCode': 400,
        'body': json.dumps("'doc' not found in the keys of request")
    }
    resume_string = event['doc']
    res = skills_experience_level_identification(resume_string, skills_api)
    # try:
    #     response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    #     file_contents = response['Body'].read()
    # except Exception as e:
    #     return {
    #         'statusCode': 400,
    #         'body': json.dumps({'message': 'File not found'})
    #     }

    json_txt = json.dumps({'skills': res}, cls=NpEncoder, sort_keys = True)

    return {
        'statusCode': 200,
        'body': json.dumps({'resume': json_txt})
    }

# try:

#     import json
#     import sys
#     import numpy
#     import pandas
#     import spacy

#     print("All imports ok ...")
# except Exception as e:
#     print("Error Imports : {} ".format(e))


# def lambda_handler(event, context):

#     print("Hello AWS!")
#     print("event = {}".format(event))
#     return {
#         'statusCode': 200,
#     }