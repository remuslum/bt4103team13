import json
import boto3
import io
from datetime import datetime


def update_metric_data():
    """
    Overall function to generate Cloudwatch log data for QuickSight Dashboard.
    Pulls required cloudwatch log data, reformat for QuickSight, uploads to S3

    """
    cloudwatch = boto3.client("cloudwatch")
    response = cloudwatch.get_metric_data(
        MetricDataQueries=[
            {
                'Id':'invocations',
                'MetricStat': {
                    'Metric': {
                        "Namespace": "AWS/Lambda",
                        "MetricName": "Invocations",
                        "Dimensions": [
                            {
                                "Name": "FunctionName",
                                "Value": "skills_extract"
                            }
                        ]
                    },
                    'Period': 300,
                    'Stat': 'Sum',
                    
                }
            },
            {
                'Id':'duration',
                'MetricStat': {
                    'Metric': {
                        "Namespace": "AWS/Lambda",
                        "MetricName": "Duration",
                        "Dimensions": [
                            {
                                "Name": "FunctionName",
                                "Value": "skills_extract"
                            }
                        ]
                    },
                    'Period': 300,
                    'Stat': 'Average',
                    'Unit': 'Seconds'
                }
            },
            {
                'Id':'max_duration',
                'MetricStat': {
                    'Metric': {
                        "Namespace": "AWS/Lambda",
                        "MetricName": "Duration",
                        "Dimensions": [
                            {
                                "Name": "FunctionName",
                                "Value": "skills_extract"
                            }
                        ]
                    },
                    'Period': 300,
                    'Stat': 'Maximum',
                    'Unit': 'Seconds'
                }
            },
            {
                'Id':'error',
                'MetricStat': {
                    'Metric': {
                        "Namespace": "AWS/Lambda",
                        "MetricName": "Error",
                        "Dimensions": [
                            {
                                "Name": "FunctionName",
                                "Value": "skills_extract"
                            }
                        ]
                    },
                    'Period': 300,
                    'Stat': 'Average'
                }
            }
        ],
        StartTime=datetime(2023, 1, 1),
        EndTime=datetime(2023, 4, 2)
    )
    overall_jsons_list = []
    for metric_data in response["MetricDataResults"]:
        formatted = format_metric_data(metric_data)
        overall_jsons_list.append(formatted)
    
    json_names = ["invocations.json", "avg_duration.json", "highest_duration.json", "errors.json"]
    for i in range(len(json_names)):
        write_json(overall_jsons_list[i], json_names[i])
    


def format_metric_data(metric_data):
    """
    Reformat data from Cloudwatch metric data into QuickSight ingestible format.
    Takes in Cloudwatch response json

    """
    json_list = []
    for value, timestamp in zip(metric_data["Values"], metric_data["Timestamps"]):
        data_point = {
            "value": value,
            "time":timestamp.isoformat()
        }
        json_list.append(data_point)
    return json_list


def write_json(new_json_data, fname):
    """
    Upload reformatted json data into S3 sambaash dashboard bucket
    Takes in reformatted json data and corresponding json filename
    Will write over all existing content in json file 

    """
    s3_client = boto3.client("s3")
    s3 = boto3.resource("s3")
    S3_BUCKET_NAME = 'nus-sambaash'
    object_key = "skills-engine-dashboard/" + fname
    # create json if not exists
    try:
        s3_client.get_object(
            Bucket=S3_BUCKET_NAME,
            Key=object_key
        )
    except s3_client.exceptions.NoSuchKey:
        file = io.BytesIO()
        s3_client.upload_fileobj(file, S3_BUCKET_NAME, object_key)
            
    s3_object = s3.Object(S3_BUCKET_NAME, object_key)
    json_byte = json.dumps(new_json_data).encode('UTF-8')
    s3_object.put(
        Body=json_byte, 
        ContentType='application/json'
    )


def update_json(new_skills_data, fname):
    """
    Join new json data from skills extraction model with previous json data
    and update json file tracking skills and corresponding frequency saved in S3

    """
    s3_client = boto3.client("s3")
    s3 = boto3.resource("s3")
    S3_BUCKET_NAME = 'nus-sambaash'
    object_key = "skills-engine-dashboard/" + fname
    try:
        # get object and update with new data then re-upload
        s3_object = s3_client.get_object(
                        Bucket=S3_BUCKET_NAME,
                        Key=object_key
        )
        data = s3_object['Body'].read()
        data_str = data.decode('utf8')
        original_json = json.loads(data_str)
        updated = format_skills_data(original_json, new_skills_data)
        s3_object = s3.Object(S3_BUCKET_NAME, object_key)
        json_byte = json.dumps(updated).encode('UTF-8')
        s3_object.put(
            Body=json_byte, 
            ContentType='application/json'
        )
    except s3_client.exceptions.NoSuchKey:
        # create object, add new data then upload
        new_json = {}
        for skill, freq in new_skills_data["resume"]["skills"].items():
            new_json[skill] = freq
        file = io.BytesIO()
        file.write(json.dumps(new_json).encode('UTF-8'))
        s3_client.upload_fileobj(file, S3_BUCKET_NAME, object_key)
    

def format_skills_data(original_data, new_skills_data):
    """
    Helper function of update_json to combine output data from skills extraction
    model in Lambda function with previous results into dictionary tracking skills frequency

    """
    skills = new_skills_data["resume"]["skills"]
    for skill, freq in skills.items():
        if skill not in original_data:
            original_data[skill] = freq
        else:
            original_data[skill] += freq
    return original_data

#     client = boto3.client("lambda")
#     response = client.invoke(
#         FunctionName="arn:aws:lambda:ap-southeast-1:630471847671:function:skills_extract"
#     )
#     print(response)

def lambda_handler(event, context):
    # run invoke function when skills extraction Lambda function run
    update_json(event["results"],"skills_freq.json")
    update_metric_data()

if __name__ == "__main__":
    event = []
    context = []
    lambda_handler(event, context)